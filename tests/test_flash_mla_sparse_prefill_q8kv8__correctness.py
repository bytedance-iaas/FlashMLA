import dataclasses
import math
import os
import sys
from typing import List, Optional, Tuple

import pytest
import torch

import flash_mla


@dataclasses.dataclass
class Q8PrefillCase:
    s_q: int
    s_kv: int
    topk: int
    h_q: int
    d_qk: int
    d_v: int = 512
    have_attn_sink: bool = False
    have_topk_length: bool = False
    seed: int = 0


def is_sm90_supported(device=None) -> bool:
    cap = torch.cuda.get_device_capability(device)
    return cap[0] == 9 and torch.version.cuda >= "12.3"


def quantize_fp8_e4m3_per_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().max().float().clamp_min(1e-6) / 448.0).to(torch.float32)
    x_q = (x.float() / scale.float()).to(torch.float8_e4m3fn)
    return x_q, scale


def dequantize_fp8_e4m3_per_tensor(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x_q.float() * scale.float()).to(torch.bfloat16)


def reference_torch_prefill(
    s_q: int,
    s_kv: int,
    topk: int,
    d_qk: int,
    d_v: int,
    indices: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    sm_scale: float,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # indices: [1, s_q, 1, topk]
    idx = indices[0, :, 0, :].to(torch.int64)
    invalid = (idx < 0) | (idx >= s_kv)

    if topk_length is not None:
        arange = torch.arange(topk, device=idx.device).unsqueeze(0)
        invalid = invalid | (arange >= topk_length.view(-1, 1))

    qf = q[0].float()  # [s_q, h_q, d_qk]
    kvf = kv[0, :, 0, :].float()  # [s_kv, d_qk]

    gather_idx = idx.masked_fill(invalid, 0).flatten()
    kv_sel = torch.index_select(kvf, 0, gather_idx).view(s_q, topk, d_qk)  # [s_q, topk, d_qk]

    logits = torch.einsum("shd,std->sht", qf, kv_sel)
    logits.masked_fill_(invalid.unsqueeze(1), float("-inf"))
    logits = logits * sm_scale

    max_logits = logits.max(dim=-1).values

    if attn_sink is not None:
        # Match kernel behavior: denominator includes an optional sink term.
        sink = attn_sink.view(1, -1).float()  # [1, h_q]
        exp_sum = torch.exp(logits - max_logits.unsqueeze(-1)).sum(dim=-1)
        sink_term = torch.exp(sink - max_logits)
        denom = exp_sum + sink_term
        lse = torch.log(torch.clamp_min(denom, 1e-30)) + max_logits
        probs = torch.exp(logits - lse.unsqueeze(-1))
    else:
        lse = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

    out = torch.einsum("sht,std->shd", probs, kv_sel[:, :, :d_v]).to(torch.float32)

    lonely = torch.isneginf(lse)
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0
    lse[lonely] = float("inf")
    max_logits[lonely] = float("-inf")
    return max_logits, lse, out


def build_case_tensors(case: Q8PrefillCase):
    torch.manual_seed(case.seed)

    q = torch.randn((1, case.s_q, case.h_q, case.d_qk), dtype=torch.bfloat16, device="cuda") / 10
    kv = torch.randn((1, case.s_kv, 1, case.d_qk), dtype=torch.bfloat16, device="cuda") / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((1, case.s_q, 1, case.topk), case.s_kv, dtype=torch.int32, device="cuda")
    for s in range(case.s_q):
        near_mask = torch.randint(0, 32, (min(case.topk, case.s_kv),), device="cuda") < 31
        cur_indices = torch.randperm(case.s_kv, device="cuda")[: case.topk]
        cur_indices[near_mask] = torch.randint(
            max(0, case.s_kv - 20000),
            max(1, case.s_kv) - 1,
            (int(near_mask.sum().item()),),
            device="cuda",
        )
        if len(cur_indices) < case.topk:
            cur_indices = torch.cat(
                [cur_indices, torch.full((case.topk - len(cur_indices),), 2147480000, device="cuda")]
            )
        cur_indices = cur_indices[torch.randperm(case.topk, device="cuda")]
        indices[0, s, 0] = cur_indices

    topk_length: Optional[torch.Tensor]
    if case.have_topk_length:
        topk_length = torch.randint(
            low=max(1, case.topk // 2),
            high=case.topk + 1,
            size=(case.s_q,),
            device="cuda",
            dtype=torch.int32,
        )
        for s in range(case.s_q):
            cur_topk = int(topk_length[s].item())
            if cur_topk < case.topk:
                indices[0, s, 0, cur_topk:] = case.s_kv
    else:
        topk_length = None

    attn_sink = None
    if case.have_attn_sink:
        attn_sink = (torch.randn((case.h_q,), device="cuda", dtype=torch.float32) * 0.5) - 0.5

    q_fp8, q_scale = quantize_fp8_e4m3_per_tensor(q.squeeze(0))
    kv_fp8, kv_scale = quantize_fp8_e4m3_per_tensor(kv.squeeze(0))

    sm_scale = 1.0 / math.sqrt(case.d_qk)
    return q_fp8, kv_fp8, q_scale, kv_scale, indices, topk_length, attn_sink, sm_scale


def run_case(case: Q8PrefillCase) -> Tuple[bool, str]:
    q_fp8, kv_fp8, q_scale, kv_scale, indices, topk_length, attn_sink, sm_scale = build_case_tensors(case)

    out, max_logits, lse = flash_mla.flash_mla_sparse_q8kv8_fwd(
        q_fp8,
        kv_fp8,
        indices.squeeze(0),
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=case.d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

    q_ref = dequantize_fp8_e4m3_per_tensor(q_fp8, q_scale).unsqueeze(0)
    kv_ref = dequantize_fp8_e4m3_per_tensor(kv_fp8, kv_scale).unsqueeze(0)
    ref_max, ref_lse, ref_out = reference_torch_prefill(
        case.s_q,
        case.s_kv,
        case.topk,
        case.d_qk,
        case.d_v,
        indices,
        q_ref,
        kv_ref,
        sm_scale,
        topk_length,
        attn_sink,
    )

    ok = True
    msg = ""
    try:
        torch.testing.assert_close(out.float(), ref_out, atol=5e-2, rtol=8e-2)
        torch.testing.assert_close(max_logits.float(), ref_max, atol=2e-2, rtol=8e-2)
        torch.testing.assert_close(lse.float(), ref_lse, atol=2e-2, rtol=8e-2)
    except AssertionError as e:
        ok = False
        msg = str(e)
    return ok, msg


def build_quick_cases() -> List[Q8PrefillCase]:
    base = [
        Q8PrefillCase(1, 128, 128, 64, 512, seed=0),
        Q8PrefillCase(62, 592, 128, 64, 512, seed=1),
        Q8PrefillCase(213, 1840, 256, 64, 512, seed=2),
        Q8PrefillCase(1, 128, 128, 128, 512, seed=3),
        Q8PrefillCase(62, 592, 128, 128, 512, seed=4),
        Q8PrefillCase(213, 1840, 256, 128, 512, seed=5),
        Q8PrefillCase(1, 128, 128, 64, 576, seed=6),
        Q8PrefillCase(62, 592, 128, 64, 576, seed=7),
        Q8PrefillCase(213, 1840, 256, 64, 576, seed=8),
        Q8PrefillCase(1, 128, 128, 128, 576, seed=9),
        Q8PrefillCase(62, 592, 128, 128, 576, seed=10),
        Q8PrefillCase(213, 1840, 256, 128, 576, seed=11),
    ]
    features = [
        Q8PrefillCase(62, 592, 128, 64, 512, have_attn_sink=a, have_topk_length=t, seed=12 + i)
        for i, (a, t) in enumerate([(False, False), (False, True), (True, False), (True, True)])
    ]
    return base + features


def build_full_cases() -> List[Q8PrefillCase]:
    cases: List[Q8PrefillCase] = []
    seed = 0
    for d_qk in [512, 576]:
        for h_q in [64, 128]:
            for s_kv, topk in [
                (128, 128),
                (256, 256),
                (512, 512),
                (592, 128),
                (1840, 256),
                (1592, 384),
                (1521, 512),
                (95, 128),
                (153, 256),
                (114, 384),
            ]:
                for s_q in [1, 62, 213]:
                    cases.append(Q8PrefillCase(s_q, s_kv, topk, h_q, d_qk, seed=seed))
                    seed += 1

    for d_qk in [512, 576]:
        for h_q in [64, 128]:
            for s_kv, topk in [
                (592, 128),
                (1840, 256),
                (1592, 384),
                (1521, 512),
                (95, 128),
                (153, 256),
                (114, 384),
            ]:
                for s_q in [62, 213]:
                    for have_attn_sink in [False, True]:
                        for have_topk_length in [False, True]:
                            cases.append(
                                Q8PrefillCase(
                                    s_q,
                                    s_kv,
                                    topk,
                                    h_q,
                                    d_qk,
                                    have_attn_sink=have_attn_sink,
                                    have_topk_length=have_topk_length,
                                    seed=seed,
                                )
                            )
                            seed += 1
    return cases


def run_matrix(level: str) -> None:
    if level == "full":
        cases = build_full_cases()
    else:
        cases = build_quick_cases()

    print(f"q8kv8 kernel-only test level: {level}, total cases: {len(cases)}")
    failed = []
    for case in cases:
        print("================")
        print(f"Running on {case}")
        ok, msg = run_case(case)
        if not ok:
            failed.append((case, msg))

    if failed:
        print(f"{len(failed)} / {len(cases)} q8kv8 cases failed")
        for case, msg in failed[:10]:
            print(f"- {case}")
            print(msg)
        raise AssertionError(f"q8kv8 matrix failed: {len(failed)} cases")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_sm90_supported(), reason="SM90 required for FP8 sparse prefill")
@torch.inference_mode()
def test_q8kv8_kernel_only_matrix():
    level = os.getenv("FLASHMLA_Q8KV8_TEST_LEVEL", "quick").strip().lower()
    run_matrix(level)


if __name__ == "__main__":
    if not torch.cuda.is_available() or not is_sm90_supported():
        print("SM90 + CUDA is required")
        sys.exit(0)

    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(torch.device("cuda:0"))
    level = os.getenv("FLASHMLA_Q8KV8_TEST_LEVEL", "quick").strip().lower()
    run_matrix(level)
    print(f"All {level} q8kv8 kernel-only cases passed!")
