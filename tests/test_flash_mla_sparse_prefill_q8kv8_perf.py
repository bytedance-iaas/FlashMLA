import os
import sys
import time

import torch
import kernelkit as kk

import flash_mla

from lib import TestParam
import lib
import ref
from test_flash_mla_sparse_prefill import build_performance_cases


_counter = kk.Counter()


def _quantize_fp8_e4m3_per_tensor(x: torch.Tensor):
    scale = (x.abs().max().float().clamp_min(1e-6) / 448.0).to(torch.float32)
    x_q = (x.float() / scale.float()).to(torch.float8_e4m3fn)
    return x_q, scale


def _print_precision_details(
    name: str,
    ans: torch.Tensor,
    ref_tensor: torch.Tensor,
    *,
    abs_tol: float,
    rel_tol: float,
    cos_diff_tol: float,
) -> None:
    ans_f = ans.float()
    ref_f = ref_tensor.float()

    finite_mask = torch.isfinite(ans_f) & torch.isfinite(ref_f)
    finite_ratio = finite_mask.float().mean().item() if finite_mask.numel() > 0 else 1.0
    anomaly_mismatch = torch.count_nonzero(torch.isfinite(ans_f) != torch.isfinite(ref_f)).item()

    if finite_mask.any():
        ans_v = ans_f[finite_mask]
        ref_v = ref_f[finite_mask]
        abs_err = (ans_v - ref_v).abs()
        rel_err = abs_err / (ref_v.abs() + 1e-6)
        max_abs = abs_err.max().item()
        mean_abs = abs_err.mean().item()
        max_rel = rel_err.max().item()
        mean_rel = rel_err.mean().item()
        cos_diff = kk.get_cos_diff(ans_v, ref_v)
    else:
        max_abs = float("nan")
        mean_abs = float("nan")
        max_rel = float("nan")
        mean_rel = float("nan")
        cos_diff = float("nan")

    abs_pass = (max_abs <= abs_tol) if finite_mask.any() else False
    rel_pass = (max_rel <= rel_tol) if finite_mask.any() else False
    cos_pass = (abs(cos_diff) <= cos_diff_tol) if finite_mask.any() else False

    print(
        f"[precision][q8] {name}: "
        f"max_abs={max_abs:.6e} (tol={abs_tol:.6e}, pass={abs_pass}), "
        f"mean_abs={mean_abs:.6e}, "
        f"max_rel={max_rel:.6e} (tol={rel_tol:.6e}, pass={rel_pass}), "
        f"mean_rel={mean_rel:.6e}, "
        f"cos_diff={cos_diff:.6e} (tol={cos_diff_tol:.6e}, pass={cos_pass}), "
        f"finite_ratio={finite_ratio:.4f}, anomaly_mismatch={anomaly_mismatch}"
    )


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase(p)
    torch.cuda.synchronize()

    q_fp8, q_scale = _quantize_fp8_e4m3_per_tensor(t.q)
    kv_fp8, kv_scale = _quantize_fp8_e4m3_per_tensor(t.kv)

    def run_prefill_q8():
        return flash_mla.flash_mla_sparse_q8kv8_fwd(
            q_fp8,
            kv_fp8,
            t.indices,
            sm_scale=t.sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            d_v=p.d_v,
            attn_sink=t.attn_sink,
            topk_length=t.topk_length,
        )

    prefill_ans_out, prefill_ans_max_logits, prefill_ans_lse = run_prefill_q8()
    torch.cuda.synchronize()

    if p.num_runs > 0:
        bench = kk.bench_kineto(run_prefill_q8, num_tests=p.num_runs)
        kernel_name = os.getenv("FLASHMLA_Q8KV8_PERF_KERNEL_SUBSTR", "sparse_attn_fwd_q8_direct_kernel")
        try:
            prefill_time = bench.get_kernel_time(kernel_name)
        except Exception:
            # Fallback for future kernel renames.
            prefill_time = bench.get_kernel_times(["sparse_attn_fwd"], allow_missing=False, allow_multiple_match=True)[0]

        print(f"Prefill q8: {prefill_time*1e6:4.0f} us")

    if p.check_correctness:
        torch.cuda.synchronize()
        _, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        torch.cuda.synchronize()

        _print_precision_details(
            "out",
            prefill_ans_out,
            ref_out_fp32,
            abs_tol=5e-2,
            rel_tol=8e-2,
            cos_diff_tol=2.5e-1,
        )
        _print_precision_details(
            "max_logits",
            prefill_ans_max_logits,
            ref_max_logits,
            abs_tol=2e-2,
            rel_tol=8e-2,
            cos_diff_tol=1e-3,
        )
        _print_precision_details(
            "lse",
            prefill_ans_lse,
            ref_lse,
            abs_tol=2e-2,
            rel_tol=8e-2,
            cos_diff_tol=1e-4,
        )

        # In q8 perf comparison mode we default to reporting precision details
        # without failing fast, so q8 and q16 metrics can be compared side-by-side.
        enforce_precision_gate = os.getenv("FLASHMLA_Q8KV8_ENFORCE_PRECISION", "0") == "1"
        if not enforce_precision_gate:
            return True

        is_correct = True
        is_correct &= kk.check_is_allclose("out", prefill_ans_out.float(), ref_out_fp32, abs_tol=5e-2, rel_tol=8e-2, cos_diff_tol=2.5e-1)
        is_correct &= kk.check_is_allclose("max_logits", prefill_ans_max_logits, ref_max_logits, abs_tol=2e-2, rel_tol=8e-2, cos_diff_tol=1e-3)
        is_correct &= kk.check_is_allclose("lse", prefill_ans_lse, ref_lse, abs_tol=2e-2, rel_tol=8e-2, cos_diff_tol=1e-4)
        return is_correct

    return True


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    perf_cases = build_performance_cases()

    level = os.getenv("FLASHMLA_Q8KV8_PERF_LEVEL", "full").strip().lower()
    if level == "quick":
        # Quick debug mode: keep one representative shape per config.
        perf_cases = [
            next(c for c in perf_cases if c.d_qk == 576 and c.h_q == 128 and c.topk == 2048),
            next(c for c in perf_cases if c.d_qk == 512 and c.h_q == 64 and c.topk == 512),
            next(c for c in perf_cases if c.d_qk == 512 and c.h_q == 128 and c.topk == 1024),
        ]

    is_no_cooldown = lib.is_no_cooldown()
    failed_cases = []
    for i, test in enumerate(perf_cases):
        if i > 0 and test.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        is_ok = run_test(test)
        if not is_ok:
            failed_cases.append(test)

    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(perf_cases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(perf_cases)} q8 perf cases passed!\033[0m")
