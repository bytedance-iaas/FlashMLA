import os
import time
import sys

import torch
import kernelkit as kk

from lib import TestParam
import lib
import ref

_counter = kk.Counter()

# Shared with q8 perf test to guarantee q8/q16 perf use the same representative shapes.
PERFORMANCE_CASE_TEMPLATES = [
    # V3.2
    (576, 128, 2048, [8192, 32768, 65536, 98304, 131072]),
    # MODEL1 CONFIG1
    (512, 64, 512, [8192, 32768, 49152, 65536]),
    # MODEL1 CONFIG2
    (512, 128, 1024, [8192, 32768, 49152, 65536]),
]


def build_performance_cases() -> list[TestParam]:
    return [
        TestParam(s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, have_attn_sink=True)
        for (d_qk, h_q, topk, s_kv_list) in PERFORMANCE_CASE_TEMPLATES
        for s_q in [4096]
        for s_kv in s_kv_list
    ]


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
        f"[precision][q16] {name}: "
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
    
    def run_prefill():
        return lib.run_flash_mla_sparse_fwd(p, t, False)
    
    prefill_ans_out, prefill_ans_max_logits, prefill_ans_lse = run_prefill()
    torch.cuda.synchronize()

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol(p, t)
        prefill_ans_time = kk.bench_kineto(run_prefill, num_tests=p.num_runs).get_kernel_time("sparse_attn_fwd")
        prefill_flops = flops_and_mem_vol.fwd_flop/prefill_ans_time/1e12
        prefill_mem_bw = flops_and_mem_vol.fwd_mem_vol/prefill_ans_time/1e12
        print(f"Prefill:  {prefill_ans_time*1e6:4.0f} us, {prefill_flops:6.1f} TFlops, {prefill_mem_bw:4.2f} TBps")

    if p.check_correctness:
        torch.cuda.synchronize()
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        torch.cuda.synchronize()

        should_print_precision = os.getenv("FLASHMLA_PRINT_PRECISION_DETAILS", "0") == "1" or p.num_runs > 0
        if should_print_precision:
            _print_precision_details(
                "out",
                prefill_ans_out,
                ref_out_fp32,
                abs_tol=8e-4,
                rel_tol=3.01 / 128,
                cos_diff_tol=7e-6,
            )
            _print_precision_details(
                "max_logits",
                prefill_ans_max_logits,
                ref_max_logits,
                abs_tol=1e-6,
                rel_tol=2.01 / 65536,
                cos_diff_tol=1e-7,
            )
            _print_precision_details(
                "lse",
                prefill_ans_lse,
                ref_lse,
                abs_tol=1e-6,
                rel_tol=2.01 / 65536,
                cos_diff_tol=1e-7,
            )

        is_correct = True
        is_correct &= kk.check_is_allclose("out", prefill_ans_out.float(), ref_out_fp32, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        is_correct &= kk.check_is_allclose("max_logits", prefill_ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        is_correct &= kk.check_is_allclose("lse", prefill_ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)

        return is_correct
    else:
        return True


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    correctness_cases = [
        # Regular shapes
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [
            1, 62, 213
        ]
    ]

    correctness_cases_with_features = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, have_attn_sink=have_attn_sink, have_topk_length=have_topk_length, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [62, 213]
        for have_sink_lse in [False, True]
        for have_attn_sink in [False, True]
        for have_topk_length in [False, True]
    ]

    corner_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=True, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_q, s_kv, topk in [
            (1, 128, 128),
            (1, 256, 256),
            (1234, 4321, 4096),
            (4096, 2048, 2048)
        ]
    ] + [
        # In these cases, some blocks may not have any valid topk indices
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (32, 2048),
            (64, 8192)
        ]
        for s_q in [1, 1024]
    ] + [
        # In this testcase, s_q is really large, so we cannot put it on the second dimension of grid shape
        TestParam(70000, 256, 256, h_q=h_q, check_correctness=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
    ]

    performance_cases = build_performance_cases()

    testcases = correctness_cases + correctness_cases_with_features + corner_cases + performance_cases

    is_no_cooldown = lib.is_no_cooldown()
    failed_cases = []
    for test in testcases:
        if test != testcases[0] and test.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)
    
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")

