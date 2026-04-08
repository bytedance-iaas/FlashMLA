#!/usr/bin/env python3
"""Benchmark q8kv8 sparse prefill kernel using CUDA events (no kineto).
Avoids the kineto hang seen with d_qk=576.
"""
import os, sys, time, math

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import flash_mla

from lib import TestParam
import lib
import ref
from test_flash_mla_sparse_prefill import build_performance_cases


def _quantize_fp8(x):
    scale = (x.abs().max().float().clamp_min(1e-6) / 448.0).to(torch.float32)
    return (x.float() / scale.float()).to(torch.float8_e4m3fn), scale


def bench_cuda_events(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = max(1, len(times) * 3 // 4)
    return sum(times[:n]) / n


@torch.inference_mode()
def run_bench(p: TestParam):
    if p.seed == -1:
        p.seed = 42
    t = lib.generate_testcase(p)
    torch.cuda.synchronize()
    q8, qs = _quantize_fp8(t.q)
    kv8, kvs = _quantize_fp8(t.kv)

    def fn():
        return flash_mla.flash_mla_sparse_q8kv8_fwd(
            q8, kv8, t.indices, sm_scale=t.sm_scale,
            q_scale=qs, kv_scale=kvs, d_v=p.d_v,
            attn_sink=t.attn_sink, topk_length=t.topk_length,
        )

    ms = bench_cuda_events(fn)
    tag = f"sq={p.s_q:<5d} skv={p.s_kv:<7d} hq={p.h_q:<4d} dqk={p.d_qk:<4d} topk={p.topk:<5d}"
    print(f"[q8]  {tag}  {ms*1e3:7.0f} us")
    return ms


@torch.inference_mode()
def run_bench_q16(p: TestParam):
    if p.seed == -1:
        p.seed = 42
    t = lib.generate_testcase(p)
    torch.cuda.synchronize()

    def fn():
        return flash_mla.flash_mla_sparse_fwd(
            t.q, t.kv, t.indices, sm_scale=t.sm_scale,
            d_v=p.d_v, attn_sink=t.attn_sink, topk_length=t.topk_length,
        )

    ms = bench_cuda_events(fn)
    tag = f"sq={p.s_q:<5d} skv={p.s_kv:<7d} hq={p.h_q:<4d} dqk={p.d_qk:<4d} topk={p.topk:<5d}"
    print(f"[q16] {tag}  {ms*1e3:7.0f} us")
    return ms


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    cases = build_performance_cases()

    level = os.getenv("FLASHMLA_Q8KV8_PERF_LEVEL", "full").strip().lower()
    if level == "quick":
        cases = [
            next(c for c in cases if c.d_qk == 576 and c.h_q == 128 and c.topk == 2048),
            next(c for c in cases if c.d_qk == 512 and c.h_q == 64 and c.topk == 512),
            next(c for c in cases if c.d_qk == 512 and c.h_q == 128 and c.topk == 1024),
        ]

    results = []
    for i, c in enumerate(cases):
        if i > 0:
            time.sleep(0.3)
        ms_q8 = run_bench(c)
        ms_q16 = run_bench_q16(c)
        delta = (ms_q8 - ms_q16) / ms_q16 * 100 if ms_q16 > 0 else 0
        sign = "+" if delta >= 0 else ""
        print(f"       -> delta: {sign}{delta:.1f}%")
        results.append((c, ms_q8, ms_q16, delta))

    print("\n=== Summary ===")
    print(f"{'config':<55s} {'q8 us':>8s} {'q16 us':>8s} {'delta':>8s}")
    for c, q8, q16, d in results:
        tag = f"dqk={c.d_qk} hq={c.h_q} topk={c.topk} skv={c.s_kv}"
        sign = "+" if d >= 0 else ""
        print(f"{tag:<55s} {q8*1e3:8.0f} {q16*1e3:8.0f} {sign}{d:6.1f}%")
