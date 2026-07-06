# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark TorchAO MX-FP4 weight-only linear dispatch on ROCm (gfx950).

On ROCm, ``MXTensor`` weight-only MX-FP4 matmul is routed to AITER's
``gemm_a16wfp4`` (bf16 activation x fp4 weight) via
``_mxfp4_rocm_dispatch``.  This script benchmarks that dispatch path against
a direct AITER ``gemm_a16wfp4`` baseline and reports:

  * ``iters_per_second``        -- torchao dispatch throughput
  * ``speed_ratio``             -- torchao / aiter (parity target: >= 0.95)
  * ``rel_err``                 -- max abs output difference vs aiter

Usage (on an MI355X / gfx950 node with aiter installed)::

    python benchmarks/mx_formats/benchmark_mxfp4_rocm.py
    python benchmarks/mx_formats/benchmark_mxfp4_rocm.py --M 2048 --K 8192 --N 8192
"""

import argparse

import torch

from torchao.prototype.mx_formats.mx_tensor import (
    KernelPreference,
    MXTensor,
    to_mx,
)

torch.manual_seed(0)

DEFAULT_M = 1024
DEFAULT_K = 4096
DEFAULT_N = 4096
WARMUP_ITERS = 5
TIMED_ITERS = 20


def _bench(fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS) -> float:
    """Return iters/sec for ``fn`` (a zero-arg callable)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    elapsed_s = t0.elapsed_time(t1) / 1000.0
    return iters / elapsed_s


def main(M: int, K: int, N: int) -> None:
    assert torch.version.hip, "this benchmark targets ROCm (gfx950)"
    dev = "cuda"

    # bf16 activation (M, K) and weight (N, K)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w_hp = torch.randn(N, K, dtype=torch.bfloat16, device=dev)

    # MX-FP4 weight-only quantization (block_size=32, no swizzled scales)
    w_mxfp4 = to_mx(
        w_hp,
        torch.float4_e2m1fn_x2,
        block_size=32,
        kernel_preference=KernelPreference.AUTO,
        is_swizzled_scales=False,
    )
    print(
        f"[bench] w_mxfp4 qdata={tuple(w_mxfp4.qdata.shape)} "
        f"scale={tuple(w_mxfp4.scale.shape)} elem_dtype={w_mxfp4.elem_dtype}"
    )

    # --- torchao dispatch path: F.linear / matmul -> _addmm_mx_dispatch
    #     -> _mxfp4_rocm_dispatch -> aiter.gemm_a16wfp4 ---
    def torchao_dispatch():
        return torch.matmul(x, w_mxfp4.t())

    y_torchao = torchao_dispatch()
    print(f"[bench] torchao dispatch output: shape={tuple(y_torchao.shape)} "
          f"dtype={y_torchao.dtype} nan={torch.isnan(y_torchao).any().item()}")

    # --- AITER baseline: direct gemm_a16wfp4 ---
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    w_qdata = w_mxfp4.qdata.t().contiguous()  # (N, K//2) uint8
    w_scale = w_mxfp4.scale.t().contiguous().view(torch.uint8)  # (N, K//32) uint8

    def aiter_baseline():
        return gemm_a16wfp4(x, w_qdata, w_scale, dtype=torch.bfloat16)

    y_aiter = aiter_baseline()
    print(f"[bench] aiter baseline output: shape={tuple(y_aiter.shape)} "
          f"dtype={y_aiter.dtype} nan={torch.isnan(y_aiter).any().item()}")

    rel_err = (y_torchao - y_aiter).abs().max().item()
    torchao_ips = _bench(torchao_dispatch)
    aiter_ips = _bench(aiter_baseline)
    speed_ratio = torchao_ips / aiter_ips

    print(f"[bench] M={M} K={K} N={N} dtype=torch.bfloat16")
    print(f"[bench] torchao_dispatch_ips={torchao_ips:.2f}")
    print(f"[bench] aiter_baseline_ips={aiter_ips:.2f}")
    print(f"[bench] rel_err_torchao_vs_aiter={rel_err:.6f}")
    print(f"[bench] speed_ratio_torchao_over_aiter={speed_ratio:.4f}")
    print(f"iters_per_second: {torchao_ips:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    main(args.M, args.K, args.N)
