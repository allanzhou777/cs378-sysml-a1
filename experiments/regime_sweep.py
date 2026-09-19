"""Part 4b: custom regime benchmark.

Usage (from attn_kernel_assignment/):
  python3 ../experiments/regime_sweep.py --H 32 128 --N 512 1024 2048 4096 8192 16384 --block 8 16 32 --timeout 120
"""
import argparse
import ctypes
import os
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSIGN_ROOT = os.path.join(ROOT, "..", "attn_kernel_assignment")
d = 64


def attn_flops(B, H, N, d):
    return 2 * (2.0 * N * N * d) * B * H


def build(block):
    build_dir = os.path.join(ASSIGN_ROOT, "build")
    os.makedirs(build_dir, exist_ok=True)
    so = os.path.join(build_dir, f"flash_regime_block{block}.so")
    src = os.path.join(ASSIGN_ROOT, "src", "flash_attention_kernel.cu")
    subprocess.run(
        ["nvcc", "-O2", f"-DBLOCK={block}", "-o", so, "--shared", src,
         "-Xcompiler", "-fPIC"],
        check=True, cwd=ASSIGN_ROOT)
    lib = ctypes.CDLL(so)
    lib.launch_flash_attn_fw.argtypes = [
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, 1, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p]
    lib.launch_flash_attn_fw.restype = None
    return lib


def time_ms(lib, B, H, N, iters=3):
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((B, H, N, d), dtype=np.float32)
    K = rng.standard_normal((B, H, N, d), dtype=np.float32)
    V = rng.standard_normal((B, H, N, d), dtype=np.float32)
    O = np.zeros_like(Q)
    args = (Q.ravel(), K.ravel(), V.ravel(), O.reshape(-1), B, H, N, d, 0, None)
    lib.launch_flash_attn_fw(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        lib.launch_flash_attn_fw(*args)
    return (time.perf_counter() - t0) * 1e3 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--N", type=int, nargs="+",
                     default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--block", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--B", type=int, default=1)
    args = ap.parse_args()

    libs = {b: build(b) for b in args.block}
    print(f"{'H':>5} {'N':>7} " + " | ".join(f"Br=Bc={b:<4}" for b in args.block)
          + "   (TFLOPs/s)")
    print("-" * (14 + 12 * len(args.block)))
    for H in args.H:
        for N in args.N:
            if any(N % b != 0 for b in args.block):
                print(f"{H:>5} {N:>7}  (skipped: N not a multiple of all block sizes)")
                continue
            cells = []
            for b in args.block:
                t0 = time.perf_counter()
                try:
                    ms = time_ms(libs[b], args.B, H, N)
                    tflops = attn_flops(args.B, H, N, d) / (ms * 1e-3) / 1e12
                    cells.append(f"{tflops:>8.4f}")
                except Exception:
                    cells.append(f"{'ERR':>8}")
                print(f"    ...H={H} N={N} block={b} took "
                      f"{time.perf_counter()-t0:.1f}s", flush=True)
            print(f"{H:>5} {N:>7} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
