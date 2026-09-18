"""Part 1: reference attention and the shared test harness.

You implement the two TODO blocks:
  * ASSIGN1_1_1  -- attention_reference(): softmax(Q K^T / sqrt(d)) V
  * ASSIGN1_1_2  -- the max-error comparison inside check_against_reference()

Everything downstream (grade_naive.py, grade_flash.py, bench/sweep.py) imports
check_against_reference() from this file, so get it right first. NumPy is the
ground truth; kernels run in float32 unless a test says otherwise.
"""
import numpy as np

FP32_TOL = 1e-4
FP16_TOL = 1e-2


def attention_reference(Q, K, V, causal=False):
    """Exact attention. Q, K, V: (B, H, N, d) float arrays. Returns (B, H, N, d).

    S = Q @ K^T / sqrt(d);  optionally causal-masked;  P = softmax(S);  O = P @ V.
    Uses the numerically stable softmax (subtract the row max).
    """
    # BEGIN ASSIGN1_1_1
    d = Q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    S = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale  # (B, H, N, N)

    if causal:
        N = S.shape[-1]
        future = np.triu(np.ones((N, N), dtype=bool), k=1)  # j > i is masked
        S = np.where(future, -np.inf, S)

    S = S - np.max(S, axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / np.sum(P, axis=-1, keepdims=True)
    O = np.matmul(P, V)
    return O.astype(Q.dtype)
    # END ASSIGN1_1_1


def check_against_reference(kernel_out, Q, K, V, causal=False, dtype="fp32",
                            label=""):
    """Compare a kernel's output to attention_reference and print PASS/FAIL.

    Returns True on pass. Tolerance is 1e-4 for fp32, 1e-2 for fp16.
    """
    ref = attention_reference(Q, K, V, causal=causal)
    tol = FP16_TOL if dtype == "fp16" else FP32_TOL
    # BEGIN ASSIGN1_1_2
    max_err = float(np.max(np.abs(kernel_out.astype(np.float64) -
                                   ref.astype(np.float64))))
    ok = max_err <= tol
    # END ASSIGN1_1_2
    tag = f"{label} " if label else ""
    print(f"{tag}causal={int(causal)} dtype={dtype} "
          f"max_err={max_err:.2e} tol={tol:.0e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def make_inputs(B, H, N, d, seed=2024, dtype=np.float32):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, N, d)).astype(dtype)
    K = rng.standard_normal((B, H, N, d)).astype(dtype)
    V = rng.standard_normal((B, H, N, d)).astype(dtype)
    return Q, K, V
