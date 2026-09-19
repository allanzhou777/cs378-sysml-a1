"""Quick plots for the report, built from the numbers already gathered in
report_assets/*.txt. Local, no GPU needed -- just pandas + matplotlib.

  pip install pandas matplotlib
  python3 report_assets/make_plots.py
"""
import pandas as pd
import matplotlib.pyplot as plt

# ---- Part 2: naive baseline runtime + HBM traffic vs N (graded_tests_output.txt) ----
part2 = pd.DataFrame({
    "N": [512, 1024, 2048, 4096],
    "runtime_ms": [7.31, 23.92, 89.49, 365.25],
    "HBM_MB": [28.3, 107.0, 415.2, 1635.8],
}).set_index("N")

part2["HBM_MB"].plot(kind="bar", title="Part 2: naive baseline HBM traffic vs N")
plt.ylabel("MB moved")
plt.tight_layout()
plt.show()

part2["runtime_ms"].plot(kind="bar", title="Part 2: naive baseline runtime vs N")
plt.ylabel("ms")
plt.tight_layout()
plt.show()

# ---- Part 3: HBM traffic, naive vs flash (graded_tests_output.txt) ----
part3 = pd.DataFrame({
    "N": [512, 1024, 2048, 4096],
    "naive_MB": [28.3, 107.0, 415.2, 1635.8],
    "flash_MB": [4.2, 8.5, 16.9, 33.8],
}).set_index("N")

part3.plot(kind="bar", logy=True,
           title="Part 3: HBM traffic, naive vs flash (log scale)")
plt.ylabel("MB moved")
plt.tight_layout()
plt.show()

(part3["naive_MB"] / part3["flash_MB"]).plot(
    kind="line", marker="o", title="Part 3: naive/flash HBM traffic ratio vs N")
plt.ylabel("ratio (x)")
plt.tight_layout()
plt.show()

# ---- Part 4c: graded sweep, H=16 fixed (part4c_sweep_output.txt) ----
part4c = pd.DataFrame({
    "N": [64, 128, 512, 1024, 4096],
    "Br=Bc=8": [0.008, 0.009, 0.010, 0.010, 0.010],
    "Br=Bc=16": [0.012, 0.015, 0.019, 0.019, 0.019],
    "Br=Bc=32": [0.012, 0.016, 0.020, 0.021, 0.021],
}).set_index("N")

part4c.plot(kind="line", marker="o", logx=True,
            title="Part 4c: TFLOPs/s vs N by block size (H=16)")
plt.ylabel("TFLOPs/s")
plt.tight_layout()
plt.show()

# ---- Part 4b: custom H sweep (part4b_regime_sweep.txt) ----
# long axis: one row per (H, N, block) triple actually measured.
part4b = pd.DataFrame([
    (32, 512, 8, 0.0193), (32, 512, 16, 0.0350), (32, 512, 32, 0.0379),
    (32, 1024, 8, 0.0198), (32, 1024, 16, 0.0368), (32, 1024, 32, 0.0399),
    (32, 2048, 8, 0.0199), (32, 2048, 16, 0.0373), (32, 2048, 32, 0.0406),
    (32, 4096, 8, 0.0200), (32, 4096, 16, 0.0377), (32, 4096, 32, 0.0410),
    (32, 8192, 8, 0.0201), (32, 8192, 16, 0.0379), (32, 8192, 32, 0.0412),
    (32, 16384, 32, 0.0413),
    (128, 512, 8, 0.0468), (128, 512, 16, 0.0542), (128, 512, 32, 0.0463),
    (128, 1024, 8, 0.0493), (128, 1024, 16, 0.0576), (128, 1024, 32, 0.0470),
    (128, 2048, 8, 0.0503), (128, 2048, 16, 0.0590), (128, 2048, 32, 0.0448),
    (128, 4096, 8, 0.0509), (128, 4096, 16, 0.0603), (128, 4096, 32, 0.0453),
    (128, 8192, 8, 0.0511), (128, 8192, 16, 0.0602), (128, 8192, 32, 0.0456),
    (128, 16384, 16, 0.0609), (128, 16384, 32, 0.0457),
], columns=["H", "N", "block", "tflops"])

for H in (32, 128):
    pivot = part4b[part4b["H"] == H].pivot(index="N", columns="block", values="tflops")
    pivot.plot(kind="line", marker="o", logx=True,
               title=f"Part 4b: TFLOPs/s vs N by block size (H={H})")
    plt.ylabel("TFLOPs/s")
    plt.tight_layout()
    plt.show()
