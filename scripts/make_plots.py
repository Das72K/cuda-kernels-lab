#!/usr/bin/env python3
"""
make_plots.py: turns the raw benchmark CSVs into tables and plots.

Input : results/raw/run*/<program>.csv   (one folder per full run of scripts/run_all.sh)
Output: results/tables/*.md              (markdown tables)
        results/plots/*.png              (roofline, optimization ladders, transfers, size sweep)
        README.md                        (tables between <!-- BEGIN:name --> / <!-- END:name --> markers)

Statistics: each run reports the median of 15 trials per kernel. Here, per (level, kernel, n),
we take the MEDIAN ACROSS RUNS and report the min-max across runs as the range.
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")                 # no display needed (WSL, CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
RAW    = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"
PLOTS  = ROOT / "results" / "plots"
README = ROOT / "README.md"


# ================================================================== data
def load_runs() -> pd.DataFrame:
    frames = []
    for run_dir in sorted(RAW.glob("run*")):
        for csv in sorted(run_dir.glob("*.csv")):
            df = pd.read_csv(csv)
            df["run"] = run_dir.name
            frames.append(df)
    if not frames:
        raise SystemExit(f"No CSV files under {RAW}/run*/ - run scripts/run_all.sh first.")
    df = pd.concat(frames, ignore_index=True)
    # work per launch, recovered from each run's median throughput
    df["bytes"] = df["gbps"].fillna(0.0) * df["ms_median"] * 1e6
    df["flops"] = df["gflops"].fillna(0.0) * df["ms_median"] * 1e6
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["level", "kernel", "n"], sort=False)
             .agg(runs=("run", "nunique"),
                  ms=("ms_median", "median"),
                  ms_lo=("ms_median", "min"),
                  ms_hi=("ms_median", "max"),
                  bytes=("bytes", "median"),
                  flops=("flops", "median"),
                  pct_library=("pct_library", "median"),
                  max_err=("max_err", "max"),
                  passed=("pass", lambda s: bool((s == "PASS").all())))
             .reset_index())
    agg["gbps"]      = agg["bytes"] / (agg["ms"]    / 1e3) / 1e9
    agg["gbps_lo"]   = agg["bytes"] / (agg["ms_hi"] / 1e3) / 1e9
    agg["gbps_hi"]   = agg["bytes"] / (agg["ms_lo"] / 1e3) / 1e9
    agg["gflops"]    = agg["flops"] / (agg["ms"]    / 1e3) / 1e9
    agg["gflops_lo"] = agg["flops"] / (agg["ms_hi"] / 1e3) / 1e9
    agg["gflops_hi"] = agg["flops"] / (agg["ms_lo"] / 1e3) / 1e9
    return agg


def pick(agg, level, kernel=None, n=None):
    sel = agg[agg["level"] == level]
    if kernel is not None:
        sel = sel[sel["kernel"] == kernel]
    if n is not None:
        sel = sel[sel["n"] == n]
    return sel


def one(agg, level, kernel, n=None):
    sel = pick(agg, level, kernel, n)
    if len(sel) != 1:
        raise SystemExit(f"Expected one row for {level} / {kernel} / n={n}, found {len(sel)}")
    return sel.iloc[0]


def short(kernel: str) -> str:
    return kernel.split()[0]


# ================================================================== formatting
def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def rng(lo, hi, digits):
    return f"{lo:.{digits}f}–{hi:.{digits}f}"


def check(r):
    return "PASS" if r["passed"] else "**FAIL**"


# ================================================================== tables
def table_summary(agg, bw, peak, runs):
    n1 = pick(agg, "1_vector_add")["n"].max()
    l1 = one(agg, "1_vector_add", "GPU kernel", n1)
    l2 = pick(agg, "2_reduction").query("kernel in ['v4 first add on load', 'v5 warp shuffle']") \
                                 .sort_values("gbps").iloc[-1]
    l3 = one(agg, "3_matmul", "v4 float4 + transposed A")
    rows = [
        ["1 Vector add", f"{n1 / 1e6:.0f}M floats", "memory", f"{l1['gbps']:.0f} GB/s", "—",
         f"{100 * l1['gbps'] / bw:.0f}% of {bw:.0f} GB/s"],
        ["2 Reduction", f"{l2['n'] / 1e6:.1f}M ints", "memory", f"{short(l2['kernel'])}: {l2['gbps']:.0f} GB/s",
         f"{l2['pct_library']:.0f}% of CUB", f"{100 * l2['gbps'] / bw:.0f}% of {bw:.0f} GB/s"],
        ["3 SGEMM", "n = 2048, FP32", "compute", f"v4: {l3['gflops'] / 1e3:.2f} TFLOPS",
         f"{l3['pct_library']:.0f}% of cuBLAS", f"{100 * l3['gflops'] / peak:.0f}% of {peak / 1e3:.1f} TFLOPS"],
    ]
    text = md_table(["Level", "Workload", "Bound by", "Best own kernel", "vs library", "vs measured ceiling"], rows)
    text += (f"\nCeilings measured on this GPU by `bench/peak.cu`: DRAM {bw:.1f} GB/s (best read test), "
             f"FP32 {peak / 1e3:.2f} TFLOPS (FMA test). All values: median across {runs} full runs.\n")
    return text


def table_peaks(agg, spec_bw):
    rows = []
    for _, r in pick(agg, "0_peak").iterrows():
        if r["gbps"] > 0:
            rows.append([r["kernel"], f"{r['ms']:.3f}", rng(r["ms_lo"], r["ms_hi"], 3),
                         f"{r['gbps']:.1f} GB/s", f"{100 * r['gbps'] / spec_bw:.1f}%"])
        else:
            rows.append([r["kernel"], f"{r['ms']:.3f}", rng(r["ms_lo"], r["ms_hi"], 3),
                         f"{r['gflops'] / 1e3:.2f} TFLOPS", "—"])
    return md_table(["test", "ms (median)", "range across runs", "result", f"% of {spec_bw:.0f} GB/s spec"], rows)


def table_level1(agg):
    rows = []
    for n in sorted(pick(agg, "1_vector_add")["n"].unique()):
        cpu = one(agg, "1_vector_add", "CPU (1 core)", n)
        k   = one(agg, "1_vector_add", "GPU kernel", n)
        tp  = one(agg, "1_vector_add", "GPU total pageable", n)
        tn  = one(agg, "1_vector_add", "GPU total pinned", n)
        ok  = all(x["passed"] for x in (cpu, k, tp, tn))
        rows.append([f"{n:,}", f"{cpu['ms']:.4f}", f"{k['ms']:.4f}", f"{k['gbps']:.1f}",
                     f"{cpu['ms'] / k['ms']:.1f}×", f"{tp['ms']:.3f}", f"{cpu['ms'] / tp['ms']:.2f}×",
                     f"{tn['ms']:.3f}", f"{cpu['ms'] / tn['ms']:.2f}×", "PASS" if ok else "**FAIL**"])
    return md_table(["n", "CPU ms", "kernel ms", "kernel GB/s", "kernel vs CPU",
                     "total ms (pageable)", "vs CPU", "total ms (pinned)", "vs CPU", "check"], rows)


def table_level2(agg, bw):
    rows = []
    for _, r in pick(agg, "2_reduction").iterrows():
        gpu = r["kernel"] != "CPU (1 core)"
        rows.append([r["kernel"], f"{r['ms']:.4f}", rng(r["ms_lo"], r["ms_hi"], 4), f"{r['gbps']:.1f}",
                     f"{100 * r['gbps'] / bw:.1f}%" if gpu else "—",
                     f"{r['pct_library']:.1f}%" if gpu else "—", check(r)])
    return md_table(["version", "ms (median)", "range across runs", "GB/s",
                     "% of measured ceiling", "% of CUB", "check"], rows)


def table_level3(agg, peak):
    rows = []
    for _, r in pick(agg, "3_matmul").iterrows():
        cpu  = r["kernel"].startswith("CPU")
        tf32 = "TF32" in r["kernel"]
        rows.append([r["kernel"], f"{r['ms']:.3f}", rng(r["ms_lo"], r["ms_hi"], 3), f"{r['gflops']:.0f}",
                     "—" if (cpu or tf32) else f"{100 * r['gflops'] / peak:.1f}%",
                     "—" if cpu else f"{r['pct_library']:.1f}%", f"{r['max_err']:.1e}", check(r)])
    return md_table(["version", "ms (median)", "range across runs", "GFLOPS",
                     "% of FP32 ceiling", "% of cuBLAS", "max err", "check"], rows)


def table_sweep(agg):
    rows = []
    for n in sorted(pick(agg, "3_matmul_sweep")["n"].unique()):
        ref = one(agg, "3_matmul_sweep", "cuBLAS sgemm", n)
        v3  = one(agg, "3_matmul_sweep", "v3 register tiling 4x4", n)
        v4  = one(agg, "3_matmul_sweep", "v4 float4 + transposed A", n)
        ok  = all(x["passed"] for x in (ref, v3, v4))
        rows.append([n, f"{ref['gflops']:.0f}", f"{v3['gflops']:.0f} ({v3['pct_library']:.1f}%)",
                     f"{v4['gflops']:.0f} ({v4['pct_library']:.1f}%)", "PASS" if ok else "**FAIL**"])
    return md_table(["n", "cuBLAS GFLOPS", "v3 GFLOPS (% cuBLAS)", "v4 GFLOPS (% cuBLAS)", "check"], rows)


# ================================================================== plots
def plot_roofline(agg, bw, peak, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ai = np.logspace(-2, 3.5, 400)
    ax.plot(ai, np.minimum(ai * bw, peak), color="black", lw=2, label="measured roofline")
    ridge = peak / bw
    ax.axvline(ridge, color="grey", ls=":", lw=1)
    ax.text(ridge * 1.08, peak * 0.035, f"ridge point\n{ridge:.0f} FLOP/byte", fontsize=8, color="grey")
    ax.text(0.012, peak * 1.12, f"FP32 ceiling {peak / 1e3:.1f} TFLOPS (FMA test)", fontsize=8)
    ax.text(0.012, 0.012 * bw * 1.6, f"DRAM ceiling {bw:.0f} GB/s", fontsize=8, rotation=31)

    # Level 1: 1 FLOP per 12 bytes
    n1 = pick(agg, "1_vector_add")["n"].max()
    k = one(agg, "1_vector_add", "GPU kernel", n1)
    ax.scatter(1 / 12, n1 / (k["ms"] / 1e3) / 1e9, color="tab:blue", s=40, zorder=3, label="L1 vector add")

    # Level 2: 1 integer add per 4 bytes (operations counted like FLOPs)
    first = True
    for _, r in pick(agg, "2_reduction").iterrows():
        if r["kernel"].startswith("CPU"):
            continue
        y = r["n"] / (r["ms"] / 1e3) / 1e9
        ax.scatter(0.25, y, color="tab:orange", s=30, zorder=3, label="L2 reduction" if first else None)
        first = False
        if r["kernel"].startswith(("v1", "v3", "CUB")):
            ax.annotate(short(r["kernel"]), (0.25, y), xytext=(-18, -3), textcoords="offset points", fontsize=7)

    # Level 3: minimum-traffic (algorithmic) intensity 2n^3 / (3 * 4n^2) = n/6
    ai3 = 2048 / 6
    first = True
    for i, (_, r) in enumerate(pick(agg, "3_matmul").iterrows()):
        if r["kernel"].startswith("CPU") or "TF32" in r["kernel"]:
            continue
        ax.scatter(ai3, r["gflops"], color="tab:green", s=30, zorder=3, label="L3 SGEMM (FP32)" if first else None)
        first = False
        ax.annotate(short(r["kernel"]), (ai3, r["gflops"]), xytext=(6 if i % 2 else -24, -3),
                    textcoords="offset points", fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (operations per DRAM byte; L3: minimum-traffic estimate)")
    ax.set_ylabel("performance (GOP/s)")
    ax.set_title("Measured roofline, RTX 4050 Laptop GPU")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_chart(rows, value, lo, hi, ylabel, title, lines, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(rows))
    vals = rows[value].to_numpy()
    err = np.vstack([vals - rows[lo].to_numpy(), rows[hi].to_numpy() - vals])
    ax.bar(x, vals, yerr=err, capsize=3, color="tab:blue")
    for y, label in lines:
        ax.axhline(y, ls="--", color="black", lw=1)
        ax.text(len(x) - 0.45, y, label, ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x, rows["kernel"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transfers(agg, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for kernel, style in [("CPU (1 core)", "o-"), ("GPU kernel", "s-"),
                          ("GPU total pageable", "^-"), ("GPU total pinned", "v-")]:
        sel = pick(agg, "1_vector_add", kernel).sort_values("n")
        ax.plot(sel["n"], sel["ms"], style, label=kernel, ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (floats per array)")
    ax.set_ylabel("time (ms, median across runs)")
    ax.set_title("Level 1: the PCIe copies dominate end-to-end time")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(agg, peak, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for kernel, style in [("cuBLAS sgemm", "o-"), ("v4 float4 + transposed A", "s-"), ("v3 register tiling 4x4", "^-")]:
        sel = pick(agg, "3_matmul_sweep", kernel).sort_values("n")
        err = np.vstack([sel["gflops"] - sel["gflops_lo"], sel["gflops_hi"] - sel["gflops"]])
        ax.errorbar(sel["n"], sel["gflops"], yerr=err, fmt=style, capsize=3, label=kernel, ms=4)
    ax.axhline(peak, ls="--", color="black", lw=1)
    ax.text(512, peak * 1.01, f"FP32 ceiling {peak / 1e3:.1f} TFLOPS", fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (square matrices)")
    ax.set_ylabel("GFLOPS (median across runs)")
    ax.set_title("Level 3: SGEMM across sizes")
    ax.set_ylim(0, peak * 1.12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ================================================================== README injection
def inject(text, name, content):
    pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.S)
    if not pattern.search(text):
        print(f"README: no markers for '{name}', skipped")
        return text
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    df   = load_runs()
    agg  = aggregate(df)
    runs = df["run"].nunique()

    bw      = pick(agg, "0_peak")["gbps"].max()                  # best measured bandwidth
    peak    = one(agg, "0_peak", "FMA (FP32)")["gflops"]          # measured FP32 throughput
    spec_bw = (df["gbps"] * 100.0 / df["pct_peak_bw"]).dropna().median()   # driver spec, recovered from the CSVs

    tables = {
        "summary":      table_summary(agg, bw, peak, runs),
        "peaks":        table_peaks(agg, spec_bw),
        "level1":       table_level1(agg),
        "level2":       table_level2(agg, bw),
        "level3":       table_level3(agg, peak),
        "level3_sweep": table_sweep(agg),
    }
    for name, content in tables.items():
        (TABLES / f"{name}.md").write_text(content)

    plot_roofline(agg, bw, peak, PLOTS / "roofline.png")
    l2 = pick(agg, "2_reduction").query("not kernel.str.startswith('CPU')", engine="python")
    bar_chart(l2, "gbps", "gbps_lo", "gbps_hi", "GB/s (median, range across runs)",
              "Level 2: reduction versions vs the measured DRAM ceiling",
              [(bw, f"measured ceiling {bw:.0f} GB/s")], PLOTS / "level2_ladder.png")
    l3 = pick(agg, "3_matmul").query("not kernel.str.startswith('CPU')", engine="python")
    cublas = one(agg, "3_matmul", "cuBLAS sgemm (reference)")["gflops"]
    bar_chart(l3, "gflops", "gflops_lo", "gflops_hi", "GFLOPS (median, range across runs)",
              "Level 3: SGEMM optimization ladder (n = 2048)",
              [(peak, f"FP32 ceiling {peak / 1e3:.1f} TFLOPS"), (cublas, "cuBLAS FP32")],
              PLOTS / "level3_ladder.png")
    plot_transfers(agg, PLOTS / "level1_transfers.png")
    plot_sweep(agg, peak, PLOTS / "level3_sweep.png")

    text = README.read_text()
    for name, content in tables.items():
        text = inject(text, name, content)
    README.write_text(text)

    print(f"{runs} runs, {len(agg)} aggregated rows")
    print(f"ceilings: DRAM {bw:.1f} GB/s (spec {spec_bw:.0f}), FP32 {peak / 1e3:.2f} TFLOPS")
    print(tables["summary"])


if __name__ == "__main__":
    main()
