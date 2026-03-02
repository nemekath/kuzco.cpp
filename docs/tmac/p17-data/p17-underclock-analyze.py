#!/usr/bin/env python3
"""P17 Underclocking Test: Kernel-level timing at baseline vs reduced memory clock.

Compares per-kernel timing between auto clocks (mclk=1249MHz) and underclocked
(mclk=772MHz, -38%) to determine whether kernels are memory-bound or FMA-latency-bound.

If a kernel is memory-bound: duration increases ~proportionally to bandwidth reduction.
If a kernel is FMA-latency-bound: duration unchanged (memory is not the bottleneck).
"""

import csv
import re
from collections import defaultdict

TYPE_MAP = {
    2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 39: "MXFP4",
}

def parse_kernel_name(name):
    fused = "fused_glu" in name
    has_bias = False
    if not fused:
        m = re.search(r"ggml_type\)\d+,\s*(true|false)", name)
        if m:
            has_bias = m.group(1) == "true"
    m = re.search(r"ggml_type\)(\d+)", name)
    type_id = int(m.group(1)) if m else -1
    qtype = TYPE_MAP.get(type_id, f"type_{type_id}")
    return qtype, has_bias, fused

def is_tmac(name):
    return "tmac" in name

def is_non_tmac_kernel(name):
    """Identify non-T-MAC GPU kernels (attention, RMS norm, etc.)"""
    return not is_tmac(name) and ("Kernel_Name" not in name)

def parse_trace(filepath):
    kernels = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Kernel_Name", "")
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            dur = end - start
            grid = int(row["Grid_Size_X"])

            if is_tmac(name):
                qtype, has_bias, is_fused = parse_kernel_name(name)
                ftype = "fused" if is_fused else ("bias" if has_bias else "unfused")
                kernels.append({
                    "name": f"{qtype} {ftype} [grd={grid}]",
                    "category": "tmac",
                    "qtype": qtype,
                    "ftype": ftype,
                    "grid": grid,
                    "duration_ns": dur,
                })
            else:
                # Non-tmac kernel — classify by name
                short = name.split("(")[0].split("<")[0].strip()
                if len(short) > 50:
                    short = short[:50]
                kernels.append({
                    "name": short,
                    "category": "non-tmac",
                    "qtype": "",
                    "ftype": "",
                    "grid": grid,
                    "duration_ns": dur,
                })
    return kernels

def group_kernels(kernels, tmac_only=False):
    """Group by name, return {name: [durations_ns]}"""
    groups = defaultdict(list)
    for k in kernels:
        if tmac_only and k["category"] != "tmac":
            continue
        groups[k["name"]].append(k["duration_ns"])
    return groups

def analyze_pair(label, baseline_file, slow_file):
    base = parse_trace(baseline_file)
    slow = parse_trace(slow_file)

    base_groups = group_kernels(base, tmac_only=True)
    slow_groups = group_kernels(slow, tmac_only=True)

    # Also get non-tmac totals
    base_non_tmac = sum(k["duration_ns"] for k in base if k["category"] != "tmac")
    slow_non_tmac = sum(k["duration_ns"] for k in slow if k["category"] != "tmac")
    base_tmac_total = sum(k["duration_ns"] for k in base if k["category"] == "tmac")
    slow_tmac_total = sum(k["duration_ns"] for k in slow if k["category"] == "tmac")

    print(f"\n{'='*90}")
    print(f"  {label} — Kernel-Level Underclocking Analysis")
    print(f"  Baseline: mclk=auto (~1249MHz)  |  Underclocked: mclk=772MHz (-38%)")
    print(f"{'='*90}")

    print(f"\n  {'Kernel':<40} {'N':>3} {'Base μs':>9} {'Slow μs':>9} {'Slowdown':>9} {'Classification':>20}")
    print(f"  {'-'*92}")

    # Sort by baseline total time
    sorted_names = sorted(base_groups.keys(),
                         key=lambda n: -sum(base_groups[n]))

    for name in sorted_names:
        b_durs = base_groups[name]
        s_durs = slow_groups.get(name, [])
        if not s_durs:
            continue

        n = len(b_durs)
        b_avg = (sum(b_durs) / n) / 1000  # μs
        s_avg = (sum(s_durs) / len(s_durs)) / 1000

        slowdown = (s_avg - b_avg) / b_avg * 100

        if slowdown > 25:
            cls = "MEMORY-BOUND"
        elif slowdown > 10:
            cls = "Partially mem-bound"
        elif slowdown > 3:
            cls = "Weakly mem-sensitive"
        else:
            cls = "FMA-LATENCY-BOUND"

        print(f"  {name:<40} {n:>3} {b_avg:>9.2f} {s_avg:>9.2f} {slowdown:>+8.1f}% {cls:>20}")

    # Totals
    print(f"\n  {'─'*92}")
    b_total_us = base_tmac_total / 1000
    s_total_us = slow_tmac_total / 1000
    tmac_slowdown = (s_total_us - b_total_us) / b_total_us * 100

    b_non_us = base_non_tmac / 1000
    s_non_us = slow_non_tmac / 1000
    non_slowdown = (s_non_us - b_non_us) / b_non_us * 100 if b_non_us > 0 else 0

    print(f"  {'T-MAC total':<40} {'':>3} {b_total_us:>9.1f} {s_total_us:>9.1f} {tmac_slowdown:>+8.1f}%")
    print(f"  {'Non-T-MAC total (attn/norm/etc)':<40} {'':>3} {b_non_us:>9.1f} {s_non_us:>9.1f} {non_slowdown:>+8.1f}%")

    all_base = (base_tmac_total + base_non_tmac) / 1000
    all_slow = (slow_tmac_total + slow_non_tmac) / 1000
    all_slowdown = (all_slow - all_base) / all_base * 100
    print(f"  {'ALL kernels':<40} {'':>3} {all_base:>9.1f} {all_slow:>9.1f} {all_slowdown:>+8.1f}%")


def main():
    import os
    d = os.path.dirname(os.path.abspath(__file__))

    analyze_pair(
        "Q4_K_M (memory-bound reference)",
        os.path.join(d, "p17-uc2-high-Q4KM_kernel_trace.csv"),
        os.path.join(d, "p17-uc2-low-Q4KM_kernel_trace.csv"),
    )
    analyze_pair(
        "IQ2_XXS (hypothesized FMA-latency-bound)",
        os.path.join(d, "p17-uc2-high-IQ2XXS_kernel_trace.csv"),
        os.path.join(d, "p17-uc2-low-IQ2XXS_kernel_trace.csv"),
    )

    # Cross-comparison summary
    print(f"\n{'='*90}")
    print(f"  CROSS-TYPE UNDERCLOCKING SENSITIVITY SUMMARY")
    print(f"{'='*90}")
    print(f"\n  Memory clock reduced by ~38% (1249 → 772 MHz)")
    print(f"  Both runs: manual perf level, sclk pinned to max (2482 MHz)")
    print(f"  If memory-bound: expect ~+38% slowdown (proportional)")
    print(f"  If FMA-latency-bound: expect ~0% slowdown (insensitive)")
    print()

    for label, bf, sf in [
        ("Q4_K_M", os.path.join(d, "p17-uc2-high-Q4KM_kernel_trace.csv"),
         os.path.join(d, "p17-uc2-low-Q4KM_kernel_trace.csv")),
        ("IQ2_XXS", os.path.join(d, "p17-uc2-high-IQ2XXS_kernel_trace.csv"),
         os.path.join(d, "p17-uc2-low-IQ2XXS_kernel_trace.csv")),
    ]:
        base = parse_trace(bf)
        slow = parse_trace(sf)

        # Get fused tmac kernels only (the most important ones)
        b_fused = [k for k in base if k["category"] == "tmac" and k["ftype"] == "fused"]
        s_fused = [k for k in slow if k["category"] == "tmac" and k["ftype"] == "fused"]

        b_unfused = [k for k in base if k["category"] == "tmac" and k["ftype"] == "unfused"]
        s_unfused = [k for k in slow if k["category"] == "tmac" and k["ftype"] == "unfused"]

        b_bias = [k for k in base if k["category"] == "tmac" and k["ftype"] == "bias"]
        s_bias = [k for k in slow if k["category"] == "tmac" and k["ftype"] == "bias"]

        b_non = [k for k in base if k["category"] != "tmac"]
        s_non = [k for k in slow if k["category"] != "tmac"]

        def avg_us(lst):
            return sum(k["duration_ns"] for k in lst) / len(lst) / 1000 if lst else 0

        def slowdown_pct(b_list, s_list):
            b = avg_us(b_list)
            s = avg_us(s_list)
            return (s - b) / b * 100 if b > 0 else 0

        print(f"  {label}:")
        if b_fused:
            sd = slowdown_pct(b_fused, s_fused)
            print(f"    Fused SwiGLU:  {avg_us(b_fused):>8.2f} → {avg_us(s_fused):>8.2f} μs  ({sd:>+6.1f}%)")
        if b_bias:
            sd = slowdown_pct(b_bias, s_bias)
            print(f"    Bias-fused:    {avg_us(b_bias):>8.2f} → {avg_us(s_bias):>8.2f} μs  ({sd:>+6.1f}%)")
        if b_unfused:
            sd = slowdown_pct(b_unfused, s_unfused)
            print(f"    Unfused:       {avg_us(b_unfused):>8.2f} → {avg_us(s_unfused):>8.2f} μs  ({sd:>+6.1f}%)")
        if b_non:
            sd = slowdown_pct(b_non, s_non)
            print(f"    Non-T-MAC:     {avg_us(b_non):>8.2f} → {avg_us(s_non):>8.2f} μs  ({sd:>+6.1f}%)")
        print()


if __name__ == "__main__":
    main()
