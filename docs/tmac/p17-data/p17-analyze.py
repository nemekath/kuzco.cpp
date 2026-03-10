#!/usr/bin/env python3
"""P17 HW Counter Analysis: T-MAC kernel profiling across quant types.

Compares kernel-level timing for Q4_K_M (memory-bound ref), IQ3_S (transitional),
IQ2_XXS (hypothesized latency-bound) to validate classification.

Data sources:
- rocprofv3 --kernel-trace: per-kernel nanosecond timestamps
- rocprof v1 --pmc SQ_WAVES: wave counts + kernel metadata (VGPRs, LDS, SGPRs)
"""

import csv
import re
import statistics
from collections import defaultdict

# Correct ggml type enum values (from ggml.h)
TYPE_MAP = {
    2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 39: "MXFP4",
}

# Bits per weight for ratio comparisons
BPW = {
    "Q4_0": 4.50, "Q4_K": 4.50, "Q5_K": 5.50, "Q6_K": 6.56,
    "Q3_K": 3.44, "IQ3_S": 3.44, "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31, "IQ3_XXS": 3.06, "IQ4_XS": 4.25,
}

def parse_kernel_name(name):
    """Extract (qtype_str, has_bias, is_fused) from kernel name."""
    fused = "fused_glu" in name
    # For non-fused kernels: template param is <QType, HAS_BIAS>
    has_bias = False
    if not fused:
        # Look for true/false after the type
        m = re.search(r"ggml_type\)\d+,\s*(true|false)", name)
        if m:
            has_bias = m.group(1) == "true"

    m = re.search(r"ggml_type\)(\d+)", name)
    type_id = int(m.group(1)) if m else -1
    qtype = TYPE_MAP.get(type_id, f"type_{type_id}")
    return qtype, has_bias, fused

def parse_v3_kernel_trace(filepath):
    """Parse rocprofv3 kernel trace CSV. Returns list of dicts."""
    results = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Kernel_Name", "")
            if "tmac" not in name:
                continue
            qtype, has_bias, is_fused = parse_kernel_name(name)
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            results.append({
                "qtype": qtype, "has_bias": has_bias, "is_fused": is_fused,
                "duration_ns": end - start,
                "grid": int(row["Grid_Size_X"]),
                "wg": int(row["Workgroup_Size_X"]),
                "lds": int(row["LDS_Block_Size"]),
                "vgpr": int(row["VGPR_Count"]),
                "sgpr": int(row["SGPR_Count"]),
            })
    return results

def parse_v1_metadata(filepath):
    """Parse rocprof v1 CSV. Returns list of dicts."""
    results = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("KernelName", "")
            if "tmac" not in name:
                continue
            qtype, has_bias, is_fused = parse_kernel_name(name)
            results.append({
                "qtype": qtype, "has_bias": has_bias, "is_fused": is_fused,
                "grid": int(row["grd"]), "wg": int(row["wgr"]),
                "lds": int(row["lds"]), "vgpr": int(row["arch_vgpr"]),
                "sgpr": int(row["sgpr"]), "sq_waves": int(row["SQ_WAVES"]),
            })
    return results

def variant_key(d):
    """Create a grouping key for kernel variants."""
    ftype = "fused" if d["is_fused"] else ("bias" if d["has_bias"] else "unfused")
    return f"{d['qtype']}|{ftype}|{d['grid']}"

def analyze_model(label, v3_data, v1_data):
    """Analyze one model quant type."""
    print(f"\n{'='*78}")
    print(f"  {label} — Kernel Dispatch Summary")
    print(f"{'='*78}")

    # Group by variant
    groups = defaultdict(lambda: {"count": 0, "total_ns": 0, "durations": [],
                                   "grid": 0, "wg": 0, "lds": 0, "vgpr": 0,
                                   "sgpr": 0, "sq_waves": 0, "qtype": "",
                                   "is_fused": False, "has_bias": False})
    for k in v3_data:
        key = variant_key(k)
        g = groups[key]
        g["count"] += 1
        g["total_ns"] += k["duration_ns"]
        g["durations"].append(k["duration_ns"])
        for f in ["grid", "wg", "lds", "vgpr", "sgpr", "qtype", "is_fused", "has_bias"]:
            g[f] = k[f]

    # Add SQ_WAVES from v1
    v1_groups = defaultdict(list)
    for k in v1_data:
        v1_groups[variant_key(k)].append(k["sq_waves"])
    for key, waves in v1_groups.items():
        if key in groups:
            groups[key]["sq_waves"] = sum(waves) // len(waves)

    # Sort by total time (descending)
    sorted_keys = sorted(groups.keys(), key=lambda k: -groups[k]["total_ns"])

    total_ns = sum(g["total_ns"] for g in groups.values())
    total_count = sum(g["count"] for g in groups.values())

    WARMUP_SKIP = 4  # skip first 4 dispatches (~2 layers) for warmup correction

    print(f"\n{'Kernel Variant':<40} {'N':>4} {'Avg μs':>8} {'Stddev':>8} {'CV%':>6} "
          f"{'WU-Avg':>8} {'WU-CV%':>6} {'VGPR':>5}")
    print("-" * 98)

    for key in sorted_keys:
        g = groups[key]
        durs_us = [d / 1000 for d in g["durations"]]
        avg_us = statistics.mean(durs_us)
        stdev = statistics.stdev(durs_us) if len(durs_us) > 1 else 0
        cv = (stdev / avg_us * 100) if avg_us > 0 else 0

        # Warmup correction: skip first dispatches (cold Infinity Cache)
        skip = min(WARMUP_SKIP, len(durs_us) // 2)
        steady = durs_us[skip:]
        wu_avg = statistics.mean(steady) if steady else avg_us
        wu_stdev = statistics.stdev(steady) if len(steady) > 1 else 0
        wu_cv = (wu_stdev / wu_avg * 100) if wu_avg > 0 else 0

        g["avg_us"] = avg_us
        g["stdev"] = stdev
        g["cv"] = cv
        g["wu_avg"] = wu_avg
        g["wu_cv"] = wu_cv

        ftype = "fused" if g["is_fused"] else ("bias" if g["has_bias"] else "unfused")
        name = f"{g['qtype']} {ftype} [grd={g['grid']}, wg={g['wg']}]"
        print(f"{name:<40} {g['count']:>4} {avg_us:>8.2f} {stdev:>8.2f} {cv:>5.1f}% "
              f"{wu_avg:>8.2f} {wu_cv:>5.1f}% {g['vgpr']:>5}")

    print(f"\n{'TOTAL T-MAC kernel time':<40} {total_count:>4} {'':>8} {total_ns/1000:>9.1f}")

    return groups

def cross_type_analysis(all_groups):
    """Compare kernel timings across quant types."""
    print(f"\n{'='*78}")
    print(f"  CROSS-TYPE TIMING ANALYSIS")
    print(f"{'='*78}")

    # Summary table
    print(f"\n{'Model Quant':<12} {'T-MAC μs':>10} {'Dispatches':>10} {'Main Quant':>12}")
    print("-" * 46)
    for label in ["Q4_K_M", "IQ3_S", "IQ2_XXS"]:
        groups = all_groups[label]
        total_us = sum(g["total_ns"] for g in groups.values()) / 1000
        total_n = sum(g["count"] for g in groups.values())
        # Find the primary quant type (highest dispatch count, excluding output layer)
        main_qtype = "?"
        max_count = 0
        for key, g in groups.items():
            if g["count"] > max_count and g["grid"] < 1000000:  # exclude vocab output
                max_count = g["count"]
                main_qtype = g["qtype"]
        print(f"{label:<12} {total_us:>10.0f} {total_n:>10} {main_qtype:>12}")

    # Core comparison: match kernels by structural role (grid size + fused/unfused)
    # and compare timings across quant types
    print(f"\n{'─'*78}")
    print(f"  Per-kernel duration comparison (IQ type vs Q4_K_M reference)")
    print(f"{'─'*78}")

    ref_groups = all_groups["Q4_K_M"]

    for iq_label in ["IQ3_S", "IQ2_XXS"]:
        iq_groups = all_groups[iq_label]
        print(f"\n  ── {iq_label} vs Q4_K_M ──")

        # Find the primary IQ type used in this model
        iq_main_type = None
        for key, g in iq_groups.items():
            if g["qtype"] not in ("Q4_K", "Q5_K", "Q6_K"):
                iq_main_type = g["qtype"]
                break

        if iq_main_type is None:
            print(f"  (no IQ kernels found)")
            continue

        iq_bpw = BPW.get(iq_main_type, 0)
        q4k_bpw = BPW.get("Q4_K", 4.5)
        mem_ratio = iq_bpw / q4k_bpw if q4k_bpw > 0 else 1.0

        print(f"  Primary IQ type: {iq_main_type} ({iq_bpw:.2f} bpw)")
        print(f"  Memory-bound prediction: {mem_ratio:.2f}x Q4_K time (byte ratio)")
        print()

        # Match IQ kernels to Q4_K equivalents by grid size
        print(f"  {'Role':<22} {'Grid':>8} {iq_label+' μs':>10} {'Q4_K μs':>10} {'Ratio':>7} {'WU-Ratio':>9} {'Predicted':>9} {'Classification':>24}")
        print(f"  {'-'*103}")

        # Collect IQ kernels
        for iq_key, iq_g in sorted(iq_groups.items(), key=lambda x: -x[1]["total_ns"]):
            if iq_g["qtype"] == iq_main_type:
                ftype = "fused" if iq_g["is_fused"] else ("bias" if iq_g["has_bias"] else "unfused")
                iq_avg_us = iq_g.get("avg_us", (iq_g["total_ns"] / iq_g["count"]) / 1000)
                iq_wu_avg = iq_g.get("wu_avg", iq_avg_us)

                # Find matching Q4_K kernel by grid size and fused/bias type
                matched = False
                for ref_key, ref_g in ref_groups.items():
                    if ref_g["qtype"] == "Q4_K" and ref_g["grid"] == iq_g["grid"] and \
                       ref_g["is_fused"] == iq_g["is_fused"] and ref_g["has_bias"] == iq_g["has_bias"]:
                        ref_avg_us = ref_g.get("avg_us", (ref_g["total_ns"] / ref_g["count"]) / 1000)
                        ref_wu_avg = ref_g.get("wu_avg", ref_avg_us)
                        ratio = iq_avg_us / ref_avg_us if ref_avg_us > 0 else float('inf')
                        wu_ratio = iq_wu_avg / ref_wu_avg if ref_wu_avg > 0 else float('inf')

                        if ratio < mem_ratio * 0.85:
                            cls = "BETTER than mem-bound"
                        elif ratio < mem_ratio * 1.15:
                            cls = "~memory-bound"
                        elif ratio < 1.0:
                            cls = "transitional"
                        else:
                            cls = "LATENCY-BOUND"

                        role = f"{ftype} (N={iq_g['count']})"
                        print(f"  {role:<22} {iq_g['grid']:>8} {iq_avg_us:>10.2f} {ref_avg_us:>10.2f} "
                              f"{ratio:>6.2f}x {wu_ratio:>8.2f}x {mem_ratio:>8.2f}x {cls:>24}")
                        matched = True
                        break

                if not matched:
                    role = f"{ftype} (N={iq_g['count']})"
                    print(f"  {role:<22} {iq_g['grid']:>8} {iq_avg_us:>10.2f} {'n/a':>10} {'':>7} {'':>9} {'':>9} {'no Q4_K match':>24}")

    # Resource usage comparison
    print(f"\n{'─'*78}")
    print(f"  Resource Usage Comparison (primary quant kernels)")
    print(f"{'─'*78}")
    print(f"\n  {'Type':<12} {'VGPR (unfused)':>15} {'VGPR (fused)':>15} {'LDS (unfused)':>15} {'LDS (fused)':>15}")
    print(f"  {'-'*72}")

    for label in ["Q4_K_M", "IQ3_S", "IQ2_XXS"]:
        groups = all_groups[label]
        vgpr_u = vgpr_f = lds_u = lds_f = "-"
        for key, g in groups.items():
            # Only look at primary type (not mixed-quant layers)
            if label == "Q4_K_M" and g["qtype"] != "Q4_K":
                continue
            if label == "IQ3_S" and g["qtype"] not in ("IQ3_S",):
                continue
            if label == "IQ2_XXS" and g["qtype"] not in ("IQ2_XXS",):
                continue

            if g["is_fused"]:
                vgpr_f = str(g["vgpr"])
                lds_f = str(g["lds"])
            elif not g["has_bias"]:
                vgpr_u = str(g["vgpr"])
                lds_u = str(g["lds"])

        print(f"  {label:<12} {vgpr_u:>15} {vgpr_f:>15} {lds_u:>15} {lds_f:>15}")

    # Duration distribution per quant type (how time is split between kernel variants)
    print(f"\n{'─'*78}")
    print(f"  Time Distribution by Kernel Variant")
    print(f"{'─'*78}")

    for label in ["Q4_K_M", "IQ3_S", "IQ2_XXS"]:
        groups = all_groups[label]
        total_ns = sum(g["total_ns"] for g in groups.values())
        print(f"\n  {label}:")
        for key in sorted(groups.keys(), key=lambda k: -groups[k]["total_ns"]):
            g = groups[key]
            pct = 100 * g["total_ns"] / total_ns
            ftype = "fused" if g["is_fused"] else ("bias" if g["has_bias"] else "unfused")
            bar = "█" * int(pct / 2)
            print(f"    {g['qtype']:<10} {ftype:<8} {pct:5.1f}% {bar}")


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

    types = {
        "Q4_K_M": (os.path.join(script_dir, "p17-ktrace-Q4KM_kernel_trace.csv"),
                    os.path.join(script_dir, "p17-v1-Q4_K_M.csv")),
        "IQ3_S":  (os.path.join(script_dir, "p17-ktrace-IQ3_S_kernel_trace.csv"),
                    os.path.join(script_dir, "p17-v1-IQ3_S.csv")),
        "IQ2_XXS": (os.path.join(script_dir, "p17-ktrace-IQ2_XXS_kernel_trace.csv"),
                     os.path.join(script_dir, "p17-v1-IQ2_XXS.csv")),
    }

    all_groups = {}
    for label, (v3_file, v1_file) in types.items():
        v3_data = parse_v3_kernel_trace(v3_file)
        v1_data = parse_v1_metadata(v1_file)
        all_groups[label] = analyze_model(label, v3_data, v1_data)

    cross_type_analysis(all_groups)
    variance_analysis(all_groups)


def variance_analysis(all_groups):
    """Variance asymmetry analysis."""
    print(f"\n{'='*78}")
    print(f"  VARIANCE ASYMMETRY ANALYSIS")
    print(f"  (CV comparison as independent classification evidence)")
    print(f"{'='*78}")

    print(f"\n  {'Type':<12} {'Kernel':<28} {'CV%':>6} {'WU-CV%':>7} {'Warmup Δ':>9}")
    print(f"  {'-'*66}")

    for label in ["Q4_K_M", "IQ3_S", "IQ2_XXS"]:
        groups = all_groups[label]
        primary_type = {"Q4_K_M": "Q4_K", "IQ3_S": "IQ3_S", "IQ2_XXS": "IQ2_XXS"}[label]

        for key in sorted(groups.keys(), key=lambda k: -groups[k]["total_ns"]):
            g = groups[key]
            if g["qtype"] != primary_type:
                continue
            ftype = "fused" if g["is_fused"] else ("bias" if g["has_bias"] else "unfused")
            cv = g.get("cv", 0)
            wu_cv = g.get("wu_cv", 0)
            wu_avg = g.get("wu_avg", g.get("avg_us", 0))
            avg = g.get("avg_us", 0)
            warmup_pct = (avg - wu_avg) / wu_avg * 100 if wu_avg > 0 else 0
            name = f"{ftype} [grd={g['grid']}]"
            print(f"  {label:<12} {name:<28} {cv:>5.1f}% {wu_cv:>6.1f}% {warmup_pct:>+8.1f}%")

    print(f"\n  Interpretation:")
    print(f"  - High CV (>10%) = memory-bound (cache hit/miss variance dominates)")
    print(f"  - Low CV (<5%)   = latency-bound (deterministic FMA chain dominates)")
    print(f"  - Warmup Δ > 0   = first dispatches slower (cold Infinity Cache)")


if __name__ == "__main__":
    main()
