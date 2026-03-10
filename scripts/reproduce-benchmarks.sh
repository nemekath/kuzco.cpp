#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# reproduce-benchmarks.sh — Reproduce kuzco.cpp T-MAC performance claims
# ═══════════════════════════════════════════════════════════════════════
#
# This script lets you verify T-MAC speedup numbers on your own hardware.
# It runs paired interleaved benchmarks (T-MAC ↔ stock) and computes
# speedup with confidence intervals.
#
# Usage:
#   scripts/reproduce-benchmarks.sh <model.gguf> [OPTIONS]
#
# Example:
#   # Download a model (any Q4_K_M from HuggingFace)
#   huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
#     --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" --local-dir models/
#
#   # Run benchmark
#   scripts/reproduce-benchmarks.sh models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
#
# Output: CSV file + summary with speedup and 95% confidence interval.
#
# Requirements:
#   - AMD RDNA3 GPU (RX 7900 XTX validated)
#   - ROCm installed (tested with 7.1)
#   - kuzco.cpp built with: cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
#   - HIP_VISIBLE_DEVICES set to exclude iGPU if present
#
# What this measures:
#   tg128 = 128-token generation (the workload T-MAC accelerates).
#   T-MAC: normal run. Stock: same binary with GGML_HIP_NO_TMAC=1.
#   Paired interleaved = tmac,stock,tmac,stock,... to cancel thermal drift.
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
export LC_NUMERIC=C  # Force decimal point for all locales

# ─── Defaults ──────────────────────────────────────────────────────────
N=5              # Measured runs per variant (after warmup)
WARMUP=1         # Warmup runs to discard
BENCH="./build/bin/llama-bench"
NGL=99           # GPU layers (-ngl)
METRIC="tg128"   # What to measure

# ─── Argument parsing ──────────────────────────────────────────────────
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)       N="$2"; shift 2 ;;
        --warmup)  WARMUP="$2"; shift 2 ;;
        --bench)   BENCH="$2"; shift 2 ;;
        --ngl)     NGL="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,/^# ═══.*═══$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)        echo "Unknown option: $1 (try --help)"; exit 1 ;;
        *)         MODEL="$1"; shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf> [--n N] [--ngl LAYERS]"
    echo "Try --help for details."
    exit 1
fi

# ─── Validation ────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: llama-bench not found at: $BENCH"
    echo "Build first: cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 && cmake --build build -j\$(nproc)"
    exit 1
fi

# Check for RDNA3
if ! rocminfo 2>/dev/null | grep -q "gfx11"; then
    echo "WARNING: No RDNA3 GPU (gfx11xx) detected. T-MAC requires RDNA3."
    echo "         Results will show 0% speedup (both variants use stock kernels)."
fi

# ─── Environment snapshot ─────────────────────────────────────────────
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
HIP_VER=$(hipcc --version 2>/dev/null | grep -oP 'HIP version: \K\S+' || echo "unknown")
GPU_NAME=$(rocminfo 2>/dev/null | grep -oP 'Marketing Name:\s+\K.*' | head -1 || echo "unknown")
KERNEL_VER=$(uname -r)
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
MODEL_BASE=$(basename "$MODEL")

CSV_FILE="reproduce-${MODEL_BASE%.gguf}-$(date +%Y%m%d-%H%M%S).csv"

echo "═══════════════════════════════════════════════════════════════"
echo " kuzco.cpp Benchmark Reproduction"
echo "═══════════════════════════════════════════════════════════════"
echo " Model:   $MODEL_BASE"
echo " GPU:     $GPU_NAME"
echo " ROCm:    $ROCM_VER"
echo " Commit:  $COMMIT"
echo " Runs:    N=$N (+ $WARMUP warmup)"
echo " Metric:  $METRIC"
echo " NGL:     $NGL"
echo " Output:  $CSV_FILE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── CSV header ────────────────────────────────────────────────────────
{
    echo "# date=$(date -Iseconds),rocm=$ROCM_VER,hip=$HIP_VER,gpu=$GPU_NAME,kernel=$KERNEL_VER,commit=$COMMIT"
    echo "model,metric,run,variant,tokens_per_sec"
} > "$CSV_FILE"

# ─── Benchmark function ───────────────────────────────────────────────
# Extracts avg_ts from llama-bench CSV output.
# llama-bench writes CSV to stderr, so we merge with 2>&1.
# Uses $(NF-1) to handle GPU names containing commas (e.g., "RX 7900 XTX, RX 7900 XTX").
run_bench() {
    local env_prefix="$1"
    local args="-m $MODEL -ngl $NGL -r 1 -o csv"

    case "$METRIC" in
        pp512)  args="$args -p 512 -n 0" ;;
        tg128)  args="$args -p 0 -n 128" ;;
    esac

    local result
    result=$(env $env_prefix $BENCH $args 2>&1 | awk -F',' '
        /^"/ { gsub(/"/, "", $(NF-1)); print $(NF-1); exit }
    ')

    if [[ -z "$result" || "$result" == "0" ]]; then
        echo "ERROR: Failed to extract throughput from llama-bench" >&2
        return 1
    fi
    echo "$result"
}

# ─── Run paired interleaved benchmark ─────────────────────────────────
TOTAL=$((WARMUP + N))
TMAC_RUNS=()
STOCK_RUNS=()

for i in $(seq 1 $TOTAL); do
    is_warmup=""
    run_num=$((i - WARMUP))
    if (( i <= WARMUP )); then
        is_warmup=" (warmup, discarded)"
        run_num="W$i"
    fi

    echo -n "  Run $i/$TOTAL — T-MAC:$is_warmup "
    t=$(run_bench "")
    echo "$t t/s"

    echo -n "  Run $i/$TOTAL — Stock:$is_warmup "
    s=$(run_bench "GGML_HIP_NO_TMAC=1")
    echo "$s t/s"

    if (( i > WARMUP )); then
        TMAC_RUNS+=("$t")
        STOCK_RUNS+=("$s")
        echo "$MODEL_BASE,$METRIC,$run_num,tmac,$t" >> "$CSV_FILE"
        echo "$MODEL_BASE,$METRIC,$run_num,stock,$s" >> "$CSV_FILE"
    fi
    echo ""
done

# ─── Compute statistics ───────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo " Results"
echo "═══════════════════════════════════════════════════════════════"

awk -v n="$N" '
BEGIN {
    # Read T-MAC values
    for (i = 1; i <= n; i++) t[i] = ARGV[i]
    # Read Stock values
    for (i = 1; i <= n; i++) s[i] = ARGV[n + i]

    # Means
    sum_t = 0; sum_s = 0
    for (i = 1; i <= n; i++) { sum_t += t[i]; sum_s += s[i] }
    mean_t = sum_t / n
    mean_s = sum_s / n

    # Paired differences and speedup
    for (i = 1; i <= n; i++) d[i] = t[i] - s[i]
    sum_d = 0
    for (i = 1; i <= n; i++) sum_d += d[i]
    mean_d = sum_d / n

    # SD of differences
    ss = 0
    for (i = 1; i <= n; i++) ss += (d[i] - mean_d)^2
    sd_d = (n > 1) ? sqrt(ss / (n - 1)) : 0

    # t-statistic and approximate p-value
    se = (n > 1) ? sd_d / sqrt(n) : 0
    t_stat = (se > 0) ? mean_d / se : 999

    # 95% CI (t critical values for df=4..19, two-tailed)
    # Precomputed from t-distribution
    split("12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228 2.201 2.179 2.160 2.145 2.131", tcrit)
    df = n - 1
    tc = (df >= 1 && df <= 15) ? tcrit[df] : 1.96

    ci_low = mean_d - tc * se
    ci_high = mean_d + tc * se

    speedup = (mean_s > 0) ? 100 * (mean_t - mean_s) / mean_s : 0
    ci_low_pct = (mean_s > 0) ? 100 * ci_low / mean_s : 0
    ci_high_pct = (mean_s > 0) ? 100 * ci_high / mean_s : 0

    printf " T-MAC mean:  %.2f t/s\n", mean_t
    printf " Stock mean:  %.2f t/s\n", mean_s
    printf " Speedup:     %+.1f%%\n", speedup
    printf " 95%% CI:      [%+.1f%%, %+.1f%%]\n", ci_low_pct, ci_high_pct
    printf " t-statistic: %.2f (df=%d)\n", t_stat, df
    printf " N=%d paired runs (interleaved)\n", n

    if (ci_low_pct > 0.5)
        printf "\n VERDICT: SIGNIFICANT speedup (CI lower bound > +0.5%%)\n"
    else if (ci_high_pct < -0.5)
        printf "\n VERDICT: REGRESSION (CI upper bound < -0.5%%)\n"
    else
        printf "\n VERDICT: NOT SIGNIFICANT (CI includes zero)\n"
}' "${TMAC_RUNS[@]}" "${STOCK_RUNS[@]}"

echo ""
echo " Raw data: $CSV_FILE"
echo "═══════════════════════════════════════════════════════════════"
