#!/usr/bin/env bash
# T-MAC regression test: interleaved A-B benchmark with statistical comparison.
# Usage: scripts/tmac-regression.sh [--models "model1 model2 ..."] [--n N] [--csv FILE]
#
# Requirements:
#   - Single llama-bench binary built with GGML_HIP_TMAC=ON.
#   - Stock baseline uses GGML_HIP_NO_TMAC=1 env var to disable T-MAC at runtime.
#   - Override binary path with BENCH env var.
#
# Exit code 1 on regression (CI-friendly).

set -euo pipefail
export LC_NUMERIC=C  # Force decimal point (not comma) for de_DE and similar locales

# Auto-detect and exclude iGPUs (e.g. gfx1036) to prevent segfaults
source "$(dirname "$0")/hip-gpu-guard.sh"

# ─── Defaults ───────────────────────────────────────────────────────────
N=${N:-5}
WARMUP=${WARMUP:-1}
BENCH="${BENCH:-./build/bin/llama-bench}"
CSV_FILE="${CSV_FILE:-tmac-regression-$(date +%Y%m%d-%H%M%S).csv}"
MULTIGPU=0

# Thresholds: regression flagged if T-MAC is slower by more than this percentage
THRESH_TG=2.0   # tg128: -2% = regression
THRESH_PP=3.0   # pp512: -3% = regression

# Paired t-test: p-value threshold for statistical significance
P_THRESH=0.05

# Default models (override with --models)
# Symlinks in models/ — create with: MODEL_DIR=/path/to/models scripts/setup-test-models.sh
MODELS=(
    "models/1B-Q4_K_M.gguf"
    "models/22B-Q4_K_M.gguf"
    "models/OLMoE-Q4_K_M.gguf"
    "models/1B-IQ3_S.gguf"              # IQ3_S: highest T-MAC gain (+34.4%)
    "models/1B-IQ4_XS.gguf"             # IQ4_XS: +11.1% (N=10)
    "models/1B-Q5_0.gguf"               # Q5_0: Q4_K fallback type (ne0%256!=0)
    "models/1B-Q5_K_M.gguf"             # Q5_K: sub-block parallel like Q4_K, 5-bit with qh high bits
    "models/1B-IQ3_XXS.gguf"            # IQ3_XXS: highest Dense gain (+36.6%), P16
    "models/1B-IQ2_XXS.gguf"            # IQ2_XXS: worst bug history (grid/block)
    "models/1B-IQ2_XS.gguf"             # IQ2_XS: +17.4%, P16
    "models/1B-IQ1_M.gguf"              # IQ1_M: 1.75 bpw, 17th type, v1.5
    "models/1B-Q3_K_L.gguf"             # Q3_K: completes K-quant family, P3
    "models/8B-Q5_K_M.gguf"             # 8B Q5_K_M: most common deployment size
    "models/OLMoE-IQ3_S.gguf"           # OLMoE IQ3_S: MoE+IQ coverage
    "models/OLMoE-IQ3_XXS.gguf"         # OLMoE IQ3_XXS: MoE+P16 validation, +28.2%
    # "models/Mixtral-8x7B-IQ3_S.gguf"    # MoE+IQ3_S: +45.4% dual GPU — requires HIP_VISIBLE_DEVICES=0,1
    # "models/GLM-4.7-Flash-Q4_K_M.gguf"  # nb_sub guard test — uncomment when model available
)

# ─── Argument parsing ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) IFS=' ' read -ra MODELS <<< "$2"; shift 2 ;;
        --n)      N="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --csv)    CSV_FILE="$2"; shift 2 ;;
        --bench)  BENCH="$2"; shift 2 ;;
        --quick)  N=2; WARMUP=0; echo "Quick mode: N=2, no warmup (fast iteration, not statistically rigorous)"; shift ;;
        --multigpu) MULTIGPU=1; echo "Multi-GPU smoke test enabled (requires 2 GPUs)"; shift ;;
        --help|-h)
            cat <<'USAGE'
Usage: scripts/tmac-regression.sh [OPTIONS]

Options:
  --models "m1 m2"  Override model list (space-separated paths)
  --n N             Number of measured runs per config (default: 5)
  --warmup N        Warmup runs to discard (default: 1)
  --csv FILE        Output CSV path (default: timestamped)
  --bench PATH      llama-bench binary (default: ./build/bin/llama-bench)
  --quick           Quick mode: N=2, no warmup. Fast iteration, not
                    statistically rigorous — use for development only.
  --multigpu        Run dual-GPU coherence smoke test after single-GPU regression.
                    Requires HIP_VISIBLE_DEVICES with 2 GPUs.
  --help            Show this help message

Environment:
  GGML_HIP_NO_TMAC=1   Used internally for stock baseline comparison

Examples:
  # Full regression test (default models, N=5)
  scripts/tmac-regression.sh

  # Quick smoke test during development
  scripts/tmac-regression.sh --quick

  # Single model, more runs
  scripts/tmac-regression.sh --models "models/1B-Q4_K_M.gguf" --n 10
USAGE
            exit 0 ;;
        *)        echo "Unknown option: $1 (try --help)"; exit 1 ;;
    esac
done

# ─── Validation ────────────────────────────────────────────────────────
if (( N < 4 )); then
    echo "WARNING: N=$N is too small for reliable paired t-test (df=$((N-1))). Recommend N>=5."
    echo "         Results will be reported but p-values may be unreliable."
fi

if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: benchmark binary not found or not executable: $BENCH"
    echo "Build with: cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_HIP_TMAC=ON"
    exit 1
fi

for model in "${MODELS[@]}"; do
    if [[ ! -f "$model" ]]; then
        echo "ERROR: model file not found: $model"
        exit 1
    fi
done

# ─── Environment snapshot (P15 QoS driver tracking) ──────────────────
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
HIP_VER=$(hipcc --version 2>/dev/null | sed -n 's/.*HIP version: \(\S*\).*/\1/p' || echo "unknown")
GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*Card Series:[[:space:]]*//p' | head -1 || echo "unknown")
KERNEL_VER=$(uname -r)
TMAC_COMMIT=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")

# ─── CSV header (with environment metadata for P15 driver tracking) ───
{
    echo "# date=$(date -Iseconds),rocm=$ROCM_VER,hip=$HIP_VER,gpu=$GPU_NAME,kernel=$KERNEL_VER,commit=$TMAC_COMMIT"
    echo "model,metric,run,variant,tokens_per_sec"
} > "$CSV_FILE"

# ─── Helper: run single benchmark, extract t/s ────────────────────────
run_bench() {
    local bench_cmd="$1" model="$2" metric="$3" env_prefix="$4"
    local args="-m $model -r 1 -o csv"

    case "$metric" in
        pp512)  args="$args -p 512 -n 0" ;;
        tg128)  args="$args -p 0 -n 128" ;;
    esac

    # llama-bench CSV output: header line + data line.
    # Find avg_ts column by name from header, extract from data line.
    local raw
    raw=$(env $env_prefix $bench_cmd $args 2>/dev/null)
    echo "$raw" | awk -F',' '
        NR==1 { for(i=1;i<=NF;i++) if($i=="avg_ts") col=i }
        NR==2 && col { gsub(/"/, "", $col); print $col }
    '
}

# Shared stats helpers (compute_mean, compute_sd, paired_ttest, paired_ttest_full).
# Note: run_bench is NOT sourced — this script's version takes bench_cmd as first arg
# and omits -ngl (llama-bench defaults to full offload).
source "$(dirname "$0")/lib/bench-helpers.sh"

# ─── Main benchmark loop: interleaved A-B ─────────────────────────────
echo "T-MAC Regression Test"
echo "====================="
echo "Date: $(date -Iseconds)"
echo "ROCm: $ROCM_VER | HIP: $HIP_VER | GPU: $GPU_NAME"
echo "Kernel: $KERNEL_VER | T-MAC commit: $TMAC_COMMIT"
echo "N=$N measured runs + $WARMUP warmup (discarded), interleaved A-B ordering"
echo "Thresholds: tg128 >= -${THRESH_TG}%, pp512 >= -${THRESH_PP}%"
echo "Statistical test: paired t-test, p < $P_THRESH"
echo "Binary: $BENCH"
echo "Stock baseline: GGML_HIP_NO_TMAC=1"
echo "CSV output: $CSV_FILE"
echo ""

REGRESSION=0

for model in "${MODELS[@]}"; do
    model_name=$(basename "$model" .gguf)
    echo "── $model_name ──"

    for metric in tg128 pp512; do
        declare -a tmac_results=()
        declare -a stock_results=()

        # Warmup runs (discarded)
        for ((w=1; w<=WARMUP; w++)); do
            run_bench "$BENCH" "$model" "$metric" "" >/dev/null
            run_bench "$BENCH" "$model" "$metric" "GGML_HIP_NO_TMAC=1" >/dev/null
            printf "  %s warmup %d/%d (discarded)\n" "$metric" "$w" "$WARMUP"
        done

        # Measured runs
        for ((i=1; i<=N; i++)); do
            # Interleaved: alternate which runs first to cancel thermal drift
            if (( i % 2 == 1 )); then
                t=$(run_bench "$BENCH" "$model" "$metric" "")
                s=$(run_bench "$BENCH" "$model" "$metric" "GGML_HIP_NO_TMAC=1")
            else
                s=$(run_bench "$BENCH" "$model" "$metric" "GGML_HIP_NO_TMAC=1")
                t=$(run_bench "$BENCH" "$model" "$metric" "")
            fi

            tmac_results+=("$t")
            stock_results+=("$s")
            echo "$model_name,$metric,$i,tmac,$t" >> "$CSV_FILE"
            echo "$model_name,$metric,$i,stock,$s" >> "$CSV_FILE"
            printf "  %s run %d/%d: tmac=%.2f  stock=%.2f\n" "$metric" "$i" "$N" "$t" "$s"
        done

        tmac_mean=$(compute_mean "${tmac_results[@]}")
        stock_mean=$(compute_mean "${stock_results[@]}")
        delta=$(awk "BEGIN {printf \"%.2f\", ($tmac_mean - $stock_mean) / $stock_mean * 100}")

        # Paired t-test
        p_value=$(paired_ttest tmac_results stock_results)

        # Select threshold
        thresh=$THRESH_TG
        [[ "$metric" == "pp512" ]] && thresh=$THRESH_PP

        # Check for regression: FAIL if delta < -threshold AND statistically significant
        status="PASS"
        neg_thresh=$(awk "BEGIN {printf \"%.2f\", -$thresh}")
        is_regressed=$(awk "BEGIN {print ($delta < $neg_thresh) ? 1 : 0}")
        is_significant=$(awk "BEGIN {print ($p_value < $P_THRESH) ? 1 : 0}")
        if (( is_regressed && is_significant )); then
            status="FAIL"
            REGRESSION=1
        fi

        printf "  %s: tmac=%.2f  stock=%.2f  delta=%s%%  p=%s  [%s]\n\n" \
            "$metric" "$tmac_mean" "$stock_mean" "$delta" "$p_value" "$status"
    done
done

# ─── Multi-GPU smoke test (optional) ──────────────────────────────────
if (( MULTIGPU )); then
    echo "── Multi-GPU Smoke Test ──"
    COMPLETION="${BENCH%llama-bench}llama-completion"
    MGPU_MODEL="models/1B-Q4_K_M.gguf"

    if [[ ! -x "$COMPLETION" ]]; then
        echo "  SKIP: llama-completion not found at $COMPLETION"
    elif [[ ! -f "$MGPU_MODEL" ]]; then
        echo "  SKIP: model $MGPU_MODEL not found"
    else
        # Dual-GPU coherence: does the model produce sensible output with tensor split?
        MGPU_OUTPUT=$(echo "" | HIP_VISIBLE_DEVICES=0,1 "$COMPLETION" \
            -m "$MGPU_MODEL" -p "The capital of France is" -n 20 -ngl 99 \
            --tensor-split 0.5,0.5 2>&1)
        MGPU_EXIT=$?

        if (( MGPU_EXIT != 0 )); then
            echo "  Multi-GPU: FAIL (exit code $MGPU_EXIT)"
            REGRESSION=1
        elif echo "$MGPU_OUTPUT" | grep -q "Active Ratio"; then
            MGPU_AR=$(echo "$MGPU_OUTPUT" | grep "Active Ratio" | tail -1)
            echo "  Multi-GPU: PASS (coherent output, $MGPU_AR)"
        else
            echo "  Multi-GPU: PASS (coherent output, no Active Ratio reported)"
        fi
    fi
    echo ""
fi

# ─── Summary ───────────────────────────────────────────────────────────
echo "Results saved to: $CSV_FILE"

if (( REGRESSION )); then
    echo "REGRESSION DETECTED — exit code 1"
    exit 1
else
    echo "All metrics within threshold — PASS"
    exit 0
fi
