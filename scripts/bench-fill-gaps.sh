#!/usr/bin/env bash
# bench-fill-gaps.sh — Fill benchmark gaps identified by review process.
#
# Usage: scripts/bench-fill-gaps.sh [--section NAME] [--all]
# Sections: iq-absolute, seq-length, deepseek-upgrade, iq-8b, power, deepseek-671b
#
# Requirements:
#   - MODEL_DIR env var pointing to directory with .gguf model files
#   - Single llama-bench binary built with GGML_HIP_TMAC=ON
#   - HIP_VISIBLE_DEVICES=0 set (to exclude iGPU)
#
# Estimated runtime: ~2.5h for --all (without 671B)

set -euo pipefail
export LC_NUMERIC=C  # Force decimal point for de_DE and similar locales

# Exclude iGPU (gfx1036 segfaults) — default to GPU 0 if not set
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# ─── Defaults ───────────────────────────────────────────────────────────
BENCH="${BENCH:-./build/bin/llama-bench}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
CSV_FILE="bench-fill-gaps-${TIMESTAMP}.csv"
SECTION=""
RUN_ALL=0

# ─── Argument parsing ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --section) SECTION="$2"; shift 2 ;;
        --all)     RUN_ALL=1; shift ;;
        --csv)     CSV_FILE="$2"; shift 2 ;;
        --bench)   BENCH="$2"; shift 2 ;;
        --help|-h)
            cat <<'USAGE'
Usage: scripts/bench-fill-gaps.sh [OPTIONS]

Fill benchmark gaps identified by benchmark review for public documentation.

Options:
  --section NAME  Run only one section (see below)
  --all           Run all sections sequentially
  --csv FILE      Output CSV path (default: timestamped)
  --bench PATH    llama-bench binary (default: ./build/bin/llama-bench)
  --help          Show this help message

Sections:
  iq-absolute      Missing IQ absolute t/s values (~30 min)
  seq-length       Variable generation lengths tg128/512/2048 (~1.5h)
  deepseek-upgrade DeepSeek-R1 8B N=3 → N=10 (~20 min)
  iq-8b            IQ3_XXS on 8B model — cache artifact test (~20 min)
  power            Power consumption via rocm-smi (~10 min)
  deepseek-671b    DeepSeek-R1 671B smoke test (optional, dual GPU + CPU)

Environment:
  MODEL_DIR        Required. Path to directory with .gguf model files.
  HIP_VISIBLE_DEVICES  Should be set to 0 (exclude iGPU).
  BENCH            Override llama-bench path.

Examples:
  MODEL_DIR=/path/to/models scripts/bench-fill-gaps.sh --section iq-absolute
  MODEL_DIR=/path/to/models scripts/bench-fill-gaps.sh --all
USAGE
            exit 0 ;;
        *)  echo "Unknown option: $1 (try --help)"; exit 1 ;;
    esac
done

# ─── Validation ─────────────────────────────────────────────────────────
if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "ERROR: MODEL_DIR environment variable is required."
    echo "Usage: MODEL_DIR=/path/to/models scripts/bench-fill-gaps.sh --section iq-absolute"
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR"
    exit 1
fi

if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: benchmark binary not found or not executable: $BENCH"
    echo "Build with: cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100"
    exit 1
fi

if [[ -z "$SECTION" ]] && (( ! RUN_ALL )); then
    echo "ERROR: specify --section NAME or --all"
    echo "Try --help for available sections."
    exit 1
fi

# ─── Environment snapshot ──────────────────────────────────────────────
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
HIP_VER=$(hipcc --version 2>/dev/null | grep -oP 'HIP version: \K\S+' || echo "unknown")
GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'Card Series:\s+\K.*' | head -1 || echo "unknown")
KERNEL_VER=$(uname -r)
TMAC_COMMIT=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")

# ─── CSV header ─────────────────────────────────────────────────────────
{
    echo "# date=$(date -Iseconds),rocm=$ROCM_VER,hip=$HIP_VER,gpu=$GPU_NAME,kernel=$KERNEL_VER,commit=$TMAC_COMMIT"
    echo "section,model,metric,run,variant,tokens_per_sec"
} > "$CSV_FILE"

# ─── Helpers ────────────────────────────────────────────────────────────

# Find a model file by pattern (case-insensitive glob in MODEL_DIR)
find_model() {
    local pattern="$1"
    local result
    result=$(find "$MODEL_DIR" -maxdepth 1 -iname "$pattern" -print -quit 2>/dev/null)
    echo "$result"
}

# Run single benchmark, extract t/s from CSV output
run_bench() {
    local model="$1" metric="$2" env_prefix="$3"
    local args="-m $model -r 1 -ngl 99 -o csv"

    case "$metric" in
        pp512)   args="$args -p 512 -n 0" ;;
        tg128)   args="$args -p 0 -n 128" ;;
        tg512)   args="$args -p 0 -n 512" ;;
        tg2048)  args="$args -p 0 -n 2048" ;;
    esac

    local raw
    if [[ -n "$env_prefix" ]]; then
        raw=$(env $env_prefix $BENCH $args 2>/dev/null)
    else
        raw=$($BENCH $args 2>/dev/null)
    fi
    echo "$raw" | awk -F',' '
        NR==1 { for(i=1;i<=NF;i++) if($i=="avg_ts") col=i }
        NR==2 && col { gsub(/"/, "", $col); print $col }
    '
}

# Compute mean from array of values
compute_mean() {
    local arr=("$@")
    awk -v n="${#arr[@]}" 'BEGIN {s=0; for(i=1;i<ARGC;i++) s+=ARGV[i]; printf "%.4f", s/n}' "${arr[@]}"
}

# Compute SD from array of values
compute_sd() {
    local arr=("$@")
    awk -v n="${#arr[@]}" 'BEGIN {
        s=0; for(i=1;i<ARGC;i++) s+=ARGV[i]; m=s/n;
        ss=0; for(i=1;i<ARGC;i++) ss+=(ARGV[i]-m)^2;
        printf "%.4f", sqrt(ss/(n-1))
    }' "${arr[@]}"
}

# Paired t-test (returns: t-stat, p-value, 95% CI lower, 95% CI upper of speedup%)
paired_ttest_full() {
    local -n arr_a=$1  # T-MAC
    local -n arr_b=$2  # Stock
    local n=${#arr_a[@]}

    awk -v n="$n" 'BEGIN {
        # Compute speedup ratios
        for (i = 1; i <= n; i++) {
            stock = ARGV[n + i]
            tmac = ARGV[i]
            speedup[i] = (tmac - stock) / stock * 100
        }

        # Mean speedup
        sum = 0
        for (i = 1; i <= n; i++) sum += speedup[i]
        mean_sp = sum / n

        # SD of speedup
        ss = 0
        for (i = 1; i <= n; i++) ss += (speedup[i] - mean_sp)^2
        sd_sp = sqrt(ss / (n - 1))
        se = sd_sp / sqrt(n)

        # t-statistic on paired differences (raw values, not ratios)
        for (i = 1; i <= n; i++) d[i] = ARGV[i] - ARGV[n + i]
        sum_d = 0
        for (i = 1; i <= n; i++) sum_d += d[i]
        mean_d = sum_d / n
        ss_d = 0
        for (i = 1; i <= n; i++) ss_d += (d[i] - mean_d)^2
        sd_d = sqrt(ss_d / (n - 1))
        if (sd_d == 0) { printf "inf 0.0000 %.2f %.2f", mean_sp, mean_sp; exit }
        t = mean_d / (sd_d / sqrt(n))
        if (t < 0) t_abs = -t; else t_abs = t

        # Approximate p-value (normal approx, conservative for df >= 4)
        z = t_abs
        b1 = 0.319381530; b2 = -0.356563782; b3 = 1.781477937
        b4 = -1.821255978; b5 = 1.330274429; p_coeff = 0.2316419
        tt = 1.0 / (1.0 + p_coeff * z)
        phi = (1.0 / sqrt(2 * 3.14159265358979)) * exp(-z * z / 2.0)
        cdf = 1.0 - phi * (b1*tt + b2*tt^2 + b3*tt^3 + b4*tt^4 + b5*tt^5)
        p = 2.0 * (1.0 - cdf)

        # 95% CI for speedup (t-critical approx 2.262 for df=9, 2.776 for df=4)
        # Use simple lookup for common df values
        df = n - 1
        if (df == 4) t_crit = 2.776
        else if (df == 9) t_crit = 2.262
        else if (df == 11) t_crit = 2.201
        else t_crit = 2.0 + 1.0 / df  # rough approximation

        ci_lo = mean_sp - t_crit * se
        ci_hi = mean_sp + t_crit * se

        printf "%.2f %.4f %.1f %.1f", t_abs, p, ci_lo, ci_hi
    }' "${arr_a[@]}" "${arr_b[@]}"
}

# Run interleaved A-B benchmark for a model+metric, return results
# Usage: run_interleaved SECTION MODEL METRIC N WARMUP
run_interleaved() {
    local section="$1" model="$2" metric="$3" n="$4" warmup="$5"
    local model_name
    model_name=$(basename "$model" .gguf)

    declare -a tmac_results=()
    declare -a stock_results=()

    # Warmup runs (discarded)
    for ((w=1; w<=warmup; w++)); do
        run_bench "$model" "$metric" "" >/dev/null
        run_bench "$model" "$metric" "GGML_HIP_NO_TMAC=1" >/dev/null
        printf "  %s warmup %d/%d (discarded)\n" "$metric" "$w" "$warmup"
    done

    # Measured runs (interleaved)
    for ((i=1; i<=n; i++)); do
        if (( i % 2 == 1 )); then
            t=$(run_bench "$model" "$metric" "")
            s=$(run_bench "$model" "$metric" "GGML_HIP_NO_TMAC=1")
        else
            s=$(run_bench "$model" "$metric" "GGML_HIP_NO_TMAC=1")
            t=$(run_bench "$model" "$metric" "")
        fi

        tmac_results+=("$t")
        stock_results+=("$s")
        echo "$section,$model_name,$metric,$i,tmac,$t" >> "$CSV_FILE"
        echo "$section,$model_name,$metric,$i,stock,$s" >> "$CSV_FILE"
        printf "  %s run %d/%d: tmac=%.2f  stock=%.2f\n" "$metric" "$i" "$n" "$t" "$s"
    done

    # Compute statistics
    local tmac_mean stock_mean tmac_sd stock_sd speedup stats t_stat p_val ci_lo ci_hi
    tmac_mean=$(compute_mean "${tmac_results[@]}")
    stock_mean=$(compute_mean "${stock_results[@]}")
    tmac_sd=$(compute_sd "${tmac_results[@]}")
    stock_sd=$(compute_sd "${stock_results[@]}")
    speedup=$(awk "BEGIN {printf \"%.1f\", ($tmac_mean - $stock_mean) / $stock_mean * 100}")

    stats=$(paired_ttest_full tmac_results stock_results)
    t_stat=$(echo "$stats" | awk '{print $1}')
    p_val=$(echo "$stats" | awk '{print $2}')
    ci_lo=$(echo "$stats" | awk '{print $3}')
    ci_hi=$(echo "$stats" | awk '{print $4}')

    printf "\n  RESULT: %s %s\n" "$model_name" "$metric"
    printf "    Stock:   %.2f ± %.2f t/s\n" "$stock_mean" "$stock_sd"
    printf "    T-MAC:   %.2f ± %.2f t/s\n" "$tmac_mean" "$tmac_sd"
    printf "    Speedup: +%s%% [%s%%, %s%%]  (t=%.2f, p=%s)\n\n" "$speedup" "$ci_lo" "$ci_hi" "$t_stat" "$p_val"
}

# ─── Section: iq-absolute ───────────────────────────────────────────────
# Fill missing absolute t/s values for IQ types that show "—" in benchmarks.md
section_iq_absolute() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Section: iq-absolute — Missing IQ absolute t/s (CRITICAL)"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Estimated time: ~30 min"
    echo ""

    local models=(
        "Llama-3.2-1B-Instruct-IQ3_XXS.gguf"
        "Llama-3.2-1B-Instruct-IQ2_XXS.gguf"
        "Llama-3.2-1B-Instruct-IQ2_XS.gguf"
    )

    for mfile in "${models[@]}"; do
        local model="$MODEL_DIR/$mfile"
        if [[ ! -f "$model" ]]; then
            echo "  SKIP: $mfile not found in MODEL_DIR"
            continue
        fi
        echo "── $(basename "$model" .gguf) ──"
        run_interleaved "iq-absolute" "$model" "tg128" 10 1
    done
}

# ─── Section: seq-length ────────────────────────────────────────────────
# Measure speedup at different generation lengths to show it holds/varies
section_seq_length() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Section: seq-length — Variable generation lengths"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Estimated time: ~1.5h"
    echo ""

    local -A model_files=(
        ["1B"]="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ["22B"]="Codestral-22B-v0.1-Q4_K_M.gguf"
        ["QwQ-32B"]="Qwen_Qwen3.5-27B-Q4_K_M.gguf"  # Using 27B as QwQ-32B substitute if available
    )

    local -A model_lengths=(
        ["1B"]="tg128 tg512 tg2048"
        ["22B"]="tg128 tg512 tg2048"
        ["QwQ-32B"]="tg128 tg512"
    )

    for label in "1B" "22B" "QwQ-32B"; do
        local mfile="${model_files[$label]}"
        local model="$MODEL_DIR/$mfile"

        if [[ ! -f "$model" ]]; then
            # Try alternate names
            if [[ "$label" == "QwQ-32B" ]]; then
                model=$(find_model "*QwQ*Q4_K_M*")
                if [[ -z "$model" ]]; then
                    model=$(find_model "*qwq*q4_k*")
                fi
            fi
            if [[ -z "$model" ]] || [[ ! -f "$model" ]]; then
                echo "  SKIP: No $label Q4_K_M model found in MODEL_DIR"
                continue
            fi
        fi

        echo "── $label Q4_K_M ──"
        for metric in ${model_lengths[$label]}; do
            run_interleaved "seq-length" "$model" "$metric" 5 1
        done
    done
}

# ─── Section: deepseek-upgrade ──────────────────────────────────────────
# Upgrade DeepSeek-R1 8B from N=3 to N=10
section_deepseek_upgrade() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Section: deepseek-upgrade — DeepSeek-R1 8B N=10"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Estimated time: ~20 min"
    echo ""

    local model
    model=$(find_model "*DeepSeek*R1*8B*Q4_K*")
    if [[ -z "$model" ]]; then
        model=$(find_model "*deepseek*r1*8b*q4_k*")
    fi
    if [[ -z "$model" ]] || [[ ! -f "$model" ]]; then
        echo "  SKIP: DeepSeek-R1-Distill-Llama-8B Q4_K_M not found in MODEL_DIR"
        echo "  Download from: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF"
        return 0
    fi

    echo "── $(basename "$model" .gguf) ──"
    run_interleaved "deepseek-upgrade" "$model" "tg128" 10 1
}

# ─── Section: iq-8b ─────────────────────────────────────────────────────
# Test IQ3_XXS on 8B model — disproves/confirms cache artifact hypothesis
section_iq_8b() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Section: iq-8b — IQ3 on 8B (cache artifact test)"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Estimated time: ~20 min"
    echo ""

    local model
    # Try IQ3_XXS first, fall back to IQ3_S (IQ3_XXS not available pre-quantized for 8B)
    model=$(find_model "*8B*IQ3_XXS*")
    if [[ -z "$model" ]]; then
        model=$(find_model "*8b*iq3_xxs*")
    fi
    if [[ -z "$model" ]] || [[ ! -f "$model" ]]; then
        model=$(find_model "*8B*IQ3_S*")
        if [[ -z "$model" ]]; then
            model=$(find_model "*8b*IQ3_S*")
        fi
    fi
    if [[ -z "$model" ]] || [[ ! -f "$model" ]]; then
        echo "  SKIP: 8B IQ3_XXS/IQ3_S model not found in MODEL_DIR"
        echo "  Download from: https://huggingface.co/mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF"
        return 0
    fi

    echo "── $(basename "$model" .gguf) ──"
    run_interleaved "iq-8b" "$model" "tg128" 5 1
}

# ─── Section: power ─────────────────────────────────────────────────────
# Measure board power during sustained generation
section_power() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Section: power — Board power consumption"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Estimated time: ~10 min"
    echo ""

    if ! command -v rocm-smi &>/dev/null; then
        echo "  SKIP: rocm-smi not found"
        return 0
    fi

    local models=(
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        "Codestral-22B-v0.1-Q4_K_M.gguf"
    )
    local labels=("1B" "22B")

    for idx in "${!models[@]}"; do
        local mfile="${models[$idx]}"
        local label="${labels[$idx]}"
        local model="$MODEL_DIR/$mfile"

        if [[ ! -f "$model" ]]; then
            echo "  SKIP: $mfile not found"
            continue
        fi

        echo "── $label Q4_K_M Power Draw ──"

        for variant in "tmac" "stock"; do
            local env_prefix=""
            [[ "$variant" == "stock" ]] && env_prefix="GGML_HIP_NO_TMAC=1"

            local power_samples=()

            for ((run=1; run<=3; run++)); do
                # Start power monitoring in background
                local power_log
                power_log=$(mktemp /tmp/power-XXXXXX.log)
                (
                    while true; do
                        rocm-smi --showpower 2>/dev/null | grep -oP '\d+\.\d+ W' | head -1 >> "$power_log"
                        sleep 0.5
                    done
                ) &
                local power_pid=$!

                # Run benchmark (tg128)
                run_bench "$model" "tg128" "$env_prefix" >/dev/null

                # Stop power monitoring
                kill "$power_pid" 2>/dev/null
                wait "$power_pid" 2>/dev/null || true

                # Compute average power from log
                local avg_power
                avg_power=$(awk '{gsub(/ W/, ""); s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "0"}' "$power_log")
                power_samples+=("$avg_power")
                rm -f "$power_log"
                printf "  %s run %d/3: %s = %.1f W\n" "$label" "$run" "$variant" "$avg_power"
            done

            # Average across 3 runs
            local avg
            avg=$(compute_mean "${power_samples[@]}")
            printf "  %s %s average: %.1f W\n\n" "$label" "$variant" "$avg"
            echo "power,$label-Q4_K_M,tg128,avg,$variant,$avg" >> "$CSV_FILE"
        done
    done
}

# ─── Section: deepseek-671b ─────────────────────────────────────────────
# Optional smoke test for DeepSeek-R1 671B (requires dual GPU + CPU offload)
section_deepseek_671b() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "Section: deepseek-671b — Optional 671B smoke test"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    local model
    model=$(find_model "*DeepSeek*R1*671B*")
    if [[ -z "$model" ]]; then
        model=$(find_model "*deepseek*671b*")
    fi
    if [[ -z "$model" ]] || [[ ! -f "$model" ]]; then
        echo "  SKIP: DeepSeek-R1 671B model not found in MODEL_DIR"
        echo "  This section is optional — skip if model is not available."
        return 0
    fi

    echo "── $(basename "$model" .gguf) ──"
    echo "  Single run, dual GPU + CPU offload"

    local COMPLETION="${BENCH%llama-bench}llama-completion"
    if [[ ! -x "$COMPLETION" ]]; then
        echo "  SKIP: llama-completion not found at $COMPLETION"
        return 0
    fi

    # T-MAC run
    printf "  T-MAC: "
    local tmac_out
    tmac_out=$(echo "" | HIP_VISIBLE_DEVICES=0,1 "$COMPLETION" \
        -m "$model" -p "The capital of France is" -n 64 -ngl 99 \
        --tensor-split 0.5,0.5 2>&1)
    local tmac_ts
    tmac_ts=$(echo "$tmac_out" | grep -oP '[\d.]+ tokens per second' | head -1 | awk '{print $1}')
    local tmac_ar
    tmac_ar=$(echo "$tmac_out" | grep "Active Ratio" | tail -1 || echo "N/A")
    printf "%s t/s, %s\n" "${tmac_ts:-N/A}" "$tmac_ar"

    # Stock run
    printf "  Stock: "
    local stock_out
    stock_out=$(echo "" | HIP_VISIBLE_DEVICES=0,1 GGML_HIP_NO_TMAC=1 "$COMPLETION" \
        -m "$model" -p "The capital of France is" -n 64 -ngl 99 \
        --tensor-split 0.5,0.5 2>&1)
    local stock_ts
    stock_ts=$(echo "$stock_out" | grep -oP '[\d.]+ tokens per second' | head -1 | awk '{print $1}')
    printf "%s t/s\n" "${stock_ts:-N/A}"

    if [[ -n "$tmac_ts" ]] && [[ -n "$stock_ts" ]]; then
        local speedup
        speedup=$(awk "BEGIN {printf \"%.1f\", ($tmac_ts - $stock_ts) / $stock_ts * 100}")
        printf "  Speedup: +%s%%\n" "$speedup"
        echo "deepseek-671b,$(basename "$model" .gguf),tg64,1,tmac,$tmac_ts" >> "$CSV_FILE"
        echo "deepseek-671b,$(basename "$model" .gguf),tg64,1,stock,$stock_ts" >> "$CSV_FILE"
    fi
    echo ""
}

# ─── Dispatch ───────────────────────────────────────────────────────────
echo "T-MAC Benchmark Gap Fill"
echo "=========================="
echo "Date: $(date -Iseconds)"
echo "ROCm: $ROCM_VER | HIP: $HIP_VER | GPU: $GPU_NAME"
echo "Kernel: $KERNEL_VER | T-MAC commit: $TMAC_COMMIT"
echo "MODEL_DIR: $MODEL_DIR"
echo "Binary: $BENCH"
echo "CSV output: $CSV_FILE"

run_section() {
    case "$1" in
        iq-absolute)      section_iq_absolute ;;
        seq-length)       section_seq_length ;;
        deepseek-upgrade) section_deepseek_upgrade ;;
        iq-8b)            section_iq_8b ;;
        power)            section_power ;;
        deepseek-671b)    section_deepseek_671b ;;
        *)                echo "ERROR: Unknown section: $1"; exit 1 ;;
    esac
}

if (( RUN_ALL )); then
    for sec in iq-absolute seq-length deepseek-upgrade iq-8b power deepseek-671b; do
        run_section "$sec"
    done
else
    run_section "$SECTION"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Done. Results saved to: $CSV_FILE"
echo "═══════════════════════════════════════════════════════════════"
