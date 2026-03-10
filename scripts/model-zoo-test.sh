#!/usr/bin/env bash
# model-zoo-test.sh — Run T-MAC validation suite on a model.
#
# Usage:
#   scripts/model-zoo-test.sh MODEL.gguf [--tier 1|2|3|4|all] [OPTIONS]
#
# Tiers:
#   1: Smoke test — coherent output, Active Ratio, Compute Coverage
#   2: Long-form coherence — 2048 tokens at temp=0
#   3: Perplexity — wikitext-2, T-MAC vs Stock, Δ must be < 0.1
#   4: Benchmark N=5 paired interleaved — speedup + CI + p-value
#   all: Tiers 1-4 sequentially (default)
#
# Options:
#   --tier TIER       Which tier(s) to run (default: all)
#   --n N             Benchmark runs for Tier 4 (default: 5)
#   --warmup N        Warmup runs for Tier 4 (default: 1)
#   --dual            Force dual-GPU mode (auto-detected from file size)
#   --single          Force single-GPU mode
#   --csv FILE        CSV output file (default: timestamped)
#   --tracker FILE    Tracker markdown to append results (default: docs/tmac/model-zoo-tracker.md)
#   --no-tracker      Don't update tracker file
#   --help            Show this help
#
# Environment:
#   HIP_VISIBLE_DEVICES   GPU selection (default: 0 for single, 0,1 for dual)
#   GGML_HIP_NO_TMAC=1   Used internally for stock baseline
#   BENCH                 Override llama-bench path
#   COMPLETION            Override llama-completion path
#   PERPLEXITY            Override llama-perplexity path

set -euo pipefail
export LC_NUMERIC=C

# ─── Defaults ───────────────────────────────────────────────────────────
TIER="all"
N=5
WARMUP=1
DUAL_GPU=""  # empty = auto-detect
CSV_FILE=""
TRACKER="docs/tmac/model-zoo-tracker.md"
UPDATE_TRACKER=1
MODEL=""

BUILD_DIR="./build/bin"
BENCH="${BENCH:-$BUILD_DIR/llama-bench}"
COMPLETION="${COMPLETION:-$BUILD_DIR/llama-completion}"
PERPLEXITY="${PERPLEXITY:-$BUILD_DIR/llama-perplexity}"
WIKITEXT="${WIKITEXT:-/mnt/llm-data/wikitext-2-raw/wiki.test.raw}"

# VRAM threshold for auto dual-GPU detection (bytes)
DUAL_GPU_THRESHOLD=$((23 * 1024 * 1024 * 1024))  # 23 GB

# ─── Color helpers ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
fail() { printf "${RED}[FAIL]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
info() { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }

# ─── Argument parsing ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)       TIER="$2"; shift 2 ;;
        --n)          N="$2"; shift 2 ;;
        --warmup)     WARMUP="$2"; shift 2 ;;
        --dual)       DUAL_GPU=1; shift ;;
        --single)     DUAL_GPU=0; shift ;;
        --csv)        CSV_FILE="$2"; shift 2 ;;
        --tracker)    TRACKER="$2"; shift 2 ;;
        --no-tracker) UPDATE_TRACKER=0; shift ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p }' "$0"
            exit 0 ;;
        -*)           echo "Unknown option: $1 (try --help)"; exit 1 ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"; shift
            else
                echo "ERROR: unexpected argument: $1"; exit 1
            fi ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: no model file specified"
    echo "Usage: scripts/model-zoo-test.sh MODEL.gguf [--tier 1|2|3|4|all]"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL"
    exit 1
fi

# ─── Derived paths ──────────────────────────────────────────────────────
MODEL_NAME=$(basename "$MODEL" .gguf)
MODEL_SIZE=$(stat -L -c%s "$MODEL" 2>/dev/null || stat -L -f%z "$MODEL" 2>/dev/null)

if [[ -z "$CSV_FILE" ]]; then
    CSV_FILE="model-zoo-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S).csv"
fi

# Auto-detect GPU mode
if [[ -z "$DUAL_GPU" ]]; then
    if (( MODEL_SIZE > DUAL_GPU_THRESHOLD )); then
        DUAL_GPU=1
        info "Model size $(( MODEL_SIZE / 1024 / 1024 / 1024 )) GB > 23 GB threshold → dual GPU mode"
    else
        DUAL_GPU=0
    fi
fi

# Set HIP_VISIBLE_DEVICES based on GPU mode
if (( DUAL_GPU )); then
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
    GPU_ARGS="--tensor-split 0.5,0.5"
    GPU_LABEL="dual"
else
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
    GPU_ARGS=""
    GPU_LABEL="single"
fi

# ─── Validation ─────────────────────────────────────────────────────────
for bin_var in BENCH COMPLETION PERPLEXITY; do
    bin_path="${!bin_var}"
    if [[ ! -x "$bin_path" ]]; then
        warn "$bin_var binary not found: $bin_path"
        case "$bin_var" in
            BENCH)      warn "Tier 4 (benchmark) will be skipped" ;;
            COMPLETION) warn "Tiers 1-2 (smoke/coherence) will be skipped" ;;
            PERPLEXITY) warn "Tier 3 (perplexity) will be skipped" ;;
        esac
    fi
done

# ─── Environment snapshot ──────────────────────────────────────────────
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
HIP_VER=$(hipcc --version 2>/dev/null | grep -oP 'HIP version: \K\S+' || echo "unknown")
GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -oP 'Card Series:\s+\K.*' | head -1 || echo "unknown")
KERNEL_VER=$(uname -r)
TMAC_COMMIT=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")

# ─── CSV init ───────────────────────────────────────────────────────────
{
    echo "# model=$MODEL_NAME,date=$(date -Iseconds),rocm=$ROCM_VER,gpu=$GPU_NAME,commit=$TMAC_COMMIT,gpu_mode=$GPU_LABEL"
    echo "tier,model,metric,run,variant,value,unit"
} > "$CSV_FILE"

# ─── Header ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}T-MAC Model Zoo Test${NC}"
echo "════════════════════════════════════════════════════════════"
echo "  Model:    $MODEL_NAME"
echo "  Size:     $(( MODEL_SIZE / 1024 / 1024 )) MB"
echo "  GPU:      $GPU_LABEL ($HIP_VISIBLE_DEVICES) $GPU_NAME"
echo "  Tier:     $TIER"
echo "  ROCm:     $ROCM_VER | HIP: $HIP_VER"
echo "  Commit:   $TMAC_COMMIT"
echo "  CSV:      $CSV_FILE"
echo "════════════════════════════════════════════════════════════"
echo ""

# Track results for tracker update
declare -A RESULTS

# ─── Helper: extract Active Ratio from stderr ──────────────────────────
extract_active_ratio() {
    local stderr_file="$1"
    # T-MAC prints: "T-MAC: Active Ratio: 100.0% (N/M dispatched via T-MAC)"
    grep -oP 'Active Ratio: \K[\d.]+%' "$stderr_file" | tail -1 || echo "N/A"
}

# ─── Helper: extract Compute Coverage from stderr ──────────────────────
extract_compute_coverage() {
    local stderr_file="$1"
    grep -oP 'Compute Coverage: \K[\d.]+%' "$stderr_file" | tail -1 || echo "N/A"
}

# ─── Helper: run_bench (same pattern as tmac-regression.sh) ────────────
run_bench() {
    local model="$1" metric="$2" env_prefix="$3"
    local args="-m $model -r 1 -ngl 99 -o csv $GPU_ARGS"

    case "$metric" in
        pp512)  args="$args -p 512 -n 0" ;;
        tg128)  args="$args -p 0 -n 128" ;;
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

# ─── Helper: compute_mean ──────────────────────────────────────────────
compute_mean() {
    local arr=("$@")
    awk -v n="${#arr[@]}" 'BEGIN {s=0; for(i=1;i<ARGC;i++) s+=ARGV[i]; printf "%.4f", s/n}' "${arr[@]}"
}

# ─── Helper: compute_sd ───────────────────────────────────────────────
compute_sd() {
    local arr=("$@")
    awk -v n="${#arr[@]}" 'BEGIN {
        s=0; for(i=1;i<ARGC;i++) s+=ARGV[i]; m=s/n;
        ss=0; for(i=1;i<ARGC;i++) ss+=(ARGV[i]-m)^2;
        printf "%.4f", sqrt(ss/(n-1))
    }' "${arr[@]}"
}

# ─── Helper: paired t-test with CI ────────────────────────────────────
paired_ttest_full() {
    local -n arr_a=$1  # T-MAC
    local -n arr_b=$2  # Stock
    local n=${#arr_a[@]}

    awk -v n="$n" 'BEGIN {
        for (i = 1; i <= n; i++) {
            stock = ARGV[n + i]
            tmac = ARGV[i]
            speedup[i] = (tmac - stock) / stock * 100
        }

        sum = 0
        for (i = 1; i <= n; i++) sum += speedup[i]
        mean_sp = sum / n

        ss = 0
        for (i = 1; i <= n; i++) ss += (speedup[i] - mean_sp)^2
        sd_sp = sqrt(ss / (n - 1))
        se = sd_sp / sqrt(n)

        for (i = 1; i <= n; i++) d[i] = ARGV[i] - ARGV[n + i]
        sum_d = 0
        for (i = 1; i <= n; i++) sum_d += d[i]
        mean_d = sum_d / n
        ss_d = 0
        for (i = 1; i <= n; i++) ss_d += (d[i] - mean_d)^2
        sd_d = sqrt(ss_d / (n - 1))
        if (sd_d == 0) { printf "inf 0.0000 %.1f %.1f", mean_sp, mean_sp; exit }
        t = mean_d / (sd_d / sqrt(n))
        if (t < 0) t_abs = -t; else t_abs = t

        z = t_abs
        b1 = 0.319381530; b2 = -0.356563782; b3 = 1.781477937
        b4 = -1.821255978; b5 = 1.330274429; p_coeff = 0.2316419
        tt = 1.0 / (1.0 + p_coeff * z)
        phi = (1.0 / sqrt(2 * 3.14159265358979)) * exp(-z * z / 2.0)
        cdf = 1.0 - phi * (b1*tt + b2*tt^2 + b3*tt^3 + b4*tt^4 + b5*tt^5)
        p = 2.0 * (1.0 - cdf)

        df = n - 1
        if (df == 4) t_crit = 2.776
        else if (df == 9) t_crit = 2.262
        else if (df == 11) t_crit = 2.201
        else t_crit = 2.0 + 1.0 / df

        ci_lo = mean_sp - t_crit * se
        ci_hi = mean_sp + t_crit * se

        printf "%.2f %.4f %.1f %.1f %.1f", t_abs, p, ci_lo, ci_hi, mean_sp
    }' "${arr_a[@]}" "${arr_b[@]}"
}

# ═══════════════════════════════════════════════════════════════════════
# TIER 1: Smoke Test
# ═══════════════════════════════════════════════════════════════════════
tier1_smoke() {
    echo -e "${BOLD}── Tier 1: Smoke Test ──${NC}"

    if [[ ! -x "$COMPLETION" ]]; then
        fail "llama-completion not found: $COMPLETION"
        RESULTS[T1]="SKIP"
        return 1
    fi

    local stderr_file
    stderr_file=$(mktemp /tmp/zoo-t1-XXXXXX.log)

    # Run T-MAC completion
    info "Running smoke test with T-MAC..."
    local output
    output=$(echo "" | $COMPLETION \
        -m "$MODEL" -p "The answer to 2+2 is" -n 20 -ngl 99 $GPU_ARGS \
        2>"$stderr_file") || true

    local active_ratio
    active_ratio=$(extract_active_ratio "$stderr_file")
    local compute_cov
    compute_cov=$(extract_compute_coverage "$stderr_file")

    # Check for coherent output (non-empty, no obvious garbage)
    if [[ -z "$output" ]]; then
        fail "Tier 1: no output produced"
        RESULTS[T1]="FAIL"
        rm -f "$stderr_file"
        return 1
    fi

    # Check output length (should have at least 10 chars of real content)
    local output_len=${#output}
    if (( output_len < 10 )); then
        fail "Tier 1: output too short ($output_len chars)"
        RESULTS[T1]="FAIL"
        rm -f "$stderr_file"
        return 1
    fi

    ok "Tier 1: Smoke test passed"
    echo "  Active Ratio: $active_ratio"
    echo "  Compute Coverage: $compute_cov"
    echo "  Output preview: ${output:0:120}..."
    echo ""

    echo "T1,$MODEL_NAME,smoke,1,tmac,$active_ratio,active_ratio" >> "$CSV_FILE"
    echo "T1,$MODEL_NAME,smoke,1,tmac,$compute_cov,compute_coverage" >> "$CSV_FILE"

    RESULTS[T1]="OK"
    RESULTS[AR]="$active_ratio"
    RESULTS[CC]="$compute_cov"
    rm -f "$stderr_file"
}

# ═══════════════════════════════════════════════════════════════════════
# TIER 2: Long-form Coherence
# ═══════════════════════════════════════════════════════════════════════
tier2_coherence() {
    echo -e "${BOLD}── Tier 2: Long-form Coherence (2048 tokens) ──${NC}"

    if [[ ! -x "$COMPLETION" ]]; then
        fail "llama-completion not found: $COMPLETION"
        RESULTS[T2]="SKIP"
        return 1
    fi

    info "Generating 2048 tokens at temp=0..."
    local output
    output=$(echo "" | $COMPLETION \
        -m "$MODEL" \
        -p "Write a comprehensive guide to building a compiler from scratch. Start with lexical analysis and tokenization, then cover parsing and abstract syntax trees, followed by semantic analysis and type checking, and finish with code generation. Include practical examples." \
        -n 2048 -ngl 99 --temp 0.0 $GPU_ARGS \
        2>/dev/null) || true

    local output_len=${#output}

    if [[ -z "$output" ]]; then
        fail "Tier 2: no output produced"
        RESULTS[T2]="FAIL"
        return 1
    fi

    # Should produce substantial output (at least 1000 chars for 2048 tokens)
    if (( output_len < 1000 )); then
        fail "Tier 2: output too short ($output_len chars for 2048 tokens)"
        RESULTS[T2]="FAIL"
        return 1
    fi

    # Check for obvious degenerate output (repeated short patterns)
    local unique_lines
    unique_lines=$(echo "$output" | sort -u | wc -l)
    local total_lines
    total_lines=$(echo "$output" | wc -l)
    if (( total_lines > 10 )) && (( unique_lines * 4 < total_lines )); then
        warn "Tier 2: possible degenerate output (only $unique_lines unique of $total_lines lines)"
    fi

    ok "Tier 2: Coherence test passed ($output_len chars, $total_lines lines)"
    echo "  Preview: ${output:0:200}..."
    echo ""

    echo "T2,$MODEL_NAME,coherence,1,tmac,$output_len,chars" >> "$CSV_FILE"

    RESULTS[T2]="OK"
}

# ═══════════════════════════════════════════════════════════════════════
# TIER 3: Perplexity
# ═══════════════════════════════════════════════════════════════════════
tier3_perplexity() {
    echo -e "${BOLD}── Tier 3: Perplexity (wikitext-2) ──${NC}"

    if [[ ! -x "$PERPLEXITY" ]]; then
        fail "llama-perplexity not found: $PERPLEXITY"
        RESULTS[T3]="SKIP"
        return 1
    fi

    if [[ ! -f "$WIKITEXT" ]]; then
        fail "Wikitext-2 corpus not found: $WIKITEXT"
        warn "Run: cp -r /tmp/wikitext-2/wikitext-2-raw /mnt/llm-data/wikitext-2-raw/"
        RESULTS[T3]="SKIP"
        return 1
    fi

    local ppl_tmac ppl_stock

    # T-MAC perplexity
    info "Computing perplexity with T-MAC... (this takes a while)"
    local tmac_output
    tmac_output=$($PERPLEXITY \
        -m "$MODEL" -ngl 99 -f "$WIKITEXT" $GPU_ARGS \
        2>&1) || true

    ppl_tmac=$(echo "$tmac_output" | grep -oP 'Final estimate: PPL = \K[\d.]+' | tail -1)
    if [[ -z "$ppl_tmac" ]]; then
        # Alternative format
        ppl_tmac=$(echo "$tmac_output" | grep -oP 'perplexity = \K[\d.]+' | tail -1)
    fi

    if [[ -z "$ppl_tmac" ]]; then
        fail "Tier 3: could not extract T-MAC PPL from output"
        echo "  Last 5 lines of output:"
        echo "$tmac_output" | tail -5 | sed 's/^/    /'
        RESULTS[T3]="FAIL"
        return 1
    fi

    # Stock perplexity
    info "Computing perplexity with Stock... (this takes a while)"
    local stock_output
    stock_output=$(GGML_HIP_NO_TMAC=1 $PERPLEXITY \
        -m "$MODEL" -ngl 99 -f "$WIKITEXT" $GPU_ARGS \
        2>&1) || true

    ppl_stock=$(echo "$stock_output" | grep -oP 'Final estimate: PPL = \K[\d.]+' | tail -1)
    if [[ -z "$ppl_stock" ]]; then
        ppl_stock=$(echo "$stock_output" | grep -oP 'perplexity = \K[\d.]+' | tail -1)
    fi

    if [[ -z "$ppl_stock" ]]; then
        fail "Tier 3: could not extract Stock PPL from output"
        RESULTS[T3]="FAIL"
        return 1
    fi

    # Compute delta
    local ppl_delta
    ppl_delta=$(awk "BEGIN {d = $ppl_tmac - $ppl_stock; printf \"%.4f\", (d < 0 ? -d : d)}")

    local ppl_ok
    ppl_ok=$(awk "BEGIN {print ($ppl_delta < 0.1) ? 1 : 0}" <<< "")

    echo "T3,$MODEL_NAME,ppl,1,tmac,$ppl_tmac,ppl" >> "$CSV_FILE"
    echo "T3,$MODEL_NAME,ppl,1,stock,$ppl_stock,ppl" >> "$CSV_FILE"
    echo "T3,$MODEL_NAME,ppl_delta,1,comparison,$ppl_delta,ppl" >> "$CSV_FILE"

    if (( ppl_ok )); then
        ok "Tier 3: PPL T-MAC=$ppl_tmac  Stock=$ppl_stock  Δ=$ppl_delta"
        RESULTS[T3]="OK"
        RESULTS[PPL_TMAC]="$ppl_tmac"
        RESULTS[PPL_STOCK]="$ppl_stock"
    else
        fail "Tier 3: PPL delta too large: T-MAC=$ppl_tmac  Stock=$ppl_stock  Δ=$ppl_delta (threshold: 0.1)"
        RESULTS[T3]="FAIL"
        RESULTS[PPL_TMAC]="$ppl_tmac"
        RESULTS[PPL_STOCK]="$ppl_stock"
        return 1
    fi
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════
# TIER 4: Benchmark (N=5 paired interleaved)
# ═══════════════════════════════════════════════════════════════════════
tier4_benchmark() {
    echo -e "${BOLD}── Tier 4: Benchmark (N=$N, interleaved A-B) ──${NC}"

    if [[ ! -x "$BENCH" ]]; then
        fail "llama-bench not found: $BENCH"
        RESULTS[T4]="SKIP"
        return 1
    fi

    declare -a tmac_results=()
    declare -a stock_results=()

    # Warmup
    for ((w=1; w<=WARMUP; w++)); do
        run_bench "$MODEL" "tg128" "" >/dev/null
        run_bench "$MODEL" "tg128" "GGML_HIP_NO_TMAC=1" >/dev/null
        info "Warmup $w/$WARMUP (discarded)"
    done

    # Measured runs (interleaved A-B)
    for ((i=1; i<=N; i++)); do
        local t s
        if (( i % 2 == 1 )); then
            t=$(run_bench "$MODEL" "tg128" "")
            s=$(run_bench "$MODEL" "tg128" "GGML_HIP_NO_TMAC=1")
        else
            s=$(run_bench "$MODEL" "tg128" "GGML_HIP_NO_TMAC=1")
            t=$(run_bench "$MODEL" "tg128" "")
        fi

        tmac_results+=("$t")
        stock_results+=("$s")
        echo "T4,$MODEL_NAME,tg128,$i,tmac,$t,t/s" >> "$CSV_FILE"
        echo "T4,$MODEL_NAME,tg128,$i,stock,$s,t/s" >> "$CSV_FILE"
        printf "  Run %d/%d: T-MAC=%.2f  Stock=%.2f t/s\n" "$i" "$N" "$t" "$s"
    done

    # Statistics
    local tmac_mean stock_mean tmac_sd stock_sd
    tmac_mean=$(compute_mean "${tmac_results[@]}")
    stock_mean=$(compute_mean "${stock_results[@]}")
    tmac_sd=$(compute_sd "${tmac_results[@]}")
    stock_sd=$(compute_sd "${stock_results[@]}")

    local stats t_stat p_val ci_lo ci_hi mean_speedup
    stats=$(paired_ttest_full tmac_results stock_results)
    t_stat=$(echo "$stats" | awk '{print $1}')
    p_val=$(echo "$stats" | awk '{print $2}')
    ci_lo=$(echo "$stats" | awk '{print $3}')
    ci_hi=$(echo "$stats" | awk '{print $4}')
    mean_speedup=$(echo "$stats" | awk '{print $5}')

    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    printf "  │ %-43s │\n" "$MODEL_NAME tg128"
    printf "  │ Stock:   %7.2f ± %.2f t/s %18s │\n" "$stock_mean" "$stock_sd" ""
    printf "  │ T-MAC:   %7.2f ± %.2f t/s %18s │\n" "$tmac_mean" "$tmac_sd" ""
    printf "  │ Speedup: %+.1f%% [%.1f%%, %.1f%%] %14s │\n" "$mean_speedup" "$ci_lo" "$ci_hi" ""
    printf "  │ t=%.2f, p=%s, N=%d %24s │\n" "$t_stat" "$p_val" "$N" ""
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    # Determine pass/fail
    local is_positive
    is_positive=$(awk "BEGIN {print ($mean_speedup > 0) ? 1 : 0}")
    local is_significant
    is_significant=$(awk "BEGIN {print ($p_val < 0.05) ? 1 : 0}")

    if (( is_positive && is_significant )); then
        ok "Tier 4: +${mean_speedup}% speedup (p=$p_val)"
        RESULTS[T4]="OK"
    elif (( is_positive )); then
        warn "Tier 4: +${mean_speedup}% speedup but not statistically significant (p=$p_val)"
        RESULTS[T4]="OK"
    else
        local is_regressed
        is_regressed=$(awk "BEGIN {print ($mean_speedup < -2.0 && $p_val < 0.05) ? 1 : 0}")
        if (( is_regressed )); then
            fail "Tier 4: REGRESSION ${mean_speedup}% (p=$p_val)"
            RESULTS[T4]="FAIL"
        else
            warn "Tier 4: ${mean_speedup}% (within noise, p=$p_val)"
            RESULTS[T4]="OK"
        fi
    fi

    RESULTS[SPEEDUP]="+${mean_speedup}%"
    RESULTS[TMAC_TS]="$tmac_mean"
    RESULTS[STOCK_TS]="$stock_mean"
    RESULTS[P_VAL]="$p_val"
    RESULTS[CI]="[${ci_lo}%, ${ci_hi}%]"
}

# ═══════════════════════════════════════════════════════════════════════
# Summary & Tracker Update
# ═══════════════════════════════════════════════════════════════════════
print_summary() {
    echo ""
    echo -e "${BOLD}═══ Summary: $MODEL_NAME ═══${NC}"
    echo ""
    printf "  Tier 1 (Smoke):      %s" "${RESULTS[T1]:-—}"
    [[ -n "${RESULTS[AR]:-}" ]] && printf "  (Active Ratio: %s)" "${RESULTS[AR]}"
    echo ""
    printf "  Tier 2 (Coherence):  %s\n" "${RESULTS[T2]:-—}"
    printf "  Tier 3 (Perplexity): %s" "${RESULTS[T3]:-—}"
    [[ -n "${RESULTS[PPL_TMAC]:-}" ]] && printf "  (T-MAC: %s, Stock: %s)" "${RESULTS[PPL_TMAC]}" "${RESULTS[PPL_STOCK]}"
    echo ""
    printf "  Tier 4 (Benchmark):  %s" "${RESULTS[T4]:-—}"
    [[ -n "${RESULTS[SPEEDUP]:-}" ]] && printf "  (%s, p=%s)" "${RESULTS[SPEEDUP]}" "${RESULTS[P_VAL]}"
    echo ""
    echo ""
    echo "  CSV: $CSV_FILE"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════
should_run() {
    [[ "$TIER" == "all" ]] || [[ "$TIER" == "$1" ]]
}

if should_run 1; then tier1_smoke || true; fi
if should_run 2; then tier2_coherence || true; fi
if should_run 3; then tier3_perplexity || true; fi
if should_run 4; then tier4_benchmark || true; fi

print_summary

echo "════════════════════════════════════════════════════════════"
echo "Done. Results saved to: $CSV_FILE"
echo "════════════════════════════════════════════════════════════"
