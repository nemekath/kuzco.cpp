#!/usr/bin/env bash
# Release validation playbook — run before packaging/uploading any release.
# Gates phases sequentially: structural → smoke → perplexity → A/B regression.
# Early abort on any phase failure.
#
# Usage:
#   scripts/release-validate.sh                # full validation (~60 min)
#   scripts/release-validate.sh --quick        # upstream-merge mode (~5 min)
#   scripts/release-validate.sh --skip-phase4  # skip full A/B regression
#   scripts/release-validate.sh --phase 3      # run only phase 3 (perplexity)

set -euo pipefail
export LC_NUMERIC=C

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# Auto-detect and exclude iGPUs (e.g. gfx1036) to prevent segfaults
source scripts/hip-gpu-guard.sh

# ── Defaults ─────────────────────────────────────────────────────────────
QUICK=0
SKIP_PHASE4=0
ONLY_PHASE=0
BENCH="./build/bin/llama-bench"
COMPLETION="./build/bin/llama-completion"
PERPLEXITY="./build/bin/llama-perplexity"
SMOKE_MODEL="models/1B-Q4_K_M.gguf"
WIKITEXT="/mnt/truenas-llm-models/wikitext-2-raw/wiki.test.raw"
RESULTS_DIR="release-validation-$(date +%Y%m%d-%H%M%S)"

# Perplexity test models — covers stock (Q4_0), K-quant (Q4_K_M), IQ (IQ3_S)
PPL_MODELS=(
    "models/1B-Q4_0.gguf"
    "models/1B-Q4_K_M.gguf"
    "models/1B-IQ3_S.gguf"
)
PPL_THRESHOLD=0.1  # max allowed |PPL_tmac - PPL_stock|

# ── Argument parsing ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)        QUICK=1; shift ;;
        --skip-phase4)  SKIP_PHASE4=1; shift ;;
        --phase)        ONLY_PHASE="$2"; shift 2 ;;
        --smoke-model)  SMOKE_MODEL="$2"; shift 2 ;;
        --help|-h)
            cat <<'USAGE'
Usage: scripts/release-validate.sh [OPTIONS]

Phases (sequential, each gates the next):
  1. Structural    (~5s)    Dispatch sites, build flag, T-MAC files
  2. Quick smoke   (~30s)   Math, text gen, T-MAC activation, single bench
  3. Perplexity    (~10m)   3 quant types, wikitext-2, T-MAC vs stock Δ=0.000
  4. A/B regression (~45m)  Full tmac-regression.sh (15 models, N=5, paired)

Options:
  --quick           Upstream-merge mode: phases 1-3 + regression --quick (N=2)
  --skip-phase4     Skip full A/B regression (phases 1-3 only)
  --phase N         Run only phase N (1-4)
  --smoke-model M   Override smoke test model (default: models/1B-Q4_K_M.gguf)
  --help            Show this help

Examples:
  # Full release validation
  scripts/release-validate.sh

  # Quick check after upstream merge (no kernel changes)
  scripts/release-validate.sh --quick

  # Re-run just perplexity after fixing an issue
  scripts/release-validate.sh --phase 3
USAGE
            exit 0 ;;
        *)  echo "Unknown option: $1 (try --help)"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────
RED='\033[1;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log()  { printf "${CYAN}[validate]${NC} %s\n" "$*"; }
ok()   { printf "${GREEN}  [OK]${NC}  %s\n" "$*"; }
fail() { printf "${RED}  [FAIL]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}  [WARN]${NC} %s\n" "$*"; }

PHASE_RESULTS=()

should_run() {
    local phase=$1
    [[ "$ONLY_PHASE" == "0" || "$ONLY_PHASE" == "$phase" ]]
}

phase_header() {
    local num=$1 name=$2 est=$3
    printf "\n${BOLD}═══ Phase %d: %s (~%s) ═══${NC}\n" "$num" "$name" "$est"
}

# ── Pre-flight checks ───────────────────────────────────────────────────
log "T-MAC Release Validation Playbook"
log "Date: $(date -Iseconds)"
log "Commit: $(git rev-parse --short HEAD)"
log "Mode: $([ "$QUICK" = "1" ] && echo "quick (upstream merge)" || echo "full release")"
echo ""

if [[ ! -x "$BENCH" ]]; then
    fail "llama-bench not found: $BENCH — run cmake/make first"
    exit 1
fi

mkdir -p "$RESULTS_DIR"
log "Results directory: $RESULTS_DIR/"
echo ""

# ═════════════════════════════════════════════════════════════════════════
# PHASE 1: Structural Integrity
# ═════════════════════════════════════════════════════════════════════════
if should_run 1; then
    phase_header 1 "Structural Integrity" "5s"

    if scripts/check-dispatch-sites.sh > "$RESULTS_DIR/phase1-dispatch.txt" 2>&1; then
        ok "All 6 dispatch sites intact"
        cat "$RESULTS_DIR/phase1-dispatch.txt" | grep -E '\[OK\]' | sed 's/^/    /'
    else
        fail "Dispatch site check failed"
        cat "$RESULTS_DIR/phase1-dispatch.txt"
        PHASE_RESULTS+=("Phase 1: FAIL")
        echo ""
        printf "${RED}Phase 1 FAILED — aborting. Fix dispatch sites before continuing.${NC}\n"
        exit 1
    fi

    # Verify T-MAC kernel files exist
    for f in ggml/src/ggml-cuda/tmac.cu ggml/src/ggml-cuda/tmac.cuh; do
        if [[ -f "$f" ]]; then
            ok "$f exists ($(wc -l < "$f") lines)"
        else
            fail "$f missing"
            exit 1
        fi
    done

    PHASE_RESULTS+=("Phase 1: PASS")
    echo ""
fi

# ═════════════════════════════════════════════════════════════════════════
# PHASE 2: Quick Smoke Tests
# ═════════════════════════════════════════════════════════════════════════
if should_run 2; then
    phase_header 2 "Quick Smoke Tests" "30s"

    if [[ ! -f "$SMOKE_MODEL" ]]; then
        fail "Smoke model not found: $SMOKE_MODEL"
        exit 1
    fi

    # Test 1: Math coherence
    log "Test 1/4: Math coherence"
    MATH_OUT=$(echo "" | "$COMPLETION" \
        -m "$SMOKE_MODEL" -p "What is 2+2? Answer with just the number:" \
        -n 10 -ngl 99 2>/dev/null || true)
    if echo "$MATH_OUT" | grep -q "4"; then
        ok "Math coherence (output contains '4')"
    else
        fail "Math coherence — output: $MATH_OUT"
        PHASE_RESULTS+=("Phase 2: FAIL")
        exit 1
    fi

    # Test 2: Text generation produces output
    log "Test 2/4: Text generation"
    TEXT_OUT=$(echo "" | "$COMPLETION" \
        -m "$SMOKE_MODEL" -p "The capital of France is" \
        -n 20 -ngl 99 2>/dev/null || true)
    if [[ -n "$TEXT_OUT" ]]; then
        ok "Text generation ($(echo "$TEXT_OUT" | wc -w) words)"
    else
        fail "Text generation produced no output"
        PHASE_RESULTS+=("Phase 2: FAIL")
        exit 1
    fi

    # Test 3: T-MAC activation
    log "Test 3/4: T-MAC activation"
    TMAC_OUT=$(echo "" | "$COMPLETION" \
        -m "$SMOKE_MODEL" -p "Hello" -n 5 -ngl 99 2>&1 || true)
    if echo "$TMAC_OUT" | grep -q "Active Ratio"; then
        ACTIVE=$(echo "$TMAC_OUT" | grep "Active Ratio" | tail -1 | grep -oP '[\d.]+%' | head -1)
        ok "T-MAC active ($ACTIVE)"
    else
        warn "T-MAC activation not detected in output"
    fi

    # Test 4: Benchmark runs
    log "Test 4/4: Single benchmark"
    if "$BENCH" -m "$SMOKE_MODEL" -t 4 -ngl 99 -n 32 -r 1 \
        > "$RESULTS_DIR/phase2-bench.txt" 2>&1; then
        ok "Benchmark completes successfully"
    else
        fail "Benchmark failed"
        cat "$RESULTS_DIR/phase2-bench.txt"
        PHASE_RESULTS+=("Phase 2: FAIL")
        exit 1
    fi

    PHASE_RESULTS+=("Phase 2: PASS")
    echo ""
fi

# ═════════════════════════════════════════════════════════════════════════
# PHASE 3: Perplexity Verification
# ═════════════════════════════════════════════════════════════════════════
if should_run 3; then
    phase_header 3 "Perplexity Verification" "10min"

    if [[ ! -x "$PERPLEXITY" ]]; then
        fail "llama-perplexity not found: $PERPLEXITY"
        exit 1
    fi
    if [[ ! -f "$WIKITEXT" ]]; then
        fail "Wikitext-2 not found: $WIKITEXT"
        warn "Download: scripts/get-wikitext-2.sh"
        exit 1
    fi

    PPL_PASS=0
    PPL_TOTAL=0

    for model in "${PPL_MODELS[@]}"; do
        model_name=$(basename "$model" .gguf)
        PPL_TOTAL=$((PPL_TOTAL + 1))

        if [[ ! -f "$model" ]]; then
            warn "Skipping $model_name — model file not found"
            continue
        fi

        log "Perplexity: $model_name (T-MAC)"
        tmac_out=$("$PERPLEXITY" \
            -m "$model" -ngl 99 -f "$WIKITEXT" 2>&1) || true
        ppl_tmac=$(echo "$tmac_out" | grep -oP 'Final estimate: PPL = \K[\d.]+' | tail -1)
        [[ -z "$ppl_tmac" ]] && ppl_tmac=$(echo "$tmac_out" | grep -oP 'perplexity = \K[\d.]+' | tail -1)

        if [[ -z "$ppl_tmac" ]]; then
            fail "$model_name: could not extract T-MAC PPL"
            echo "$tmac_out" | tail -3 | sed 's/^/    /'
            continue
        fi

        log "Perplexity: $model_name (stock)"
        stock_out=$(GGML_HIP_NO_TMAC=1 "$PERPLEXITY" \
            -m "$model" -ngl 99 -f "$WIKITEXT" 2>&1) || true
        ppl_stock=$(echo "$stock_out" | grep -oP 'Final estimate: PPL = \K[\d.]+' | tail -1)
        [[ -z "$ppl_stock" ]] && ppl_stock=$(echo "$stock_out" | grep -oP 'perplexity = \K[\d.]+' | tail -1)

        if [[ -z "$ppl_stock" ]]; then
            fail "$model_name: could not extract stock PPL"
            continue
        fi

        ppl_delta=$(awk "BEGIN {d = $ppl_tmac - $ppl_stock; printf \"%.4f\", (d < 0 ? -d : d)}")
        ppl_ok=$(awk "BEGIN {print ($ppl_delta < $PPL_THRESHOLD) ? 1 : 0}")

        echo "$model_name,tmac,$ppl_tmac,stock,$ppl_stock,delta,$ppl_delta" >> "$RESULTS_DIR/phase3-ppl.csv"

        if (( ppl_ok )); then
            ok "$model_name: T-MAC=$ppl_tmac  Stock=$ppl_stock  Δ=$ppl_delta"
            PPL_PASS=$((PPL_PASS + 1))
        else
            fail "$model_name: PPL Δ=$ppl_delta exceeds threshold $PPL_THRESHOLD"
            fail "  T-MAC=$ppl_tmac  Stock=$ppl_stock"
        fi
    done

    if (( PPL_PASS == PPL_TOTAL )); then
        PHASE_RESULTS+=("Phase 3: PASS ($PPL_PASS/$PPL_TOTAL models)")
    else
        PHASE_RESULTS+=("Phase 3: FAIL ($PPL_PASS/$PPL_TOTAL models)")
        printf "\n${RED}Phase 3 FAILED — perplexity regression detected.${NC}\n"
        exit 1
    fi
    echo ""
fi

# ═════════════════════════════════════════════════════════════════════════
# PHASE 4: A/B Regression Benchmark
# ═════════════════════════════════════════════════════════════════════════
if should_run 4 && [[ "$SKIP_PHASE4" != "1" ]]; then
    if [[ "$QUICK" == "1" ]]; then
        phase_header 4 "A/B Regression (quick)" "3min"
        REGR_ARGS="--quick"
    else
        phase_header 4 "A/B Regression (full)" "45min"
        REGR_ARGS=""
    fi

    REGR_CSV="$RESULTS_DIR/phase4-regression.csv"

    if scripts/tmac-regression.sh $REGR_ARGS \
        --csv "$REGR_CSV" 2>&1 | tee "$RESULTS_DIR/phase4-regression.log"; then
        PHASE_RESULTS+=("Phase 4: PASS")
    else
        PHASE_RESULTS+=("Phase 4: FAIL")
        printf "\n${RED}Phase 4 FAILED — performance regression detected.${NC}\n"
        exit 1
    fi
    echo ""
elif should_run 4; then
    PHASE_RESULTS+=("Phase 4: SKIPPED (--skip-phase4)")
fi

# ═════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════
echo ""
printf "${BOLD}═══ Release Validation Summary ═══${NC}\n"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Date:   $(date -Iseconds)"
echo "Mode:   $([ "$QUICK" = "1" ] && echo "quick" || echo "full")"
echo ""
for r in "${PHASE_RESULTS[@]}"; do
    if [[ "$r" == *PASS* ]]; then
        printf "${GREEN}  ✓ %s${NC}\n" "$r"
    elif [[ "$r" == *FAIL* ]]; then
        printf "${RED}  ✗ %s${NC}\n" "$r"
    else
        printf "${YELLOW}  - %s${NC}\n" "$r"
    fi
done
echo ""
log "Results saved to: $RESULTS_DIR/"

ALL_PASS=1
for r in "${PHASE_RESULTS[@]}"; do
    [[ "$r" == *FAIL* ]] && ALL_PASS=0
done

if (( ALL_PASS )); then
    printf "${GREEN}${BOLD}GO — Release validation passed. Safe to package and upload.${NC}\n"
    exit 0
else
    printf "${RED}${BOLD}NO-GO — Fix failures before releasing.${NC}\n"
    exit 1
fi
