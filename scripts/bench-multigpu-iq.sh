#!/bin/bash
# Multi-GPU IQ Benchmark Campaign
# Systematic dual 7900 XTX benchmarks for IQ quant types
# Protocol: N=12 per config (discard first 2 = N_eff=10), tg128, sequential
# 2026-02-23
set -euo pipefail

BENCH="./bin/llama-bench"
MODEL_DIR="${MODEL_DIR:-./models}"
OUTDIR="/tmp/bench_multigpu_iq_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

N=12  # runs per config (first 2 warmup, effective N=10)

echo "=============================================="
echo " Multi-GPU IQ Benchmark Campaign"
echo " Output: $OUTDIR"
echo " N=$N per config (N_eff=$((N-2)))"
echo " $(date)"
echo "=============================================="

run_bench() {
    local label="$1"
    local model="$2"
    local split="$3"
    local gpus="$4"
    local tmac="$5"
    local extra="${6:-}"
    local outfile="$OUTDIR/${label}.txt"

    echo ""
    echo ">>> [$label] split=$split gpus=$gpus tmac=$tmac $extra"
    echo "    Model: $(basename "$model")"

    local env_prefix="HIP_VISIBLE_DEVICES=$gpus"
    if [ "$tmac" = "off" ]; then
        env_prefix="$env_prefix GGML_HIP_NO_TMAC=1"
    fi

    local split_arg=""
    if [ "$split" = "row" ]; then
        split_arg="--split-mode row"
    elif [ "$split" = "layer" ]; then
        split_arg="--split-mode layer"
    fi

    env $env_prefix ${extra:+$extra} "$BENCH" \
        -m "$model" \
        -p 0 -n 128 -ngl 99 \
        $split_arg \
        -r "$N" 2>&1 | tee "$outfile"

    echo "    Saved: $outfile"
}

# ============================================
# BLOCK 1: 70B IQ2_XXS — THE DECISION GATE
# ============================================
echo ""
echo "=== BLOCK 1: 70B IQ2_XXS (18GB) — Decision Gate ==="

M70_IQ2="$MODEL_DIR/Llama-3.3-70B-Instruct-IQ2_XXS.gguf"

# 1a. Row-split, T-MAC ON
run_bench "70B_IQ2XXS_row_tmac" "$M70_IQ2" "row" "0,1" "on"

# 1b. Row-split, Stock
run_bench "70B_IQ2XXS_row_stock" "$M70_IQ2" "row" "0,1" "off"

# 1c. Layer-split, T-MAC ON
run_bench "70B_IQ2XXS_layer_tmac" "$M70_IQ2" "layer" "0,1" "on"

# 1d. Layer-split, Stock
run_bench "70B_IQ2XXS_layer_stock" "$M70_IQ2" "layer" "0,1" "off"

# ============================================
# BLOCK 2: HSA_ENABLE_SDMA=0 test
# ============================================
echo ""
echo "=== BLOCK 2: HSA_ENABLE_SDMA=0 Test ==="

# Quick test: 5 runs only
N_SDMA=5
N_SAVE=$N
N=$N_SDMA

run_bench "70B_IQ2XXS_row_tmac_nosdma" "$M70_IQ2" "row" "0,1" "on" "HSA_ENABLE_SDMA=0"
run_bench "70B_IQ2XXS_row_stock_nosdma" "$M70_IQ2" "row" "0,1" "off" "HSA_ENABLE_SDMA=0"

N=$N_SAVE

# ============================================
# BLOCK 3: 70B Q4_0 Reference (re-measure N=12)
# ============================================
echo ""
echo "=== BLOCK 3: 70B Q4_0 Row-Split Reference ==="

M70_Q4="$MODEL_DIR/Llama-3.3-70B-Instruct-Q4_0.gguf"

run_bench "70B_Q4_0_row_tmac" "$M70_Q4" "row" "0,1" "on"
run_bench "70B_Q4_0_row_stock" "$M70_Q4" "row" "0,1" "off"

# ============================================
# BLOCK 4: OLMoE IQ3_S — MoE on Multi-GPU
# ============================================
echo ""
echo "=== BLOCK 4: OLMoE IQ3_S Multi-GPU ==="

MOLMOE="$MODEL_DIR/olmoe-1b-7b-0924-instruct-iq3_s.gguf"

# 4a. Single-GPU baseline (re-measure for clean comparison)
run_bench "OLMoE_IQ3S_single_tmac" "$MOLMOE" "none" "0" "on"
run_bench "OLMoE_IQ3S_single_stock" "$MOLMOE" "none" "0" "off"

# 4b. Row-split dual
run_bench "OLMoE_IQ3S_row_tmac" "$MOLMOE" "row" "0,1" "on"
run_bench "OLMoE_IQ3S_row_stock" "$MOLMOE" "row" "0,1" "off"

# 4c. Layer-split dual
run_bench "OLMoE_IQ3S_layer_tmac" "$MOLMOE" "layer" "0,1" "on"
run_bench "OLMoE_IQ3S_layer_stock" "$MOLMOE" "layer" "0,1" "off"

# ============================================
# BLOCK 5: Single-GPU IQ baselines (quick refresh)
# ============================================
echo ""
echo "=== BLOCK 5: Single-GPU IQ Baselines (N=12) ==="

# 70B IQ2_XXS single-GPU (re-measure for clean comparison)
run_bench "70B_IQ2XXS_single_tmac" "$M70_IQ2" "none" "0" "on"
run_bench "70B_IQ2XXS_single_stock" "$M70_IQ2" "none" "0" "off"

# ============================================
# SUMMARY
# ============================================
echo ""
echo "=============================================="
echo " COMPLETE — $(date)"
echo " All results in: $OUTDIR"
echo "=============================================="
echo ""
echo "Files:"
ls -la "$OUTDIR"/*.txt
