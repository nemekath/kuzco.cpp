#!/usr/bin/env bash
# T-MAC dispatch site health check — run after every upstream rebase.
# Verifies all 6 dispatch sites are intact and the T-MAC marker comments exist.
# Exit code 1 if any site is missing (CI-friendly).

set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

FAIL=0

check_marker() {
    local file="$1" marker="$2" description="$3"
    if grep -q "$marker" "$file" 2>/dev/null; then
        printf "  [OK]  %s — %s\n" "$marker" "$description"
    else
        printf "  [FAIL] %s — %s\n" "$marker" "$description"
        FAIL=1
    fi
}

echo "T-MAC Dispatch Site Health Check"
echo "================================"
echo ""

echo "ggml-cuda.cu:"
check_marker "ggml/src/ggml-cuda/ggml-cuda.cu" "T-MAC dispatch site 1" "Unfused non-split (ne2 host loop)"
check_marker "ggml/src/ggml-cuda/ggml-cuda.cu" "T-MAC dispatch site 2" "Unfused split-tensor callback"
check_marker "ggml/src/ggml-cuda/ggml-cuda.cu" "T-MAC dispatch site 3" "MoE unfused (src1_expert_stride)"
check_marker "ggml/src/ggml-cuda/ggml-cuda.cu" "T-MAC dispatch site 5" "Fused split-tensor callback"
check_marker "ggml/src/ggml-cuda/ggml-cuda.cu" "T-MAC dispatch site 6" "Bias-fused non-split (ne2 loop)"

echo ""
echo "mmvq.cu:"
check_marker "ggml/src/ggml-cuda/mmvq.cu" "T-MAC dispatch site 4" "Fused SwiGLU non-split + MoE"

echo ""
echo "Core files:"
check_marker "ggml/src/ggml-cuda/tmac.cu"  "tmac_gemv_direct"     "Kernel template exists"
check_marker "ggml/src/ggml-cuda/tmac.cuh" "tmac_dispatch_simple" "Central type dispatch exists"
check_marker "ggml/src/ggml-cuda/tmac.cuh" "GGML_HIP_TMAC"       "Build flag guard exists"

echo ""
if (( FAIL )); then
    echo "DISPATCH SITES MISSING — rebase may have broken T-MAC integration"
    exit 1
else
    echo "All 6 dispatch sites + core files intact — PASS"
    exit 0
fi
