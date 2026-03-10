#!/bin/bash
set -euo pipefail

SRC=/src
BUILD=/build
MODELS=/models
DEFAULT_MODEL="Llama-3.2-1B-Instruct-Q4_0.gguf"

# Cache hipconfig results (avoids repeated process spawns)
HIP_CLANG_PATH="$(hipconfig -l 2>/dev/null)/clang" || HIP_CLANG_PATH="unknown"
HIP_ROOT="$(hipconfig -R 2>/dev/null)" || HIP_ROOT="unknown"

require_binary() {
    if [ ! -f "$BUILD/bin/$1" ]; then
        echo "ERROR: Build first with: ./rocm72-test.sh build"
        exit 1
    fi
}

rocm_info() {
    echo "=== ROCm Environment ==="
    echo "HIP version: $(hipconfig --version 2>/dev/null || echo 'unknown')"
    echo "HIP platform: $(hipconfig --platform 2>/dev/null || echo 'unknown')"
    echo "Clang: $("$HIP_CLANG_PATH" --version 2>/dev/null | head -1 || echo 'unknown')"
    rocminfo 2>/dev/null | grep -E '(Name:|Marketing Name:)' | head -4 || true
    echo "========================"
}

do_build() {
    echo "Building kuzco.cpp with ROCm 7.2 (T-MAC enabled, gfx1100 only)..."
    cd "$BUILD"

    HIPCXX="$HIP_CLANG_PATH" HIP_PATH="$HIP_ROOT" \
    cmake -S "$SRC" -B . \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1100 \
        -DGGML_HIP_TMAC=ON \
        -DGGML_HIP_ROCWMMA_FATTN=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        2>&1

    cmake --build . --config Release -j"$(nproc)" 2>&1
    echo ""
    echo "Build complete. Binaries in /build/bin/"
    ls -la bin/llama-bench bin/llama-cli bin/llama-completion 2>/dev/null
}

do_bench() {
    require_binary llama-bench

    local BENCH="$BUILD/bin/llama-bench"
    local MODEL="${1:-$MODELS/$DEFAULT_MODEL}"

    echo "=== Quick Benchmark (ROCm 7.2) ==="
    rocm_info
    echo ""

    # HIP_VISIBLE_DEVICES is set at the container level by the caller
    echo "--- T-MAC ON ---"
    "$BENCH" -p 0 -n 128 -ngl 99 -m "$MODEL"

    echo ""
    echo "--- T-MAC OFF (stock baseline) ---"
    GGML_HIP_NO_TMAC=1 "$BENCH" -p 0 -n 128 -ngl 99 -m "$MODEL"
}

do_smoke() {
    require_binary llama-completion

    echo "=== Smoke Test (ROCm 7.2) ==="
    rocm_info
    echo ""

    echo "--- Llama 1B Q4_0 ---"
    echo "" | "$BUILD/bin/llama-completion" \
        -m "$MODELS/$DEFAULT_MODEL" \
        -p "The answer to 2+2 is" -n 20 -ngl 99 2>&1

    echo ""
    echo "--- Smoke test complete ---"
}

do_regression() {
    require_binary llama-bench

    echo "=== Quick Regression (ROCm 7.2 vs stock) ==="
    rocm_info
    echo ""

    if [ -x "$SRC/scripts/tmac-regression.sh" ]; then
        cd "$BUILD"
        # Map model symlink names to actual /models paths
        local REGRESSION_MODELS=""
        for f in "$MODELS"/*Q4_K_M.gguf; do
            [ -e "$f" ] && REGRESSION_MODELS="$REGRESSION_MODELS $f"
        done
        if [ -z "$REGRESSION_MODELS" ]; then
            REGRESSION_MODELS="$MODELS/$DEFAULT_MODEL"
        fi
        BENCH="$BUILD/bin/llama-bench" "$SRC/scripts/tmac-regression.sh" --quick \
            --models "$REGRESSION_MODELS"
    else
        echo "tmac-regression.sh not found, running manual bench..."
        do_bench "$@"
    fi
}

case "${1:-shell}" in
    build)
        rocm_info
        do_build
        ;;
    bench)
        shift || true
        do_bench "$@"
        ;;
    smoke)
        do_smoke
        ;;
    regression)
        shift || true
        do_regression "$@"
        ;;
    info)
        rocm_info
        ;;
    shell)
        rocm_info
        echo ""
        echo "Commands:  build | bench | smoke | regression | info"
        echo "Source:    /src (read-only mount)"
        echo "Build:    /build"
        echo "Models:   /models (read-only mount)"
        echo ""
        exec /bin/bash
        ;;
    *)
        echo "Usage: $0 {build|bench|smoke|regression|info|shell}"
        exit 1
        ;;
esac
