#!/usr/bin/env bash
# Build, test, and package kuzco.cpp release binaries.
# Designed to run inside the Docker container from ci/release-build/Dockerfile,
# but also works natively on any system with ROCm + cmake.
#
# Environment variables:
#   TAG         — release tag (default: read from git describe)
#   MODEL_DIR   — path to GGUF models for smoke tests (default: /models)
#   OUT_DIR     — output directory for tarball (default: /out)
#   SMOKE_MODEL — model filename for smoke tests (default: Llama-3.2-1B-Instruct-Q4_0.gguf)
#   SKIP_TESTS  — set to 1 to skip GPU smoke tests
#   DRY_RUN     — set to 1 to print actions without executing

set -euo pipefail

TAG="${TAG:-$(git describe --tags --always 2>/dev/null || echo "dev")}"
MODEL_DIR="${MODEL_DIR:-/models}"
OUT_DIR="${OUT_DIR:-/out}"
# Resolve to absolute path before we cd into build dir
[[ "$OUT_DIR" != /* ]] && OUT_DIR="$(pwd)/$OUT_DIR"
SMOKE_MODEL="${SMOKE_MODEL:-Llama-3.2-1B-Instruct-Q4_0.gguf}"
SKIP_TESTS="${SKIP_TESTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
# Auto-detect and exclude iGPUs (e.g. gfx1036) to prevent segfaults
GUARD="$(cd "$(dirname "$0")/../.." && pwd)/scripts/hip-gpu-guard.sh"
if [ -f "$GUARD" ]; then
    source "$GUARD"
else
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
fi
JOBS="${JOBS:-$(nproc)}"

# Detect platform for package naming
if [ -f /etc/os-release ]; then
    . /etc/os-release
    PLATFORM="${ID}${VERSION_ID:-}"
else
    PLATFORM="$(uname -s)-$(uname -m)"
fi
ARCH="$(uname -m)"
PACKAGE_NAME="kuzco-cpp-${TAG}-${PLATFORM}-${ARCH}-rocm"

log() { printf '\033[1;36m>>> %s\033[0m\n' "$*"; }
err() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

if [ "$DRY_RUN" = "1" ]; then
    log "[DRY RUN] Would build ${PACKAGE_NAME}"
    log "[DRY RUN] TAG=${TAG} PLATFORM=${PLATFORM} ARCH=${ARCH}"
    log "[DRY RUN] MODEL_DIR=${MODEL_DIR} OUT_DIR=${OUT_DIR}"
    log "[DRY RUN] SMOKE_MODEL=${SMOKE_MODEL}"
    [ "$SKIP_TESTS" = "1" ] && log "[DRY RUN] Tests: SKIPPED" || log "[DRY RUN] Tests: enabled"
    exit 0
fi

# ── Build ──────────────────────────────────────────────────────────────
log "Building ${PACKAGE_NAME} (${JOBS} jobs)"
SRC_DIR="$(pwd)"
# Use /build if it exists and is writable (container), otherwise build in-tree
if [ -d /build ] && [ -w /build ]; then
    BUILD_DIR="/build/build-release"
else
    BUILD_DIR="${SRC_DIR}/build-release"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100

cmake --build . --parallel "$JOBS"
log "Build complete"

# ── GPU Smoke Tests ────────────────────────────────────────────────────
if [ "$SKIP_TESTS" = "1" ]; then
    log "Skipping GPU smoke tests (SKIP_TESTS=1)"
else
    export LD_LIBRARY_PATH="${BUILD_DIR}/bin:${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
    SMOKE="${MODEL_DIR}/${SMOKE_MODEL}"
    [ -f "$SMOKE" ] || err "Smoke model not found: ${SMOKE}"

    log "Smoke test 1/4: math coherence"
    MATH_OUT=$(echo "" | ./bin/llama-completion -m "$SMOKE" \
        -p "What is 2+2? Answer with just the number:" \
        -n 10 -ngl 99 2>/dev/null || true)
    echo "$MATH_OUT" | grep -q "4" || err "Math coherence test failed"

    log "Smoke test 2/4: text generation"
    TEXT_OUT=$(echo "" | ./bin/llama-completion -m "$SMOKE" \
        -p "The capital of France is" \
        -n 20 -ngl 99 2>/dev/null || true)
    [ -n "$TEXT_OUT" ] || err "Text generation produced no output"

    log "Smoke test 3/4: T-MAC activation"
    TMAC_LOG=$(echo "" | ./bin/llama-completion -m "$SMOKE" \
        -p "Hello" -n 5 -ngl 99 2>&1 || true)
    if echo "$TMAC_LOG" | grep -qi "tmac"; then
        log "T-MAC detected in output"
    else
        log "T-MAC not detected (may be normal if model type not supported)"
    fi

    log "Smoke test 4/4: benchmark"
    ./bin/llama-bench -m "$SMOKE" -t 4 -ngl 99 -n 32 -r 1 2>/dev/null \
        || err "Benchmark failed"

    log "All smoke tests passed"
fi

# ── Package ────────────────────────────────────────────────────────────
log "Packaging ${PACKAGE_NAME}"
STAGING="${BUILD_DIR}/staging"
rm -rf "$STAGING"
mkdir -p "${STAGING}/bin" "${STAGING}/lib"

# Binaries
BINARIES=(llama-cli llama-completion llama-bench llama-server llama-quantize llama-perplexity)
for bin in "${BINARIES[@]}"; do
    if [ -f "bin/${bin}" ]; then
        cp "bin/${bin}" "${STAGING}/bin/"
        strip "${STAGING}/bin/${bin}"
    else
        log "Warning: bin/${bin} not found, skipping"
    fi
done

# Libraries
for lib in lib/*.so*; do
    [ -f "$lib" ] || continue
    cp "$lib" "${STAGING}/lib/"
    # Only strip regular files, not symlinks
    [ -L "${STAGING}/lib/$(basename "$lib")" ] || strip "${STAGING}/lib/$(basename "$lib")" 2>/dev/null || true
done

# Metadata
cp "${SRC_DIR}/LICENSE" "${STAGING}/" 2>/dev/null || true
cat > "${STAGING}/README.txt" <<HEREDOC
kuzco.cpp ${TAG}
Built on: $(date -u +"%Y-%m-%d %H:%M UTC")
Platform: ${PLATFORM} ${ARCH}
ROCm:     $(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
glibc:    $(ldd --version 2>&1 | head -1 | grep -oP '[\d.]+$' || echo "unknown")

T-MAC GEMV kernels enabled for RDNA3 (gfx1100).
Opt-out: GGML_HIP_NO_TMAC=1

Usage:
  export HIP_VISIBLE_DEVICES=0
  ./bin/llama-cli -m model.gguf -ngl 99 -p "Hello"
HEREDOC

# Create tarball
mkdir -p "$OUT_DIR"
cd "${STAGING}/.."
TARBALL="${OUT_DIR}/${PACKAGE_NAME}.tar.gz"
tar czf "$TARBALL" -C staging .
log "Package created: ${TARBALL} ($(du -h "$TARBALL" | cut -f1))"
