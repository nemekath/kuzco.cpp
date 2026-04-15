#!/bin/bash
# ROCm 7.2 test wrapper — runs kuzco.cpp in an isolated container
# with GPU passthrough. Host ROCm 7.1.1 is untouched.
#
# The container persists between commands so build artifacts survive.
# Use "destroy" to remove it, or it auto-removes on "shell" exit.
#
# Usage:
#   ./rocm72-test.sh              # interactive shell (auto-removes on exit)
#   ./rocm72-test.sh build        # cmake + make inside container
#   ./rocm72-test.sh bench        # quick T-MAC benchmark
#   ./rocm72-test.sh smoke        # correctness smoke test
#   ./rocm72-test.sh regression   # paired regression test
#   ./rocm72-test.sh info         # show ROCm 7.2 environment
#   ./rocm72-test.sh destroy      # remove persistent container
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="kuzco-rocm72-test"
CONTAINER_NAME="kuzco-rocm72"
MODEL_DIR="${KUZCO_MODEL_DIR:?Set KUZCO_MODEL_DIR to your model directory}"
CMD="${1:-shell}"

# Build image if it doesn't exist.
# Uses a temp dir as build context so Docker doesn't send the entire repo (~GB).
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building test container (first time only)..."
    BUILD_CTX=$(mktemp -d)
    cp "$SCRIPT_DIR/Dockerfile.rocm72-test" "$SCRIPT_DIR/rocm72-entrypoint.sh" "$BUILD_CTX/"
    docker build \
        -f "$BUILD_CTX/Dockerfile.rocm72-test" \
        -t "$IMAGE_NAME" \
        "$BUILD_CTX"
    rm -rf "$BUILD_CTX"
fi

# Handle destroy command
if [ "$CMD" = "destroy" ]; then
    docker rm -f "$CONTAINER_NAME" 2>/dev/null && echo "Container removed." || echo "No container to remove."
    exit 0
fi

# Ensure persistent container is running
if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true; then
    # Remove stopped container if it exists
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo "Starting persistent ROCm 7.2 container..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --group-add "$(stat -c '%g' /dev/kfd)" \
        --security-opt seccomp=unconfined \
        -v "$SCRIPT_DIR:/src:ro" \
        -v "$MODEL_DIR:/models:ro" \
        -v "$MODEL_DIR:$MODEL_DIR:ro" \
        ${KUZCO_STORAGE_DIR:+-v "$KUZCO_STORAGE_DIR:$KUZCO_STORAGE_DIR:ro"} \
        -e HIP_VISIBLE_DEVICES=0 \
        --entrypoint sleep \
        "$IMAGE_NAME" \
        infinity >/dev/null
fi

# Interactive shell gets special treatment — attach directly, remove on exit
if [ "$CMD" = "shell" ]; then
    exec docker exec -it "$CONTAINER_NAME" /usr/local/bin/entrypoint.sh shell
fi

# All other commands: exec into the persistent container
exec docker exec -t "$CONTAINER_NAME" /usr/local/bin/entrypoint.sh "$@"
