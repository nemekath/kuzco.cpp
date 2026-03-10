#!/usr/bin/env bash
# Release packaging convenience wrapper.
# Builds both Ubuntu 22.04 (container) and local (Arch/current) packages,
# then optionally uploads to GitHub release.
#
# Usage:
#   scripts/release.sh --tag v2.0.0              # build both packages
#   scripts/release.sh --tag v2.0.0 --dry-run    # print what would happen
#   scripts/release.sh --tag v2.0.0 --upload      # build + upload to GitHub
#   scripts/release.sh --tag v2.0.0 --local-only  # skip container build
#   scripts/release.sh --tag v2.0.0 --container-only  # skip local build

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="kuzco-release-builder"
MODEL_DIR="${KUZCO_MODEL_DIR:-/models}"
RELEASE_DIR="${REPO_ROOT}/release"

TAG=""
DRY_RUN=0
UPLOAD=0
LOCAL_ONLY=0
CONTAINER_ONLY=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --tag TAG           Release tag (required, e.g. v2.0.0)
  --dry-run           Print what would happen without executing
  --upload            Upload packages to GitHub release after building
  --local-only        Only build local package (skip container)
  --container-only    Only build container package (skip local)
  --model-dir DIR     Model directory (default: ${MODEL_DIR})
  -h, --help          Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)        TAG="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --upload)     UPLOAD=1; shift ;;
        --local-only) LOCAL_ONLY=1; shift ;;
        --container-only) CONTAINER_ONLY=1; shift ;;
        --model-dir)  MODEL_DIR="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

[ -z "$TAG" ] && { echo "Error: --tag is required"; usage; }

log() { printf '\033[1;32m[release]\033[0m %s\n' "$*"; }

mkdir -p "$RELEASE_DIR"

# ── Container Build (Ubuntu 22.04) ────────────────────────────────────
if [ "$LOCAL_ONLY" != "1" ]; then
    log "=== Container build (Ubuntu 22.04 + ROCm 7.1) ==="

    # Build Docker image if needed
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        log "Building Docker image ${IMAGE_NAME}..."
        if [ "$DRY_RUN" = "1" ]; then
            log "[DRY RUN] docker build -t ${IMAGE_NAME} ${REPO_ROOT}/ci/release-build/"
        else
            docker build -t "$IMAGE_NAME" "${REPO_ROOT}/ci/release-build/"
        fi
    else
        log "Docker image ${IMAGE_NAME} already exists"
    fi

    if [ "$DRY_RUN" = "1" ]; then
        log "[DRY RUN] Would run container build with GPU passthrough"
        log "[DRY RUN] Models: ${MODEL_DIR} (read-only)"
        log "[DRY RUN] Output: ${RELEASE_DIR}/"
    else
        log "Running container build..."
        docker run --rm \
            --device /dev/kfd --device /dev/dri \
            --user "$(id -u):$(id -g)" \
            -v "${REPO_ROOT}:/src:ro" \
            -v "${MODEL_DIR}:/models:ro" \
            -v "${RELEASE_DIR}:/out" \
            -e "TAG=${TAG}" \
            -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}" \
            -w /src \
            "$IMAGE_NAME" \
            bash ci/release-build/build-and-test.sh
    fi
fi

# ── Local Build ───────────────────────────────────────────────────────
if [ "$CONTAINER_ONLY" != "1" ]; then
    log "=== Local build ($(uname -r)) ==="

    if [ "$DRY_RUN" = "1" ]; then
        log "[DRY RUN] Would run build-and-test.sh locally"
        TAG="$TAG" MODEL_DIR="$MODEL_DIR" OUT_DIR="$RELEASE_DIR" DRY_RUN=1 \
            bash "${REPO_ROOT}/ci/release-build/build-and-test.sh"
    else
        cd "$REPO_ROOT"
        TAG="$TAG" MODEL_DIR="$MODEL_DIR" OUT_DIR="$RELEASE_DIR" \
            bash ci/release-build/build-and-test.sh
    fi
fi

# ── Upload ────────────────────────────────────────────────────────────
if [ "$UPLOAD" = "1" ]; then
    log "=== Upload to GitHub release ${TAG} ==="
    TARBALLS=("${RELEASE_DIR}"/kuzco-cpp-${TAG}-*.tar.gz)

    if [ ${#TARBALLS[@]} -eq 0 ]; then
        echo "Error: No packages found in ${RELEASE_DIR}/" >&2
        exit 1
    fi

    for tb in "${TARBALLS[@]}"; do
        log "Uploading $(basename "$tb")..."
        if [ "$DRY_RUN" = "1" ]; then
            log "[DRY RUN] gh release upload ${TAG} ${tb} --repo nemekath/kuzco.cpp"
        else
            gh release upload "$TAG" "$tb" --repo nemekath/kuzco.cpp
        fi
    done
fi

log "Done. Packages in ${RELEASE_DIR}/:"
ls -lh "${RELEASE_DIR}"/kuzco-cpp-${TAG}-*.tar.gz 2>/dev/null || log "(no packages found — dry run?)"
