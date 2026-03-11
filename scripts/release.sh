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

# ── Changelog extraction ─────────────────────────────────────────────
# Extract the body of a changelog section for a given tag.
# Usage: extract_changelog v2.0.1 → outputs markdown body
extract_changelog() {
    local tag="$1"
    awk -v tag="$tag" '
        BEGIN { found=0 }
        /^## v/ {
            if (found) exit
            if (index($0, "## " tag " ") == 1 || $0 == "## " tag) { found=1; next }
        }
        /^---$/ { if (found) exit }
        found { print }
    ' "${REPO_ROOT}/CHANGELOG.md"
}

# Extract just the title line (e.g. "v2.0.1 — PII guardrails...")
extract_title() {
    local tag="$1"
    awk -v tag="$tag" '
        /^## v/ {
            if (index($0, "## " tag " ") == 1 || $0 == "## " tag) {
                sub(/^## /, ""); print; exit
            }
        }
    ' "${REPO_ROOT}/CHANGELOG.md"
}

# ── Changelog validation ─────────────────────────────────────────────
CHANGELOG_BODY="$(extract_changelog "$TAG")"
CHANGELOG_TITLE="$(extract_title "$TAG")"

if [ -z "$CHANGELOG_BODY" ]; then
    echo "Error: No changelog entry found for ${TAG}" >&2
    echo "  Add a '## ${TAG} — ...' section to CHANGELOG.md before releasing." >&2
    exit 1
fi

log "Found changelog entry: ${CHANGELOG_TITLE}"

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

    # Create annotated tag if it doesn't exist locally
    if ! git tag -l "$TAG" | grep -q .; then
        log "Creating annotated tag ${TAG}..."
        if [ "$DRY_RUN" = "1" ]; then
            log "[DRY RUN] git tag -a ${TAG} -m '${CHANGELOG_TITLE}'"
        else
            git tag -a "$TAG" -m "$CHANGELOG_TITLE"
        fi
    else
        log "Tag ${TAG} already exists locally"
    fi

    # Push tag to origin
    log "Pushing tag ${TAG} to origin..."
    if [ "$DRY_RUN" = "1" ]; then
        log "[DRY RUN] git push origin ${TAG}"
    else
        git push origin "$TAG" 2>/dev/null || log "Tag ${TAG} already exists on remote"
    fi

    # Push current branch
    CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    log "Pushing branch ${CURRENT_BRANCH} to origin..."
    if [ "$DRY_RUN" = "1" ]; then
        log "[DRY RUN] git push origin ${CURRENT_BRANCH}"
    else
        git push origin "$CURRENT_BRANCH"
    fi

    # Create or update GitHub release
    if gh release view "$TAG" --repo nemekath/kuzco.cpp &>/dev/null; then
        log "GitHub release ${TAG} exists — uploading assets with --clobber"
        for tb in "${TARBALLS[@]}"; do
            log "Uploading $(basename "$tb")..."
            if [ "$DRY_RUN" = "1" ]; then
                log "[DRY RUN] gh release upload ${TAG} ${tb} --clobber --repo nemekath/kuzco.cpp"
            else
                gh release upload "$TAG" "$tb" --clobber --repo nemekath/kuzco.cpp
            fi
        done
    else
        log "Creating GitHub release ${TAG}..."
        if [ "$DRY_RUN" = "1" ]; then
            log "[DRY RUN] gh release create ${TAG} with notes from CHANGELOG.md"
            for tb in "${TARBALLS[@]}"; do
                log "[DRY RUN]   asset: $(basename "$tb")"
            done
        else
            gh release create "$TAG" "${TARBALLS[@]}" \
                --repo nemekath/kuzco.cpp \
                --title "$CHANGELOG_TITLE" \
                --notes "$CHANGELOG_BODY"
        fi
    fi
fi

log "Done. Packages in ${RELEASE_DIR}/:"
ls -lh "${RELEASE_DIR}"/kuzco-cpp-${TAG}-*.tar.gz 2>/dev/null || log "(no packages found — dry run?)"
