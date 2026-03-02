#!/usr/bin/env bash
# backup.sh — Post-commit backup to TrueNAS NFS
#
# Backs up git state (bundle), docs, memories, and scripts to NFS mount.
# Designed to run silently from post-commit hook or manually with --verbose.
#
# Usage:
#   scripts/backup.sh              # Manual run, errors shown
#   scripts/backup.sh --verbose    # Manual run with progress output
#   scripts/backup.sh --hook       # From git hook: silent, always exit 0
#
# Environment:
#   BACKUP_DIR          Backup destination (required, or set a default in your shell profile)

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────

REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ -z "${BACKUP_DIR:-}" ]]; then
    echo "[backup] ERROR: BACKUP_DIR is not set." >&2
    echo "  Set it in your shell profile or pass it inline:" >&2
    echo "  BACKUP_DIR=/path/to/backup scripts/backup.sh" >&2
    exit 1
fi
LOCK_FILE="/tmp/kuzco-backup.lock"
BUNDLE_KEEP=5
CLAUDE_MEMORY_DIR="${CLAUDE_MEMORY_DIR:-}"

# ── Flags ───────────────────────────────────────────────────────────────────

HOOK_MODE=false
VERBOSE=false

for arg in "$@"; do
    case "$arg" in
        --hook)    HOOK_MODE=true ;;
        --verbose) VERBOSE=true ;;
        *)         echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────────

log() {
    if $VERBOSE; then
        echo "[backup] $*"
    fi
}

fail() {
    if $HOOK_MODE; then
        # Hook mode: never show errors, never fail
        exit 0
    else
        echo "[backup] ERROR: $*" >&2
        exit 1
    fi
}

# ── Mount guard ─────────────────────────────────────────────────────────────

if ! mountpoint -q "$(dirname "$BACKUP_DIR")" 2>/dev/null; then
    fail "NFS not mounted at $(dirname "$BACKUP_DIR")"
fi

# Write test — catches stale mounts and permission issues
if ! touch "${BACKUP_DIR}/.write-test" 2>/dev/null; then
    # Maybe the backup dir doesn't exist yet
    if ! mkdir -p "$BACKUP_DIR" 2>/dev/null; then
        fail "Cannot create backup directory: $BACKUP_DIR"
    fi
    if ! touch "${BACKUP_DIR}/.write-test" 2>/dev/null; then
        fail "NFS mount not writable: $BACKUP_DIR"
    fi
fi
rm -f "${BACKUP_DIR}/.write-test"

# ── Concurrency guard (flock) ──────────────────────────────────────────────

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "Another backup is running, skipping"
    $HOOK_MODE && exit 0
    exit 0
fi

# ── Ensure directory structure ──────────────────────────────────────────────

mkdir -p "$BACKUP_DIR"/bundles "$BACKUP_DIR"/docs-tmac "$BACKUP_DIR"/scripts
[[ -d "$REPO_ROOT/.serena/memories" ]] && mkdir -p "$BACKUP_DIR/serena-memories"
[[ -n "$CLAUDE_MEMORY_DIR" && -d "$CLAUDE_MEMORY_DIR" ]] && mkdir -p "$BACKUP_DIR/claude-memories"

# ── Git bundle ──────────────────────────────────────────────────────────────

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_NAME="kuzco-cpp-${TIMESTAMP}.bundle"
BUNDLE_PATH="${BACKUP_DIR}/bundles/${BUNDLE_NAME}"

log "Creating git bundle..."
if ! git -C "$REPO_ROOT" bundle create "$BUNDLE_PATH" --all --quiet 2>/dev/null; then
    fail "git bundle create failed"
fi

# Update latest symlink
ln -sf "$BUNDLE_NAME" "${BACKUP_DIR}/bundles/latest.bundle"
log "Bundle: $BUNDLE_NAME"

# ── Rsync mirrors ──────────────────────────────────────────────────────────

RSYNC_OPTS=(--archive --delete --timeout=10 --quiet)
$VERBOSE && RSYNC_OPTS=(--archive --delete --timeout=10 --verbose)

log "Syncing docs/tmac/..."
rsync "${RSYNC_OPTS[@]}" "$REPO_ROOT/docs/tmac/" "$BACKUP_DIR/docs-tmac/" 2>/dev/null || true

if [[ -d "$REPO_ROOT/.serena/memories" ]]; then
    log "Syncing .serena/memories/..."
    rsync "${RSYNC_OPTS[@]}" "$REPO_ROOT/.serena/memories/" "$BACKUP_DIR/serena-memories/" 2>/dev/null || true
fi

if [[ -n "$CLAUDE_MEMORY_DIR" && -d "$CLAUDE_MEMORY_DIR" ]]; then
    log "Syncing Claude Code memories..."
    rsync "${RSYNC_OPTS[@]}" "$CLAUDE_MEMORY_DIR/" "$BACKUP_DIR/claude-memories/" 2>/dev/null || true
fi

log "Syncing scripts/..."
rsync "${RSYNC_OPTS[@]}" "$REPO_ROOT/scripts/" "$BACKUP_DIR/scripts/" 2>/dev/null || true

# ── Bundle rotation ─────────────────────────────────────────────────────────

log "Rotating bundles (keep last $BUNDLE_KEEP)..."
# List bundles by name (timestamp-sorted due to naming scheme), delete oldest
cd "${BACKUP_DIR}/bundles"
ls -1 kuzco-cpp-*.bundle 2>/dev/null | head -n -"$BUNDLE_KEEP" | while read -r old; do
    log "  Removing old bundle: $old"
    rm -f "$old"
done

# ── Log entry ───────────────────────────────────────────────────────────────

BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUNDLE_SIZE="$(du -h "$BUNDLE_PATH" 2>/dev/null | cut -f1 || echo '?')"
echo "${TIMESTAMP} branch=${BRANCH} commit=${COMMIT} bundle=${BUNDLE_SIZE} ${BUNDLE_NAME}" \
    >> "${BACKUP_DIR}/backup.log"

log "Backup complete: ${BUNDLE_NAME} (${BUNDLE_SIZE})"
