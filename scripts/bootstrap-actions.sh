#!/usr/bin/env bash
# One-time bootstrap: re-enable GitHub Actions and disable all upstream workflows.
# Run this once after pushing upstream-merge.yml to the repo.
#
# Prerequisites: gh cli authenticated with repo access
#
# Usage:
#   scripts/bootstrap-actions.sh              # execute
#   scripts/bootstrap-actions.sh --dry-run    # print what would happen

set -euo pipefail

REPO="nemekath/kuzco.cpp"
OUR_WORKFLOWS="upstream-merge.yml tmac-ci.yml release.yml"
DRY_RUN=0

[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

log() { printf '\033[1;32m[bootstrap]\033[0m %s\n' "$*"; }

# Step 1: Re-enable Actions (currently disabled repo-wide)
ENABLED=$(gh api "repos/${REPO}/actions/permissions" -q '.enabled')
if [ "$ENABLED" = "false" ]; then
    log "Actions currently disabled. Re-enabling with restricted permissions..."
    if [ "$DRY_RUN" = "0" ]; then
        gh api -X PUT "repos/${REPO}/actions/permissions" \
            -f enabled=true \
            -f allowed_actions=all
    else
        log "[DRY RUN] gh api -X PUT repos/${REPO}/actions/permissions -f enabled=true -f allowed_actions=all"
    fi
else
    log "Actions already enabled."
fi

# Step 2: Wait for GitHub to register workflow files
log "Waiting for GitHub to discover workflow files..."
if [ "$DRY_RUN" = "0" ]; then
    sleep 5
fi

# Step 3: Disable all workflows except ours
log "Fetching workflow list..."
gh api "repos/${REPO}/actions/workflows" --paginate \
    -q '.workflows[] | "\(.id)\t\(.path)\t\(.state)"' | \
while IFS=$'\t' read -r wf_id wf_path wf_state; do
    wf_name=$(basename "$wf_path")
    IS_OURS=false
    for ours in ${OUR_WORKFLOWS}; do
        [ "$wf_name" = "$ours" ] && IS_OURS=true
    done

    if [ "$IS_OURS" = "true" ]; then
        if [ "$wf_state" = "disabled_manually" ]; then
            log "Enabling OUR workflow: ${wf_name}"
            [ "$DRY_RUN" = "0" ] && gh api -X PUT "repos/${REPO}/actions/workflows/${wf_id}/enable"
        else
            log "OK: ${wf_name} (ours, ${wf_state})"
        fi
    else
        if [ "$wf_state" != "disabled_manually" ]; then
            log "Disabling upstream: ${wf_name}"
            [ "$DRY_RUN" = "0" ] && gh api -X PUT "repos/${REPO}/actions/workflows/${wf_id}/disable"
        else
            log "Already disabled: ${wf_name}"
        fi
    fi
done

log "Done. Verify at: https://github.com/${REPO}/actions"
