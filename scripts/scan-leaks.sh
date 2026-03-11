#!/usr/bin/env bash
# scan-leaks.sh — Scan tracked files for leaked paths and secrets.
# Run before releases or after large merges to catch leaks.
#
# Usage:
#   scripts/scan-leaks.sh              # scan all tracked files
#   scripts/scan-leaks.sh --staged     # scan only staged files
#   scripts/scan-leaks.sh --fix        # show suggested replacements

set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

MODE="all"
SHOW_FIX=0
[[ "${1:-}" == "--staged" ]] && MODE="staged"
[[ "${1:-}" == "--fix" ]] && SHOW_FIX=1

RED='\033[1;31m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m'

# ── Patterns ─────────────────────────────────────────────────────────────
# Format: "grep_pattern|category|description|fix_suggestion"
PATTERNS=(
  '/home/benjamin|LOCAL_PATH|Home directory path|Use $KUZCO_MODEL_DIR or relative path'
  '/mnt/truenas|LOCAL_PATH|NFS mount path|Use $BACKUP_DIR or env var'
  '/mnt/llm-data|LOCAL_PATH|Model storage path|Use $KUZCO_STORAGE_DIR'
  '/mnt/games|LOCAL_PATH|Overflow storage path|Use $KUZCO_OVERFLOW_DIR'
  'llama-tmac-real|LOCAL_PATH|Symlink directory name|Use $KUZCO_MODEL_DIR'
  '/mnt/benjamin|LOCAL_PATH|Personal mount path|Use env var'
  'sk-[a-zA-Z0-9]{20,}|SECRET|OpenAI/Anthropic API key|Use env var or .env file'
  'ghp_[a-zA-Z0-9]{36}|SECRET|GitHub Personal Access Token|Use gh auth or env var'
  'github_pat_[a-zA-Z0-9_]{80,}|SECRET|GitHub Fine-Grained PAT|Use gh auth or env var'
  'hf_[a-zA-Z0-9]{34}|SECRET|HuggingFace token|Use hf login or env var'
  'AKIA[0-9A-Z]{16}|SECRET|AWS Access Key ID|Use aws configure or env var'
)

# ── Files to exclude ─────────────────────────────────────────────────────
EXCLUDE="CLAUDE\.md|AGENTS\.md|scripts/backup\.sh|scripts/hooks/pre-commit|scripts/scan-pii\.sh|\.local\.md$"

# ── Get file list ────────────────────────────────────────────────────────
if [[ "$MODE" == "staged" ]]; then
    FILES=$(git diff --cached --name-only 2>/dev/null)
else
    FILES=$(git ls-files 2>/dev/null)
fi

if [[ -z "$FILES" ]]; then
    echo "No files to scan."
    exit 0
fi

# Filter out excluded files
FILES=$(echo "$FILES" | grep -vE "$EXCLUDE" || true)

TOTAL_HITS=0
declare -A CATEGORY_COUNTS

printf "${CYAN}PII/Path Scanner${NC}\n"
printf "Mode: %s | Patterns: %d\n\n" "$MODE" "${#PATTERNS[@]}"

for rule in "${PATTERNS[@]}"; do
    IFS='|' read -r pattern category description fix <<< "$rule"

    hits=$(echo "$FILES" | tr '\n' '\0' | xargs -0 grep -rnE "$pattern" 2>/dev/null || true)

    if [[ -n "$hits" ]]; then
        count=$(echo "$hits" | wc -l)
        TOTAL_HITS=$((TOTAL_HITS + count))
        CATEGORY_COUNTS[$category]=$(( ${CATEGORY_COUNTS[$category]:-0} + count ))

        if [[ "$category" == "SECRET" ]]; then
            printf "${RED}[SECRET]${NC} %s (%d hits)\n" "$description" "$count"
        else
            printf "${YELLOW}[PATH]${NC}   %s (%d hits)\n" "$description" "$count"
        fi

        echo "$hits" | while IFS= read -r line; do
            printf "  %s\n" "$line"
        done

        if [[ "$SHOW_FIX" == "1" ]]; then
            printf "  ${CYAN}Fix: %s${NC}\n" "$fix"
        fi
        echo ""
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $TOTAL_HITS -eq 0 ]]; then
    printf "${GREEN}Clean — no PII, paths, or secrets found.${NC}\n"
    exit 0
else
    printf "${RED}Found %d issue(s):${NC}" "$TOTAL_HITS"
    for cat in "${!CATEGORY_COUNTS[@]}"; do
        printf " %s=%d" "$cat" "${CATEGORY_COUNTS[$cat]}"
    done
    echo ""
    echo "Run with --fix for suggested replacements."
    exit 1
fi
