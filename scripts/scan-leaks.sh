#!/usr/bin/env bash
# scan-leaks.sh — Scan tracked files for leaked paths and secrets.
# Run before releases or after large merges.
#
# Usage:
#   scripts/scan-leaks.sh              # scan all tracked files
#   scripts/scan-leaks.sh --staged     # scan only staged files
#   scripts/scan-leaks.sh --fix        # show suggested replacements
#
# Local extension: drop additional patterns into .git/hooks/local-patterns
# (one rule per line, format: "regex|description"). That file is per-clone
# and is never tracked by git.

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

# ── Generic patterns ─────────────────────────────────────────────────────
# Format: "grep_pattern|category|description|fix_suggestion"
PATTERNS=(
  'sk-[a-zA-Z0-9]{20,}|SECRET|OpenAI/Anthropic API key|Use env var or .env file'
  'ghp_[a-zA-Z0-9]{36}|SECRET|GitHub Personal Access Token|Use gh auth or env var'
  'github_pat_[a-zA-Z0-9_]{80,}|SECRET|GitHub Fine-Grained PAT|Use gh auth or env var'
  'hf_[a-zA-Z0-9]{34}|SECRET|HuggingFace token|Use hf login or env var'
  'AKIA[0-9A-Z]{16}|SECRET|AWS Access Key ID|Use aws configure or env var'
)

# Load optional local pattern extensions (per-clone, never tracked).
LOCAL_PATTERNS_FILE="$(git rev-parse --git-dir)/hooks/local-patterns"
if [[ -f "$LOCAL_PATTERNS_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    pattern="${line%%|*}"
    desc="${line#*|}"
    PATTERNS+=("$pattern|LOCAL|$desc|See .git/hooks/local-patterns")
  done < "$LOCAL_PATTERNS_FILE"
fi

EXCLUDE="scripts/hooks/pre-commit|scripts/scan-leaks\.sh"

if [[ "$MODE" == "staged" ]]; then
    FILES=$(git diff --cached --name-only 2>/dev/null)
else
    FILES=$(git ls-files 2>/dev/null)
fi

if [[ -z "$FILES" ]]; then
    echo "No files to scan."
    exit 0
fi

FILES=$(echo "$FILES" | grep -vE "$EXCLUDE" || true)

TOTAL_HITS=0
declare -A CATEGORY_COUNTS

printf "${CYAN}Path/Secret Scanner${NC}\n"
printf "Mode: %s | Patterns: %d\n\n" "$MODE" "${#PATTERNS[@]}"

for rule in "${PATTERNS[@]}"; do
    IFS='|' read -r pattern category description fix <<< "$rule"

    hits=$(echo "$FILES" | tr '\n' '\0' | xargs -0 grep -IrnE "$pattern" 2>/dev/null || true)

    if [[ -n "$hits" ]]; then
        count=$(echo "$hits" | wc -l)
        TOTAL_HITS=$((TOTAL_HITS + count))
        CATEGORY_COUNTS[$category]=$(( ${CATEGORY_COUNTS[$category]:-0} + count ))

        if [[ "$category" == "SECRET" ]]; then
            printf "${RED}[SECRET]${NC} %s (%d hits)\n" "$description" "$count"
        else
            printf "${YELLOW}[%s]${NC}  %s (%d hits)\n" "$category" "$description" "$count"
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $TOTAL_HITS -eq 0 ]]; then
    printf "${GREEN}Clean — no paths or secrets found.${NC}\n"
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
