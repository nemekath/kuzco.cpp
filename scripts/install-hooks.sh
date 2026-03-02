#!/usr/bin/env bash
# install-hooks.sh — Configure git to use our hooks directory
#
# Uses core.hooksPath (Git 2.9+) instead of symlinking into .git/hooks/.
# Run once after clone. Safe to re-run.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Ensure hooks are executable
chmod +x scripts/hooks/*

# Point git at our hooks directory
git config core.hooksPath scripts/hooks

echo "Git hooks installed: scripts/hooks/"
echo "Active hooks:"
ls -1 scripts/hooks/
