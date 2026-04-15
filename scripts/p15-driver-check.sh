#!/usr/bin/env bash
# p15-driver-check.sh — Remind to run T-MAC regression after driver updates.
#
# Called by pacman hook (rocm-tmac-regression.hook) after ROCm/amdgpu/kernel
# package updates. Logs the version change and prints a reminder.
#
# Can also be run manually to check current driver state against last
# regression run.
#
# P15: Memory Controller QoS Monitoring

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${KUZCO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_FILE="$REPO_ROOT/data/benchmarks/p15-driver-log.txt"
BASELINE="$REPO_ROOT/data/benchmarks/p15-baseline.csv"

# Current environment
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
KERNEL_VER=$(uname -r)
TIMESTAMP=$(date -Iseconds)

# Find last regression run
LAST_CSV=$(ls -t "$REPO_ROOT"/tmac-regression-*.csv 2>/dev/null | head -1)
if [[ -n "$LAST_CSV" ]]; then
    LAST_ROCM=$(head -1 "$LAST_CSV" | sed -n 's/.*rocm=\([^,]*\).*/\1/p')
    LAST_KERNEL=$(head -1 "$LAST_CSV" | sed -n 's/.*kernel=\([^,]*\).*/\1/p')
    LAST_DATE=$(head -1 "$LAST_CSV" | sed -n 's/.*date=\([^,]*\).*/\1/p')
else
    LAST_ROCM="none"
    LAST_KERNEL="none"
    LAST_DATE="never"
fi

# Log the version change
mkdir -p "$(dirname "$LOG_FILE")"
echo "$TIMESTAMP rocm=$ROCM_VER kernel=$KERNEL_VER last_regression=$LAST_DATE last_rocm=$LAST_ROCM" \
    >> "$LOG_FILE"

# Check if versions changed since last regression
CHANGED=0
if [[ "$ROCM_VER" != "$LAST_ROCM" ]]; then
    CHANGED=1
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  P15: ROCm version changed: $LAST_ROCM → $ROCM_VER"
    echo "║  T-MAC regression test recommended.                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
fi

if [[ "$KERNEL_VER" != "$LAST_KERNEL" ]]; then
    CHANGED=1
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  P15: Kernel changed: $LAST_KERNEL → $KERNEL_VER"
    echo "║  T-MAC regression test recommended (amdgpu driver is in-tree)."
    echo "╚══════════════════════════════════════════════════════════════╝"
fi

if (( CHANGED )); then
    echo ""
    echo "  Run:  cd $REPO_ROOT && scripts/tmac-regression.sh --quick --compare"
    echo "  Full: cd $REPO_ROOT && scripts/tmac-regression.sh --compare"
    echo ""
    echo "  Last regression: $LAST_DATE (ROCm $LAST_ROCM, kernel $LAST_KERNEL)"
    if [[ -f "$BASELINE" ]]; then
        echo "  Baseline: $(head -1 "$BASELINE" | sed 's/^# //')"
    fi
else
    echo "[P15] Driver versions unchanged since last regression ($LAST_DATE)."
fi
