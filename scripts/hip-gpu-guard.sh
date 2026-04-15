#!/usr/bin/env bash
# HIP GPU guard — detects iGPUs and sets HIP_VISIBLE_DEVICES to exclude them.
# Source this script at the top of any script that uses HIP/ROCm GPUs.
#
# Behavior:
#   - If HIP_VISIBLE_DEVICES is already set: validate it (warn if it includes iGPUs)
#   - If not set: auto-detect and exclude iGPUs, export HIP_VISIBLE_DEVICES
#   - If only iGPUs available: error and exit
#
# Usage:
#   source scripts/hip-gpu-guard.sh          # auto-detect and set
#   source scripts/hip-gpu-guard.sh --warn   # warn only, don't modify env
#
# Known iGPU GFX versions (segfault on HIP compute):
#   gfx1036 (RDNA2 iGPU, Ryzen 7000/9000)
#   gfx1035 (RDNA2 iGPU, Ryzen 6000)
#   gfx1034 (RDNA2 iGPU, older APUs)
#   gfx1033 (RDNA2 iGPU)
#   gfx1013 (Vega iGPU)
#   gfx902  (Vega iGPU, Ryzen 2000/3000)
#   gfx90c  (Vega iGPU, Ryzen 4000/5000)

_HIP_GUARD_WARN_ONLY=0
[[ "${1:-}" == "--warn" ]] && _HIP_GUARD_WARN_ONLY=1

_hip_guard_log() { printf '\033[1;36m[gpu-guard]\033[0m %s\n' "$*" >&2; }
_hip_guard_warn() { printf '\033[1;33m[gpu-guard]\033[0m %s\n' "$*" >&2; }
_hip_guard_err() { printf '\033[1;31m[gpu-guard]\033[0m %s\n' "$*" >&2; }

# iGPU GFX patterns (substring match)
_IGPU_PATTERNS="gfx1036|gfx1035|gfx1034|gfx1033|gfx1013|gfx902|gfx90c"

_hip_gpu_guard() {
    # Check if rocm-smi is available
    if ! command -v rocm-smi &>/dev/null; then
        _hip_guard_warn "rocm-smi not found — cannot detect iGPUs, proceeding without guard"
        return 0
    fi

    # Parse GPU list: index → gfx version
    local gpu_info
    gpu_info=$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | \
        sed -n 's/GPU\[\([0-9]*\)\].*GFX Version:[[:space:]]*\(gfx[a-z0-9]*\)/\1 \2/p')

    if [[ -z "$gpu_info" ]]; then
        _hip_guard_warn "Could not parse GPU info from rocm-smi — proceeding without guard"
        return 0
    fi

    local discrete_ids=()
    local igpu_ids=()
    local all_ids=()

    while read -r idx gfx; do
        all_ids+=("$idx")
        if echo "$gfx" | grep -qE "$_IGPU_PATTERNS"; then
            igpu_ids+=("$idx")
        else
            discrete_ids+=("$idx")
        fi
    done <<< "$gpu_info"

    # No iGPUs detected — nothing to do
    if [[ ${#igpu_ids[@]} -eq 0 ]]; then
        return 0
    fi

    # Only iGPUs — fatal error
    if [[ ${#discrete_ids[@]} -eq 0 ]]; then
        _hip_guard_err "Only iGPUs detected (${igpu_ids[*]}). No discrete GPU available."
        _hip_guard_err "T-MAC requires a discrete AMD GPU (RDNA3+)."
        return 1
    fi

    # Mix of discrete + iGPU
    local discrete_csv
    discrete_csv=$(IFS=,; echo "${discrete_ids[*]}")

    if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
        # Already set — validate
        local has_igpu=0
        for igpu in "${igpu_ids[@]}"; do
            if echo ",$HIP_VISIBLE_DEVICES," | grep -q ",$igpu,"; then
                has_igpu=1
                break
            fi
        done
        if [[ "$has_igpu" == "1" ]]; then
            _hip_guard_warn "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES includes iGPU(s) — may segfault"
            _hip_guard_warn "Recommended: HIP_VISIBLE_DEVICES=$discrete_csv"
        fi
    else
        # Not set — auto-configure
        if [[ "$_HIP_GUARD_WARN_ONLY" == "1" ]]; then
            _hip_guard_warn "iGPU detected (GPU ${igpu_ids[*]}). Set HIP_VISIBLE_DEVICES=$discrete_csv"
        else
            export HIP_VISIBLE_DEVICES="$discrete_csv"
            _hip_guard_log "iGPU excluded — HIP_VISIBLE_DEVICES=$discrete_csv"
        fi
    fi
}

_hip_gpu_guard
