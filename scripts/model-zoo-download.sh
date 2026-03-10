#!/usr/bin/env bash
# model-zoo-download.sh — Download models for T-MAC validation.
#
# Usage:
#   scripts/model-zoo-download.sh --wave 1|2|3|4|all [OPTIONS]
#   scripts/model-zoo-download.sh --model REPO_ID --include PATTERN [OPTIONS]
#
# Options:
#   --wave WAVE       Download wave 1-4 or all
#   --model REPO_ID   Download a single model from HuggingFace
#   --include PATTERN Include glob for HF download (default: *Q4_K_M*)
#   --quant QUANT     Shortcut: sets include to *QUANT* (e.g. Q4_K_M, IQ2_XXS)
#   --name FILENAME   Override output filename for symlink
#   --dir DIR         Override download directory (default: auto-select by size)
#   --dry-run         Show what would be downloaded without downloading
#   --help            Show this help

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────
MODEL_DIR="/home/benjamin/llama-tmac-real"            # Symlink directory
PRIMARY_DIR="/mnt/llm-data/models"                    # Primary storage
OVERFLOW_DIR="/mnt/games/kuzco-models"                # Overflow for large models
OVERFLOW_THRESHOLD=$((40 * 1024 * 1024 * 1024))       # 40 GB → use overflow

# ─── Argument parsing ──────────────────────────────────────────────────
WAVE=""
SINGLE_REPO=""
INCLUDE_PATTERN="*Q4_K_M*"
OUTPUT_NAME=""
DOWNLOAD_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wave)    WAVE="$2"; shift 2 ;;
        --model)   SINGLE_REPO="$2"; shift 2 ;;
        --include) INCLUDE_PATTERN="$2"; shift 2 ;;
        --quant)   INCLUDE_PATTERN="*${2}*"; shift 2 ;;
        --name)    OUTPUT_NAME="$2"; shift 2 ;;
        --dir)     DOWNLOAD_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p }' "$0"
            exit 0 ;;
        *)  echo "Unknown option: $1 (try --help)"; exit 1 ;;
    esac
done

if [[ -z "$WAVE" ]] && [[ -z "$SINGLE_REPO" ]]; then
    echo "ERROR: specify --wave or --model"
    echo "Try --help for usage."
    exit 1
fi

# ─── Validate tools ────────────────────────────────────────────────────
if ! command -v hf &>/dev/null; then
    echo "ERROR: 'hf' CLI not found. Install with: pip install huggingface_hub"
    exit 1
fi

# ─── Helpers ────────────────────────────────────────────────────────────

# Download a single model file from HF
# Args: REPO_ID INCLUDE_GLOB [OUTPUT_NAME]
download_model() {
    local repo="$1"
    local include="$2"
    local out_name="${3:-}"
    local dest_dir="${DOWNLOAD_DIR:-$PRIMARY_DIR}"

    echo ""
    echo "── Downloading: $repo ──"
    echo "  Include:  $include"
    echo "  Dest dir: $dest_dir"

    if (( DRY_RUN )); then
        echo "  [DRY RUN] Would download to $dest_dir"
        echo "  [DRY RUN] hf download $repo --include '$include' --local-dir $dest_dir"
        return 0
    fi

    # Ensure dest dir exists
    mkdir -p "$dest_dir"

    # Timestamp marker (files newer than this were downloaded by us)
    local ts_marker
    ts_marker=$(mktemp /tmp/dl-marker-XXXXXX)

    # Download
    hf download "$repo" \
        --include "$include" \
        --local-dir "$dest_dir" \
        --repo-type model

    # Find the downloaded file(s) — only those newer than our timestamp marker
    local downloaded_files
    downloaded_files=$(find "$dest_dir" -maxdepth 1 -name "*.gguf" -newer "$ts_marker" -printf '%f\n' 2>/dev/null || true)

    if [[ -z "$downloaded_files" ]]; then
        # HF might put them in a subdir matching the repo
        downloaded_files=$(find "$dest_dir" -name "*.gguf" -newer "$ts_marker" -printf '%P\n' 2>/dev/null || true)
    fi

    rm -f "$ts_marker"

    if [[ -z "$downloaded_files" ]]; then
        echo "  WARNING: no new .gguf files found after download"
        echo "  Check $dest_dir manually"
        return 1
    fi

    # Create symlinks for each downloaded file
    while IFS= read -r gguf_file; do
        local full_path="$dest_dir/$gguf_file"
        local link_name="${out_name:-$(basename "$gguf_file")}"
        local link_path="$MODEL_DIR/$link_name"

        if [[ -e "$link_path" ]]; then
            echo "  Symlink already exists: $link_path"
        else
            ln -s "$full_path" "$link_path"
            echo "  Created symlink: $link_path → $full_path"
        fi
    done <<< "$downloaded_files"

    echo "  Done."
}

# ─── Wave Definitions ──────────────────────────────────────────────────
# Format: "REPO_ID|INCLUDE_GLOB|OUTPUT_NAME"
# OUTPUT_NAME is optional (empty = use original filename)

wave1_models() {
    cat <<'EOF'
Qwen/Qwen3-8B-GGUF|*Q4_K_M*|Qwen3-8B-Q4_K_M.gguf
Qwen/Qwen3-4B-GGUF|*Q4_K_M*|Qwen3-4B-Q4_K_M.gguf
bartowski/phi-4-GGUF|*Q4_K_M*|phi-4-14B-Q4_K_M.gguf
bartowski/microsoft_Phi-4-mini-instruct-GGUF|*Q4_K_M*|phi-4-mini-3.8B-Q4_K_M.gguf
bartowski/google_gemma-3-12b-it-GGUF|*Q4_K_M*|gemma-3-12b-it-Q4_K_M.gguf
bartowski/google_gemma-3-4b-it-GGUF|*Q4_K_M*|gemma-3-4b-it-Q4_K_M.gguf
unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF|*Q4_K_M*|DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf
bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF|*Q4_K_M*|DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
bartowski/Qwen2.5-Coder-7B-Instruct-GGUF|*Q4_K_M*|Qwen2.5-Coder-7B-Q4_K_M.gguf
bartowski/Mistral-Nemo-Instruct-2407-GGUF|*Q4_K_M*|Mistral-Nemo-12B-Q4_K_M.gguf
bartowski/Mistral-7B-Instruct-v0.3-GGUF|*Q4_K_M*|Mistral-7B-v0.3-Q4_K_M.gguf
Qwen/Qwen3-VL-8B-Instruct-GGUF|*Q4_K_M*|Qwen3-VL-8B-Q4_K_M.gguf
EOF
}

wave2_models() {
    cat <<'EOF'
Qwen/Qwen3-32B-GGUF|*Q4_K_M*|Qwen3-32B-Q4_K_M.gguf
Qwen/Qwen3-14B-GGUF|*Q4_K_M*|Qwen3-14B-Q4_K_M.gguf
bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF|*Q4_K_M*|DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf
bartowski/google_gemma-3-27b-it-GGUF|*Q4_K_M*|gemma-3-27b-it-Q4_K_M.gguf
bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF|*Q4_K_M*|Mistral-Small-3.2-24B-Q4_K_M.gguf
bartowski/Qwen2.5-Coder-32B-Instruct-GGUF|*Q4_K_M*|Qwen2.5-Coder-32B-Q4_K_M.gguf
Qwen/Qwen3-30B-A3B-GGUF|*Q4_K_M*|Qwen3-30B-A3B-Q4_K_M.gguf
EOF
}

wave3_models() {
    cat <<'EOF'
bartowski/Llama-3.3-70B-Instruct-GGUF|*Q4_K_M*|Llama-3.3-70B-Q4_K_M.gguf
Qwen/Qwen3-72B-GGUF|*Q4_K_M*|Qwen3-72B-Q4_K_M.gguf
bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF|*Q4_K_M*|DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf
bartowski/Qwen2.5-72B-Instruct-GGUF|*Q4_K_M*|Qwen2.5-72B-Q4_K_M.gguf
EOF
}

wave4_models() {
    cat <<'EOF'
unsloth/Qwen3-235B-A22B-GGUF|*IQ2_XXS*|Qwen3-235B-A22B-IQ2_XXS.gguf
unsloth/DeepSeek-R1-0528-GGUF|*IQ2_XXS*|DeepSeek-R1-0528-671B-IQ2_XXS.gguf
EOF
}

# ─── Run wave ───────────────────────────────────────────────────────────
run_wave() {
    local wave_num="$1"
    local wave_func="wave${wave_num}_models"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Wave $wave_num Downloads"
    echo "═══════════════════════════════════════════════════════════════"

    # Wave 3-4 default to overflow directory
    if [[ -z "$DOWNLOAD_DIR" ]] && (( wave_num >= 3 )); then
        DOWNLOAD_DIR="$OVERFLOW_DIR"
        echo "  Using overflow directory: $OVERFLOW_DIR"
        mkdir -p "$OVERFLOW_DIR"
    fi

    local count=0
    local skipped=0

    while IFS='|' read -r repo include out_name; do
        [[ -z "$repo" ]] && continue

        # Check if already downloaded (symlink exists in MODEL_DIR)
        if [[ -n "$out_name" ]] && [[ -e "$MODEL_DIR/$out_name" ]]; then
            echo ""
            echo "── SKIP: $out_name (already exists) ──"
            (( skipped++ )) || true
            continue
        fi

        download_model "$repo" "$include" "$out_name"
        (( count++ )) || true
    done < <($wave_func)

    echo ""
    echo "Wave $wave_num complete: $count downloaded, $skipped skipped"
}

# ─── Single model download ─────────────────────────────────────────────
if [[ -n "$SINGLE_REPO" ]]; then
    download_model "$SINGLE_REPO" "$INCLUDE_PATTERN" "$OUTPUT_NAME"
    exit 0
fi

# ─── Wave dispatch ──────────────────────────────────────────────────────
case "$WAVE" in
    1)   run_wave 1 ;;
    2)   run_wave 2 ;;
    3)   run_wave 3 ;;
    4)   run_wave 4 ;;
    all)
        for w in 1 2 3 4; do
            run_wave $w
        done ;;
    *)   echo "ERROR: invalid wave: $WAVE (must be 1-4 or all)"; exit 1 ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "All requested downloads complete."
echo "Model directory: $MODEL_DIR"
echo "═══════════════════════════════════════════════════════════════"
