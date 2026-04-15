#!/usr/bin/env bash
# Creates model symlinks in models/ for CI and local testing.
# Idempotent: skips existing symlinks, warns on missing targets.
#
# Usage: scripts/setup-test-models.sh
#   MODEL_DIR=/path/to/models scripts/setup-test-models.sh

set -euo pipefail

if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "ERROR: MODEL_DIR is not set."
    echo "Set MODEL_DIR to the directory containing .gguf model files."
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LINK_DIR="${REPO_ROOT}/models"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Set MODEL_DIR to the directory containing .gguf model files."
    exit 1
fi

mkdir -p "$LINK_DIR"

# Map: short name → filename (without .gguf extension)
declare -A MODELS=(
    # Llama 1B variants
    [1B-Q4_0]="Llama-3.2-1B-Instruct-Q4_0"
    [1B-Q4_K_M]="Llama-3.2-1B-Instruct-Q4_K_M"
    [1B-Q5_0]="Llama-3.2-1B-Instruct-Q5_0"
    [1B-Q5_K_M]="Llama-3.2-1B-Instruct-Q5_K_M"
    [1B-IQ1_M]="Llama-3.2-1B-Instruct-IQ1_M"
    [1B-IQ2_XS]="Llama-3.2-1B-Instruct-IQ2_XS"
    [1B-IQ2_XXS]="Llama-3.2-1B-Instruct-IQ2_XXS"
    [1B-IQ3_S]="Llama-3.2-1B-Instruct-IQ3_S"
    [1B-IQ3_XXS]="Llama-3.2-1B-Instruct-IQ3_XXS"
    [1B-IQ4_XS]="Llama-3.2-1B-Instruct-IQ4_XS"
    # Codestral 22B
    [22B-Q4_0]="codestral-22b-v0.1-q4_0"
    [22B-Q4_K_M]="Codestral-22B-v0.1-Q4_K_M"
    # Llama 8B
    [8B-Q5_K_M]="Meta-Llama-3.1-8B-Instruct-Q5_K_M"
    # Nemotron 30B (MoE)
    [Nemotron-30B-Q4_0]="Nemotron-3-Nano-30B-Q4_0"
    [Nemotron-30B-Q4_K_M]="Nemotron-3-Nano-30B-Q4_K_M"
    # OLMoE
    [OLMoE-Q4_0]="olmoe-1b-7b-0924-instruct-q4_0"
    [OLMoE-Q4_K_M]="olmoe-1b-7b-0924-instruct-q4_k_m"
    [OLMoE-IQ3_S]="olmoe-1b-7b-0924-instruct-iq3_s"
    [OLMoE-IQ3_XXS]="olmoe-1b-7b-0924-instruct-iq3_xxs"
    # Mixtral
    [Mixtral-8x7B-IQ3_S]="Mixtral-8x7B-Instruct-v0.1.i1-IQ3_S"
    # Qwen 122B
    [122B-IQ2_XXS]="Qwen_Qwen3.5-122B-A10B-IQ2_XXS"
)

CREATED=0
SKIPPED=0
MISSING=0
DANGLING=0

for short in "${!MODELS[@]}"; do
    target="${MODEL_DIR}/${MODELS[$short]}.gguf"
    link="${LINK_DIR}/${short}.gguf"

    # Existing symlink: check if target is still valid
    if [[ -L "$link" ]]; then
        if [[ -e "$link" ]]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        else
            echo "WARN: Dangling symlink: ${short}.gguf -> $(readlink "$link")"
            rm -f "$link"
            DANGLING=$((DANGLING + 1))
            # Fall through to re-create
        fi
    fi

    if [[ ! -f "$target" ]]; then
        echo "WARN: ${MODELS[$short]}.gguf not found in $MODEL_DIR"
        MISSING=$((MISSING + 1))
        continue
    fi

    ln -sf "$target" "$link"
    CREATED=$((CREATED + 1))
done

echo "Model symlinks: ${CREATED} created, ${SKIPPED} skipped (existing), ${MISSING} missing, ${DANGLING} dangling (repaired)"
