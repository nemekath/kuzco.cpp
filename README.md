# kuzco.cpp — the fast speaking Llama!

> llama.cpp fork with T-MAC kernels for AMD RDNA3 — **+10-36% token generation throughput.**

<p align="center">
  <img src="docs/kuzco-logo.png" alt="kuzco.cpp logo" width="400">
</p>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/nemekath/kuzco.cpp)](https://github.com/nemekath/kuzco.cpp/releases)

## What is this?

kuzco.cpp is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds **T-MAC** (Table-based Matrix-vector Accumulation) kernels for AMD RDNA3 GPUs.

- Targets **AMD RDNA3** (RX 7900 XTX validated, auto-detects gfx1100/gfx1101/gfx1102)
- **Batch=1 token generation only** — prefill and batched inference fall back to stock automatically
- **26 models** across **13 architecture families**, **17 quantization types**
- **All PPL delta = 0.000** — bit-identical output to stock llama.cpp
- Transparent fallback: non-RDNA3 hardware uses stock kernels, no behavior change

## Why a fork?

kuzco.cpp is developed with AI assistance (Claude, Gemini). llama.cpp's contribution policy does not accept AI-generated code. We respect that boundary — hence an independent fork with monthly rebase against upstream.

This is a specialization for RDNA3 token generation, not a replacement for llama.cpp.

## Acknowledgments

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) by **Georgi Gerganov** and [contributors](https://github.com/ggml-org/llama.cpp/graphs/contributors). The [ggml](https://github.com/ggml-org/ggml) tensor library makes all of this possible. MIT license, same as upstream.

## Performance

All benchmarks: single AMD RX 7900 XTX, `tg128`, N=10 paired interleaved runs, 95% CI via paired t-test. Stock baseline = same binary with `GGML_HIP_NO_TMAC=1`.

### Q4_K_M (mainstream quant)

| Model | Params | Type | Stock | T-MAC | Speedup |
|-------|-------:|------|------:|------:|--------:|
| Llama 3.2 1B | 1.24B | Dense | 373.4 t/s | 449.4 t/s | **+20.3%** |
| Codestral 22B | 22.25B | Dense | 40.0 t/s | 45.7 t/s | **+14.1%** |
| OLMoE-1B-7B | 6.92B | MoE | 324.8 t/s | 372.9 t/s | **+14.8%** |
| GLM-4.7-Flash | ~16B | MLA+MoE | 87.4 t/s | 100.7 t/s | **+15.2%** |
| Ministral 14B | 14B | Dense | 64.0 t/s | 73.0 t/s | **+14.1%** |
| QwQ-32B | 32B | Dense | 29.9 t/s | 33.9 t/s | **+13.5%** |

### IQ types (aggressive quantization)

| Model | Quant | bpw | Speedup |
|-------|-------|----:|--------:|
| Llama 1B | IQ3_XXS | 3.06 | **+36.6%** |
| Llama 1B | IQ3_S | 3.44 | **+34.4%** |
| OLMoE-1B-7B | IQ3_S | 3.44 | **+29.1%** |
| Llama 70B | IQ2_XXS | 2.06 | **+25.8%** |
| Llama 1B | IQ2_XXS | 2.06 | **+24.6%** |
| Llama 1B | IQ2_XS | 2.31 | **+17.4%** |
| Llama 1B | IQ1_M | 1.75 | **+11.9%** |

### Multi-GPU (dual 7900 XTX)

| Model | Quant | Speedup |
|-------|-------|--------:|
| Mixtral 8x7B | IQ3_S | **+45.4%** |
| Llama 70B | IQ2_XXS | **+18.9%** |
| Llama 4 Scout | IQ2_XXS-UD | **+12.0%** |
| Llama 70B | Q4_0 | **+6.5%** |

### Architecture families validated

Dense, MoE, MLA+MoE, SSM-Hybrid (Jamba), SWA, MXFP4-MoE, ISWA-MoE, SSM-MoE, SSM-Hybrid-Mamba2 (Falcon H1), Pure-SSM (Mamba), RWKV, VLM (Qwen2-VL), MoE+CPU-offload.

Full benchmark data with confidence intervals: [docs/tmac/benchmarks.md](docs/tmac/benchmarks.md)

## Quick Start

### Prerequisites

- AMD RDNA3 GPU (RX 7900 series validated)
- [ROCm 6.x or later](https://rocm.docs.amd.com/)
- CMake 3.21+, C++17 compiler

### Build

```bash
git clone https://github.com/nemekath/kuzco.cpp
cd kuzco.cpp
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
make -j$(nproc)
```

T-MAC is enabled by default (`GGML_HIP_TMAC=ON`). No extra flags needed.

### Environment

```bash
# Exclude integrated GPU (prevents segfault on systems with iGPU)
export HIP_VISIBLE_DEVICES=0
```

### Run

```bash
# Interactive chat
./bin/llama-cli -m model.gguf -ngl 99

# Benchmark: compare T-MAC vs stock
./bin/llama-bench -m model.gguf -p 0 -n 128 -ngl 99                         # T-MAC
GGML_HIP_NO_TMAC=1 ./bin/llama-bench -m model.gguf -p 0 -n 128 -ngl 99     # Stock
```

### Disable T-MAC

```bash
export GGML_HIP_NO_TMAC=1    # Falls back to stock llama.cpp kernels
```

## Supported Quantization Types

17 types supported. T-MAC activates automatically when conditions are met (RDNA3 + batch=1 + supported type + alignment).

| Category | Types | bpw range |
|----------|-------|----------:|
| K-quants | Q3_K, Q4_K, Q5_K, Q6_K | 3.44 – 6.56 |
| Legacy | Q4_0, Q5_0, Q5_1, Q8_0 | 4.00 – 8.50 |
| IQ (importance) | IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | 1.75 – 4.25 |
| MXFP | MXFP4 | 4.00 |

## How it Works

T-MAC replaces the standard GEMV (matrix-vector multiply) kernel with a **warp-per-row** approach using **precomputed partial-sum lookup tables**.

Instead of dequantizing weights to FP16 and multiplying, T-MAC:
1. Precomputes all possible partial sums for each 2-bit group into a LUT (in shared memory)
2. Uses the quantized weight bits as direct LUT indices
3. Accumulates partial sums — no FP multiply needed for the core loop

The kernel covers 6 dispatch sites: unfused GEMV, fused SwiGLU, bias-fused, and MoE expert dispatch — each with split-tensor (multi-GPU) variants.

Technical deep-dive: [TMAC.md](TMAC.md)

## Relationship to Upstream

- **Independent fork** with monthly rebase against [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) master
- T-MAC files (`tmac.cu`, `tmac.cuh`) are **self-contained** — they don't exist upstream
- Dispatch sites in `ggml-cuda.cu` and `mmvq.cu` are marked with `// ── T-MAC dispatch site N/6 ──` for easy conflict resolution during rebase
- **All upstream functionality is preserved** — T-MAC is purely additive
- Non-RDNA3 hardware is completely unaffected

## Documentation

| Document | Contents |
|----------|----------|
| [TMAC.md](TMAC.md) | Technical architecture, kernel design, dispatch flow |
| [docs/tmac/benchmarks.md](docs/tmac/benchmarks.md) | Full benchmark suite with CIs and p-values |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

## License

MIT — same as [upstream llama.cpp](https://github.com/ggml-org/llama.cpp).
