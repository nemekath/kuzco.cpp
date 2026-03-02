# kuzco.cpp — the fast speaking Llama!

> llama.cpp fork with T-MAC kernels for AMD RDNA3 — **+10-36% token generation throughput.**

<p align="center">
  <img src="docs/kuzco-logo.png" alt="kuzco.cpp logo" width="400">
</p>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/nemekath/kuzco.cpp)](https://github.com/nemekath/kuzco.cpp/releases)

## What is this?

[llama.cpp](https://github.com/ggml-org/llama.cpp) lets you run large language
models (LLMs) locally on your own hardware. kuzco.cpp is a fork that makes it
**faster on AMD GPUs** — specifically the RX 7900 series (RDNA3).

It replaces the inner math kernel that runs during text generation with a custom
version optimized for AMD's GPU architecture. Everything else stays the same:
same models, same output quality, same commands.

- **+10-36% faster token generation** depending on model and quantization
- **Zero configuration** — auto-detects your GPU and activates automatically
- **Bit-identical output** — same quality as stock llama.cpp (perplexity delta = 0.000)
- **Safe fallback** — non-AMD hardware uses the stock kernel, nothing changes
- **26 models tested** across 13 architecture families, 17 quantization types

## Why a fork?

kuzco.cpp is developed with AI assistance (Claude, Gemini). llama.cpp's contribution policy does not accept AI-generated code. We respect that boundary — hence an independent fork with monthly rebase against upstream.

This is a specialization for RDNA3 token generation, not a replacement for llama.cpp.

## Acknowledgments

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) by **Georgi Gerganov** and [contributors](https://github.com/ggml-org/llama.cpp/graphs/contributors). The [ggml](https://github.com/ggml-org/ggml) tensor library makes all of this possible. MIT license, same as upstream.

## Performance

> **Reading these tables:** "t/s" = tokens per second (higher = faster). Speedup is
> relative to stock llama.cpp on the same hardware. All benchmarks run on a single
> AMD RX 7900 XTX with N=10 paired interleaved runs and 95% confidence intervals.

### Q4_K_M — the most popular quantization

Q4_K_M (~4.8 bits per weight) is the default choice for most users: good quality,
good speed, reasonable VRAM usage.

| Model | Size | What it is | Stock | T-MAC | Speedup |
|-------|-----:|------------|------:|------:|--------:|
| Llama 3.2 1B | 1.24B | Small, fast model | 373 t/s | 449 t/s | **+20.3%** |
| Codestral 22B | 22.25B | Code generation | 40.0 t/s | 45.7 t/s | **+14.1%** |
| OLMoE-1B-7B | 6.92B | Mixture-of-Experts | 325 t/s | 373 t/s | **+14.8%** |
| GLM-4.7-Flash | ~16B | Mixture-of-Experts | 87.4 t/s | 100.7 t/s | **+15.2%** |
| QwQ-32B | 32B | Reasoning model | 29.9 t/s | 33.9 t/s | **+13.5%** |

### IQ types — fitting big models into less VRAM

IQ ("importance quantization") compresses models more aggressively, using fewer
**bits per weight (bpw)**. Lower bpw = smaller file = less VRAM needed, but lower
quality. This lets you run models that wouldn't otherwise fit on your GPU.

For reference: Q4_K_M uses ~4.8 bpw. IQ2_XXS uses ~2.1 bpw — less than half the
VRAM, at the cost of lower output quality.

| Model | Quant | bpw | VRAM savings vs Q4_K | Speedup |
|-------|-------|----:|---------------------:|--------:|
| Llama 1B | IQ3_XXS | 3.06 | ~36% less | **+36.6%** |
| Llama 1B | IQ3_S | 3.44 | ~28% less | **+34.4%** |
| OLMoE-1B-7B | IQ3_S | 3.44 | ~28% less | **+29.1%** |
| Llama 70B | IQ2_XXS | 2.06 | ~57% less | **+25.8%** |
| Llama 1B | IQ2_XXS | 2.06 | ~57% less | **+24.6%** |
| Llama 1B | IQ2_XS | 2.31 | ~52% less | **+17.4%** |
| Llama 1B | IQ1_M | 1.75 | ~64% less | **+11.9%** |

**Why are IQ speedups higher?** Stock llama.cpp uses a generic lookup-table approach
for IQ types. T-MAC replaces this with an optimized implementation — the more
complex the dequantization, the more T-MAC can improve it.

### Multi-GPU (dual 7900 XTX)

Two GPUs allow running models that don't fit on a single card (e.g. Llama 70B at
~38 GB in Q4_0). T-MAC accelerates each GPU's work independently.

| Model | Quant | Speedup |
|-------|-------|--------:|
| Mixtral 8x7B | IQ3_S | **+45.4%** |
| Llama 70B | IQ2_XXS | **+18.9%** |
| Llama 4 Scout | IQ2_XXS-UD | **+12.0%** |
| Llama 70B | Q4_0 | **+6.5%** |

### Tested model architectures

T-MAC works with all major LLM architectures — not just standard transformer models:

| Architecture | Example Models | Status |
|-------------|----------------|--------|
| Dense transformer | Llama, Codestral, QwQ | Validated |
| Mixture-of-Experts (MoE) | OLMoE, Mixtral, Nemotron | Validated |
| State-Space (SSM) | Mamba, Falcon H1, Jamba | Validated |
| Linear attention | RWKV-6 | Validated |
| Vision-Language (VLM) | Qwen2-VL | Validated |

26 models across 13 architecture families. Full benchmark data with confidence
intervals and p-values: [docs/tmac/benchmarks.md](docs/tmac/benchmarks.md)

## Quick Start

### Prerequisites

- AMD RDNA3 GPU (RX 7900 series validated)
- [ROCm](https://rocm.docs.amd.com/) (tested with 7.1; 6.x expected to work but untested)
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

17 types supported. T-MAC activates automatically when conditions are met (RDNA3 +
batch=1 + supported type + alignment). No configuration needed — just use any
supported quantization and T-MAC takes care of the rest.

> **Which quant should I use?** Start with **Q4_K_M** — it's the best balance of
> quality, speed, and VRAM usage for most models. Only go lower (IQ3, IQ2) if your
> model doesn't fit in VRAM at Q4_K_M.

| Category | Types | Bits per weight | Use case |
|----------|-------|----------------:|----------|
| K-quants | Q3_K, Q4_K, Q5_K, Q6_K | 3.4 – 6.6 | **Recommended.** Best quality/size trade-off |
| Legacy | Q4_0, Q5_0, Q5_1, Q8_0 | 4.0 – 8.5 | Older format, still works well |
| IQ (importance) | IQ1_M – IQ4_XS (8 types) | 1.75 – 4.25 | Extreme compression for large models |
| MXFP | MXFP4 | 4.0 | OCP Microscaling format (some MoE models) |

## How it Works

During token generation, the GPU spends most of its time on matrix-vector
multiplications (one token at a time). Stock llama.cpp dequantizes compressed
weights back to floating point, then multiplies. T-MAC takes a shortcut:

1. **Precomputes a lookup table** of all possible partial results in fast GPU shared memory
2. **Uses the compressed weight bits directly as table indices** — no decompression needed
3. **Accumulates the looked-up values** — skipping the expensive multiply step entirely

Additionally, T-MAC **fuses operations** that stock llama.cpp runs separately (e.g.
the gate + up projections in SwiGLU layers), cutting memory reads in half for those
layers.

The result: fewer memory reads, fewer instructions, same output. Technical
deep-dive with architecture diagrams: [TMAC.md](TMAC.md)

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
