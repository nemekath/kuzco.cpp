# T-MAC: Custom GEMV Kernels for AMD RDNA3

**+13-20% token generation throughput** on popular quantizations (Q4_K_M),
**up to +37% on IQ types**, with AMD RX 7900 XTX
(up to +45% on dual-GPU MoE configurations).

T-MAC is a set of custom GEMV (General Matrix-Vector Multiply) kernels optimized for
quantized LLM inference on AMD RDNA3 GPUs. It integrates into llama.cpp as an independent
fork with zero-overhead runtime detection.

## Supported Hardware

| GPU | Architecture | Status |
|-----|-------------|--------|
| AMD RX 7900 XTX | RDNA3 (gfx1100) | Validated |
| AMD RX 7900 XT | RDNA3 (gfx1100) | Expected to work |
| AMD W7900 / W7800 | RDNA3 (gfx1100) | Expected to work |
| AMD RX 9070 series | RDNA4 (gfx12) | Not yet validated |
| NVIDIA GPUs | — | Not supported (stock llama.cpp used automatically) |

T-MAC auto-detects RDNA3 at runtime. Non-RDNA3 GPUs use the stock llama.cpp code path
with zero overhead.

## Performance

All measurements: N=10 paired interleaved A-B, 95% confidence intervals, single GPU
(RX 7900 XTX), `tg128` (128-token generation).

### Q4_K_M (most popular quantization)

| Model | Params | Stock | T-MAC | Speedup |
|-------|--------|------:|------:|--------:|
| Llama 3.2 1B | 1.24B | 373 t/s | 449 t/s | **+20.3%** |
| Codestral 22B | 22.25B | 40.0 t/s | 45.7 t/s | **+14.1%** |
| OLMoE-1B-7B (MoE) | 6.92B | 325 t/s | 373 t/s | **+14.8%** |
| GLM-4.7-Flash (MLA+MoE) | ~16B | 87.4 t/s | 100.7 t/s | **+15.2%** |
| QwQ-32B | 32B | 29.9 t/s | 33.9 t/s | **+13.5%** |

### IQ Quantization Types (fork exclusive)

| Model | Quant | Stock | T-MAC | Speedup |
|-------|-------|------:|------:|--------:|
| Llama 3.2 1B | IQ3_XXS | 361 t/s | 495 t/s | **+36.9%** |
| Llama 3.2 1B | IQ3_S | 367 t/s | 494 t/s | **+34.4%** |
| OLMoE-1B-7B (MoE) | IQ3_S | 313 t/s | 404 t/s | **+29.1%** |
| Llama 3.3 70B | IQ2_XXS | 13.9 t/s | 17.5 t/s | **+25.8%** |
| Llama 3.2 1B | IQ2_XXS | 372 t/s | 463 t/s | **+24.4%** |
| Llama 3.2 1B | IQ2_XS | 369 t/s | 432 t/s | **+17.0%** |
| Llama 3.2 1B | IQ1_M | 452 t/s | 505 t/s | **+11.9%** |
| Llama 3.2 1B | IQ4_XS | 424 t/s | 471 t/s | **+11.1%** |
| Mixtral 8x7B (dual GPU) | IQ3_S | 55.5 t/s | 80.7 t/s | **+45.4%** |

All results are bit-identical to stock (perplexity delta = 0.000 on wikitext-2).
Mixtral measured on dual 7900 XTX (layer split); all others on single GPU.

### Supported Quantization Types (17 types)

| Type | Speedup Range | Notes |
|------|--------------|-------|
| Q4_K_M | +13-20% | Most popular, best overall value |
| Q5_K | +17% | 5-bit K-quant, sub-block parallel |
| Q3_K | +9% | Completes K-quant family |
| Q4_0 | +4-15% | Legacy format |
| Q5_0 | +15% | 5-bit legacy, PPL validated |
| Q6_K | +5-6% | Higher quality, larger blocks |
| IQ3_XXS | +29-37% | Highest single-GPU Dense gain (1B: 361→495 t/s) |
| IQ3_S | +29-34% | +45.4% on dual GPU (Mixtral 8x7B) |
| IQ2_XXS | +24-26% | Extreme compression, 70B on single GPU |
| IQ2_XS | +17% | 2-bit with extra scales |
| IQ2_S | — | Supported, not yet benchmarked |
| IQ4_XS | +11% | N=10 validated |
| IQ4_NL | — | Non-linear 4-bit, supported |
| IQ1_M | +12% | 1.75 bpw, scaleless super-block design |
| MXFP4 | +9% | OCP Microscaling 4-bit (MoE experts) |
| Q8_0, Q5_1 | +2-3% | Minor types (Nemotron shared experts) |

## Build

```bash
git clone https://github.com/nemekath/kuzco.cpp llama-tmac
cd llama-tmac
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build -j$(nproc)
```

`GGML_HIP_TMAC=ON` is the default in this fork. No extra flags needed.

**Requirements:** ROCm (tested with 7.1; 6.x expected to work), AMD GPU with RDNA3 architecture.

## Usage

T-MAC activates automatically for batch=1 GEMV on RDNA3. No configuration needed:

```bash
# Run with T-MAC (automatic)
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
  -m your-model.gguf -ngl 99 -p "Hello" -n 100

# Disable T-MAC for comparison
HIP_VISIBLE_DEVICES=0 GGML_HIP_NO_TMAC=1 ./build/bin/llama-cli \
  -m your-model.gguf -ngl 99 -p "Hello" -n 100
```

**Note:** `HIP_VISIBLE_DEVICES=0` excludes the iGPU on systems with integrated graphics
(which can cause segfaults on gfx1036).

## Verify T-MAC is Active

Run `llama-bench` and compare with/without `GGML_HIP_NO_TMAC=1`:

```bash
# With T-MAC
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-bench \
  -m your-model.gguf -p 0 -n 128 -ngl 99

# Without T-MAC (stock baseline)
HIP_VISIBLE_DEVICES=0 GGML_HIP_NO_TMAC=1 ./build/bin/llama-bench \
  -m your-model.gguf -p 0 -n 128 -ngl 99
```

T-MAC is active if you see higher `tg128` throughput without the env var.

**Note on llama-cli:** Recent llama.cpp versions default `llama-cli` to conversation mode.
For pure decode throughput measurement, use `llama-bench` or `llama-completion`. If using
`llama-cli`, pass `--no-conversation` (which redirects to `llama-completion` internally).

## Regression Testing

```bash
# Full test (N=5 paired, ~20 min)
scripts/tmac-regression.sh

# Quick smoke test (N=2, ~5 min)
scripts/tmac-regression.sh --quick

# Single model
scripts/tmac-regression.sh --models "models/1B-Q4_K_M.gguf" --n 10
```

Exit code 1 on regression (CI-friendly). Paired t-test with configurable thresholds.

## How It Works

T-MAC replaces the stock `mul_mat_vec_q` GEMV kernels with custom implementations
optimized for RDNA3's memory hierarchy:

- **Warp-per-Row:** 32 threads process one output row, 8 rows per block
- **Direct dequant:** F32 FMA instead of dp4a integer pipeline (wins at batch=1 where
  the kernel is purely memory-bound)
- **Fused SwiGLU:** Gate and up projections fused — loads activations once for two dot
  products, halving activation bandwidth for FFN layers
- **Q8_1 skip:** Raw F32 activations instead of stock's F32 -> Q8_1 -> dp4a pipeline
- **MoE dispatch:** Expert parallelism via blockIdx.y with correct per-expert activation
  strides

The kernels intercept at 6 dispatch sites in `ggml-cuda.cu` and `mmvq.cu`. Prefill
(batch > 1) falls through to stock automatically.

## Architecture

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/tmac.cu` | Kernel implementations (template-parametrized) |
| `ggml/src/ggml-cuda/tmac.cuh` | Guards, type dispatch, host declarations |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5 dispatch sites (unfused, split, MoE, fused-split, bias-fused) |
| `ggml/src/ggml-cuda/mmvq.cu` | 1 dispatch site (fused SwiGLU non-split + MoE) |
| `scripts/tmac-regression.sh` | Paired A-B regression test |

## Where T-MAC Fits

T-MAC targets a specific bottleneck: single-token GEMV during text generation.
It is complementary to:

- **Flash Attention (ROCm):** Accelerates prefill (prompt processing). T-MAC
  does not touch attention kernels.
- **composable_kernel / hipBLASLt:** Handle batched GEMM (batch>1). T-MAC
  only fires at batch=1.
- **ExLlamaV2 / vLLM:** Full inference frameworks. kuzco.cpp is a llama.cpp
  fork, not a framework — it extends llama.cpp, not replaces it.

## Fork Maintenance

This is a [GitHub fork](https://github.com/nemekath/kuzco.cpp) of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), rebased
periodically against upstream. **Current base:** `47eb12b95` (2026-02-24).
Dispatch sites are marked with `// ── T-MAC dispatch site N/6 ──` for
easy conflict resolution during rebase.

## License

Same as llama.cpp (MIT).
