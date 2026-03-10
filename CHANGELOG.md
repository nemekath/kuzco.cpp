# Changelog

> Versions v1.0–v1.6 were development milestones prior to the public fork migration.

## v2.0.0 — Proper GitHub Fork + Qwen3.5 + ROCm 7.2

**Release date:** 2026-03-10

### Highlights

- **Proper GitHub fork** of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with full upstream history
- **Qwen3.5 family** benchmarks: 9B +11.1%, 27B +9.5%, 35B-A3B (MoE) +11.7%, 27B Q8_0 dual-GPU +6.3%
- **ROCm 7.2.0 validated:** 14 models, 8 quant types, zero regressions
- 30+ models across 13 architecture families, all PPL Δ=0.000

### Added
- Qwen3.5 family benchmarks (9B, 27B, 35B-A3B, 122B)
- ROCm 7.2.0 validation infrastructure (`ci/rocm-validation/`)
- 95% confidence intervals for all model zoo entries
- Batch=1 scope caveat on all tg128 benchmark tables
- Scout Q4_K_M rerun at N=10 (Parity, CPU offload CV=14-18%)
- Model zoo utility scripts
- Pre-commit hook to prevent internal workflow leaks

### Changed
- Migrated from standalone repo to proper GitHub fork (full upstream git history)
- Default branch: `tmac` (upstream `master` untouched for rebasing)
- Scout Q4_K_M reframed from +2.6% to "Parity" (CPU offload variance too high)
- 30+ models across 13 architecture families (was 26)
- Removed all upstream files previously deleted (kept intact for cleaner rebases)

## v1.6 — Upstream Rebase + Cleanup

**Release date:** 2026-03-01

### Highlights

- **Upstream rebase** to llama.cpp `47eb12b95`
- Repository renamed to **kuzco.cpp** (`nemekath/kuzco.cpp`)
- P16 (sign LUT elimination) and P17 (HW counter validation) closed
- 30+ models across 13 architecture families validated, all PPL delta = 0.000
- 17 quantization types supported

## v1.5 — IQ1_M Kernel + Architecture Expansion

**Release date:** 2026-02-28

### Highlights

- **IQ1_M kernel** — 17th supported quantization type (1.75 bpw, scaleless super-block design)
- **4 new architecture families:** Falcon H1 (Mamba2+Attn) **+16.8%**, Mamba 2.8B (pure SSM) **+10.3%**, RWKV-6 (linear attn) **+14.5%**, Qwen2-VL (VLM) **+17.4%**
- **Sign LUT elimination** (P16 Phase 1): IQ3_XXS **+36.6%**, IQ2_XXS **+24.6%**, IQ2_XS **+17.4%**
- Qwen3.5-122B-A10B IQ2_XXS: **+7.4%** (first model exercising IQ1_M kernel, 100% Active Ratio)
- Total: 26 models, 13 architecture families, all PPL delta = 0.000

### IQ Performance (P16 sign LUT elimination)

| Model | Quant | Speedup |
|-------|-------|--------:|
| Llama 3.2 1B | IQ3_XXS | **+36.6%** |
| Llama 3.2 1B | IQ2_XXS | **+24.6%** |
| Llama 3.2 1B | IQ2_XS | **+17.4%** |
| Llama 3.2 1B | IQ1_M | **+11.9%** |

## v1.4 — Llama 4 Aliasing Fix + Hardening

**Release date:** 2026-02-27

### Highlights

- **Aliasing fix:** src1/dst pointer aliasing on fused SwiGLU (pool-allocated temp copy)
- **Llama 4 Scout IQ2_XXS-UD:** +12.0% on dual 7900 XTX (N=10, p<0.0001)
- `thread_local` dedup array, ne2==1 spec-decode guard documented
- 21 models validated

## v1.3 — IQ Types + Kahan Summation

**Release date:** 2026-02-26

### Highlights

- **3 new IQ types:** IQ3_S, IQ2_XXS, IQ4_XS (shared-memory LUT dequantization)
- **Kahan summation** for IQ2_XXS accumulator precision
- **Fused SwiGLU opt-out** for IQ types (LDS budget conflict)
- 16 quantization types supported

## v1.2 — Q3_K Kernel + Stability Hardening

**Release date:** 2026-02-25

### Highlights

- **Q3_K kernel** — 12th supported quantization type, completes K-quant family
- **+8.8%** throughput on 1B Q3_K_L (406.18 vs 373.39 t/s, N=5)
- **PPL bit-identical** to stock (15.3342, delta=0.000, wikitext-2)
- **Long-context validated** — 8B at 4096 tokens, coherent output, no KV-cache issues
- **FLOP-weighted Compute Coverage** metric

### Q3_K Performance

| Model | Quant | Stock | T-MAC | Speedup |
|-------|-------|------:|------:|--------:|
| Llama 3.2 1B | Q3_K_L | 373 t/s | 406 t/s | **+8.8%** |

### New Features

- **Q3_K GEMV kernel:** 110 bytes/block, 16 sub-blocks of 16 (same geometry as Q6_K).
  3-bit values (2-bit qs + 1-bit hmask), signed 6-bit scales. Dual-accumulator dot for
  hmask correction. K-quant family complete: Q3_K + Q4_K + Q5_K + Q6_K. 170 LOC.
- **FLOP-weighted Compute Coverage:** Reported alongside op-count Active Ratio at exit.
  Addresses case where 87.4% op coverage masked 16% FLOP coverage.
- **Regression hardening:** 8B Q5_K_M model added. `--multigpu` flag for dual-GPU smoke test.
- **Long-context stress test:** 8B Q5_K_M at 4096 tokens with ctx=8192 — coherent output,
  111.2 t/s, no repetition collapse. KV-cache concern eliminated.

## v1.1 — MXFP4 Support for OCP Microscaling MoE Models

**Release date:** 2026-02-24

### Highlights

- **MXFP4 kernel** (OCP Microscaling 4-bit) — 11th supported quantization type
- **GPT-OSS 20B: +8.9%** throughput (187.85 vs 172.42 t/s, N=10 interleaved, p=4.5e-15)
- **Active Ratio 87.4% -> 97.7%** on GPT-OSS model family (MXFP4 expert tensors now dispatched)
- **FLOP-weighted Compute Coverage** metric — reveals actual compute coverage, not just op count
- PPL bit-identical to stock (delta = 0.000, wikitext-2, 560 chunks)
- Zero regressions across 6 existing models (12 tests)

### MXFP4 Performance

| Model | Type | Stock | T-MAC | Speedup |
|-------|------|------:|------:|--------:|
| GPT-OSS 20B (MoE 128/4) | Q4_K_M | 172.4 t/s | 187.9 t/s | **+8.9%** |
| GPT-OSS 120B (CPU offload) | Q4_K_M | 37.3 t/s | 38.4 t/s | +2.9% |

### New Features

- **MXFP4 GEMV kernel:** 17-byte blocks with E8M0 shared exponent, LUT dequant via
  `kvalues_mxfp4[16]` in constant cache (no shared memory). Same template architecture
  as Q4_0 — 159 LOC across tmac.cu/tmac.cuh.
- **Compute Coverage metric:** FLOP-weighted Active Ratio reported at exit alongside
  op-count ratio. Addresses "phantom metric" problem where 87.4% op coverage masked
  only 16% FLOP coverage on MoE models.

---

## v1.0 — T-MAC: Custom GEMV Kernels for RDNA3

**Release date:** 2026-02-24

### Highlights

T-MAC v1.0 delivers **+10-34% token generation throughput** for quantized LLM inference
on AMD RDNA3 GPUs (RX 7900 XTX / XT / W7900). All 12 supported quantization types are
bit-identical to stock llama.cpp (perplexity delta = 0.000 on wikitext-2).

- **12 quantization types** supported: Q4_K, Q5_K, Q6_K, Q4_0, Q5_0, Q5_1, Q8_0, IQ3_S, IQ2_XXS, IQ4_XS
- **4 architecture families** validated: Dense, MoE, MLA+MoE, SSM-Hybrid
- **10+ models** benchmarked with N=10 paired interleaved methodology
- **6 dispatch sites** covering unfused, fused SwiGLU, bias-fused, MoE, and split-tensor paths
- **Zero configuration** — auto-detects RDNA3 at runtime, falls through to stock on other GPUs

### Performance (Q4_K_M, tg128, single RX 7900 XTX)

| Model | Params | Stock | T-MAC | Speedup |
|-------|--------|------:|------:|--------:|
| Llama 3.2 1B | 1.24B | 373 t/s | 449 t/s | **+20.3%** |
| Codestral 22B | 22.25B | 40.0 t/s | 45.7 t/s | **+14.1%** |
| OLMoE-1B-7B (MoE) | 6.92B | 325 t/s | 373 t/s | **+14.8%** |
| GLM-4.7-Flash (MLA+MoE) | ~16B | 87.4 t/s | 100.7 t/s | **+15.2%** |
| QwQ-32B | 32B | 29.9 t/s | 33.9 t/s | **+13.5%** |
| Jamba-Reasoning 3B (SSM) | 3.2B | 164 t/s | 184 t/s | **+12.5%** |

### Performance (IQ types — fork exclusive)

| Model | Quant | Stock | T-MAC | Speedup |
|-------|-------|------:|------:|--------:|
| Llama 3.2 1B | IQ3_S | 367 t/s | 494 t/s | **+34.4%** |
| OLMoE-1B-7B (MoE) | IQ3_S | 313 t/s | 404 t/s | **+29.1%** |
| Llama 3.2 1B | IQ2_XXS | 372 t/s | 455 t/s | **+22.3%** |
| Llama 3.3 70B | IQ2_XXS | 13.9 t/s | 17.5 t/s | **+25.8%** |
| Llama 3.2 1B | IQ4_XS | 424 t/s | 471 t/s | **+11.1%** |
| Mixtral 8x7B (dual GPU) | IQ3_S | 55.5 t/s | 80.7 t/s | **+45.4%** |

### Multi-GPU (dual RX 7900 XTX, row-split)

| Model | Quant | Stock | T-MAC | Speedup |
|-------|-------|------:|------:|--------:|
| Llama 70B | Q4_0 | 20.8 t/s | 22.2 t/s | **+6.5%** |
| Llama 70B | IQ2_XXS | 19.3 t/s | 22.9 t/s | **+18.9%** |
| Codestral 22B | Q4_0 | 39.8 t/s | 45.5 t/s | **+14.2%** |

### Validated Models

All measurements use N=10 paired interleaved benchmarking with 95% confidence intervals:

- Llama 3.2 1B (Q4_K_M, Q4_0, Q5_K_M, Q5_0, IQ3_S, IQ2_XXS, IQ4_XS)
- Codestral 22B (Q4_K_M, Q4_0, Q6_K_L)
- OLMoE-1B-7B MoE (Q4_K_M, Q4_0, IQ3_S)
- GLM-4.7-Flash MLA+MoE (Q4_K_M)
- QwQ-32B (Q4_K_M)
- Ministral 14B (Q4_K_M)
- Jamba-Reasoning 3B SSM-hybrid (Q4_K_M)
- Llama 3.3 70B (Q4_0, IQ2_XXS)
- Llama 3.1 8B (Q5_K_M)
- Nemotron 30B MoE (Q4_0)
- Mixtral 8x7B MoE (IQ3_S, dual GPU)

### How It Works

T-MAC replaces the stock `mul_mat_vec_q` GEMV kernels with custom implementations:

- **Warp-per-Row:** 32 threads per output row, 8 rows per block
- **Direct F32 dequant:** Skips the F32 -> Q8_1 quantization pass (81% of the gain on 1B)
- **Fused SwiGLU:** Gate + up projections share activation loads (halves activation bandwidth)
- **Sub-block parallelization:** K-quants (Q4_K, Q5_K, Q6_K) distribute sub-blocks across warp lanes
- **IQ LUT dequantization:** Shared-memory lookup tables for IQ3_S, IQ2_XXS, IQ4_XS

### Build

```bash
git clone https://github.com/nemekath/kuzco.cpp llama-tmac
cd llama-tmac
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build -j$(nproc)
```

Requires ROCm (tested with 7.1; 6.x expected to work). `GGML_HIP_TMAC=ON` is the default.

### Known Limitations

- **RDNA3 only:** Validated on gfx1100 (7900 XTX). Expected to work on gfx1101/1102/1150/1151.
  Not supported on NVIDIA or RDNA2.
- **Batch=1 only:** T-MAC targets single-token generation (tg). Prefill (batch > 1) falls
  through to stock automatically.
- **Alignment constraints:** Q4_K/Q5_K/Q6_K require ne0 % 256 == 0. Smaller types require
  ne0 % 32 == 0. Models with non-aligned dimensions (e.g., ne0=2880) partially bypass T-MAC.
- **Fork:** Not merged upstream. GitHub fork of llama.cpp, rebased periodically.
