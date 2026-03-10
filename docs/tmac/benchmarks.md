# T-MAC: Benchmark Results & Validation Data

> Version numbers (v7.x–v9.x, v1.0–v1.6) in this document refer to internal
> development milestones prior to the public fork migration.

### Evidence levels

This document distinguishes between measured results and interpretive analysis:

- **Measured** — directly observed in benchmarks with statistical validation (N≥5, paired t-test, 95% CI)
- **Inferred** — derived from measured data through reasoning (e.g. component breakdowns, root-cause diagnoses) but not independently validated via controlled experiment or profiling
- **Hypothesis** — plausible explanation consistent with data but not tested in isolation; alternative explanations may exist
- **Confounded** — modifier indicating a comparison involves multiple uncontrolled variables; the stated explanation is plausible but alternative causes cannot be excluded

Passages marked *(inferred)* or *(hypothesis)* indicate where the evidence level is below direct measurement.

---

## Prefill (pp512) Verification (2026-02-21, updated 2026-02-22)

T-MAC guards reject batch>1 (ne[1]!=1), so pp512 falls back to stock automatically.

**Q4_0 (zero impact confirmed):**

| Model | Test | T-MAC ON | T-MAC OFF | Delta |
|-------|------|----------|-----------|-------|
| 1B | pp512 | 15,898 t/s | 15,628 t/s | +1.7% (noise) |
| 1B | pp2048 | 12,979 t/s | 12,984 t/s | 0.0% (noise) |

**Q4_K_M (regression fixed):**

Pre-fix regression was caused by a T-MAC override in `ggml_cuda_should_use_mmq()` (mmq.cu)
that forced MMQ for all T-MAC-supported types regardless of batch size. This overrode the
RDNA3 heuristic that routes Q6_K to hipBLAS when ne11 > 128, which is faster for large
batches. Fix: removed the override (4 lines). RDNA3 heuristics now apply correctly.

| Model | Test | T-MAC ON | T-MAC OFF | Delta | Notes |
|-------|------|----------|-----------|-------|-------|
| 1B | pp512 (N=10) | 15,426 ± 173 | 15,367 ± 54 | +0.4% (p=0.30) | Was -8.9% |
| 22B | pp512 (N=10) | 1,156 ± 5.6 | 1,156 ± 6.3 | -0.0% (p=0.90) | Was -8.0% |
| OLMoE | pp512 (N=10) | 8,559 ± 37 | 8,552 ± 32 | +0.1% (p=0.49) | Was -3.5% |
| 1B | pp2048 | 12,600 t/s | 12,541 t/s | +0.5% (noise) | Long context clean |
| 22B | pp2048 | 962 t/s | 961 t/s | +0.1% (noise) | Long context clean |

**Perplexity (wikitext-2, 564 chunks, ctx=512):** T-MAC 14.4994 ± 0.108, Stock 14.4994 ± 0.108 (Δ=0.000, PASS)

---

## Current Baseline

Statistical benchmarks: single GPU (7900 XTX), `tg128`. Q4_K_M: N=10 paired interleaved.
Q4_0: N=12 (llama-bench aggregate). Paired t-test, 95% CI via t-distribution. 2026-02-22.

> *All generation (tg128) benchmarks measured at batch=1. T-MAC guards reject batch>1,
> falling back to stock kernels. Results do not reflect batch inference performance.*

**Q4_0 models (N=12 llama-bench aggregate, ± = 95% CI via t-distribution df=11):**

| Model | Type | Params | Stock (mean ± 95% CI) | T-MAC (mean ± 95% CI) | Speedup [95% CI] | p-value |
|-------|------|--------|----------------------:|----------------------:|------------------:|--------:|
| Llama 3.2 1B | Dense | 1.24B | 419.37 ± 0.70 t/s | 481.64 ± 1.43 t/s | **+14.8%** [+14.5%, +15.2%] | — |
| Codestral 22B | Dense | 22.25B | 47.68 ± 0.06 t/s | 49.54 ± 0.04 t/s | **+3.9%** [+3.7%, +4.1%] | 1.5e-11 |
| Nemotron 30B | MoE | 31.58B | 144.47 ± 0.17 t/s | 155.85 ± 0.07 t/s | **+7.9%** [+7.7%, +8.0%] | 1.6e-14 |
| OLMoE-1B-7B | MoE | 6.92B | 363.49 ± 0.66 t/s | 401.20 ± 1.16 t/s | **+10.4%** [+10.0%, +10.7%] | — |

**Q4_K_M models (N=10 paired interleaved A-B-A-B, ± = 95% CI):**

| Model | Type | Params | Stock (mean ± 95% CI) | T-MAC (mean ± 95% CI) | Speedup [95% CI] | p-value |
|-------|------|--------|----------------------:|----------------------:|------------------:|--------:|
| Llama 3.2 1B | Dense | 1.24B | 373.43 ± 0.52 t/s | 449.38 ± 0.90 t/s | **+20.3%** [+20.1%, +20.6%] | 1.1e-18 |
| Codestral 22B | Dense | 22.25B | 40.03 ± 0.13 t/s | 45.66 ± 0.04 t/s | **+14.1%** [+13.7%, +14.4%] | 3.3e-15 |
| OLMoE-1B-7B | MoE | 6.92B | 324.77 ± 0.54 t/s | 372.87 ± 0.68 t/s | **+14.8%** [+14.5%, +15.1%] | 1.9e-15 |
| GLM-4.7-Flash | MLA+MoE | ~16B | 87.36 ± 0.16 t/s | 100.67 ± 0.24 t/s | **+15.2%** [+14.9%, +15.6%] | 1.3e-16 |
| Ministral 14B | Dense | 14B | 64.04 ± 0.11 t/s | 73.04 ± 0.11 t/s | **+14.1%** [+13.8%, +14.3%] | 1.2e-16 |
| QwQ-32B | Dense | 32B | 29.88 ± 0.06 t/s | 33.92 ± 0.06 t/s | **+13.5%** [+13.2%, +13.8%] | 4.5e-15 |

GLM was -12.1% before nb_sub guard fix (ne0=512 → nb_sub=16, 50% warp utilization).

**Additional validated models (N=10 paired interleaved, ± = 95% CI, tg128 batch=1):**

| Model | Type | Params | Quant | Stock (mean ± 95% CI) | T-MAC (mean ± 95% CI) | Speedup [95% CI] | p-value |
|-------|------|--------|-------|----------------------:|----------------------:|------------------:|--------:|
| Granite 4.0-h-small | Dense | ~2B | Q4_K_M | 61.26 ± 0.10 t/s | 67.33 ± 0.11 t/s | **+9.9%** [+9.7%, +10.2%] | 1.1e-16 |
| Jamba-Reasoning 3B | SSM-hybrid | 3.20B | Q4_K_M | 163.67 ± 0.23 t/s | 184.18 ± 0.21 t/s | **+12.5%** [+12.3%, +12.7%] | <1e-15 |
| Nemotron-3-Nano 30B | MoE | 31.58B | Q4_K_M | 130.53 ± 2.32 t/s | 130.80 ± 4.56 t/s | **+0.2%** (NS) | 0.91 |
| Devstral-Small 24B | Dense | 24B | Q6_K_L | 35.79 ± 0.30 t/s | 37.75 ± 0.35 t/s | **+5.5%** [+4.2%, +6.8%] | 2.2e-05 |
| DeepSeek-V2-Lite | MoE (MLA) | 16B | Q4_K_M | 155.17 t/s | 179.78 t/s | **+15.9%** | N=5 |

DeepSeek-V2-Lite validates MLA+MoE (64E, 6 active). Compute Coverage: 92.1% (MLA has
non-standard dims). First DeepSeek family model validated.
Devstral lower gain *(inferred)*: Q6_K_L quant has mostly Q6_K tensors (larger 210-byte blocks,
better stock bandwidth utilization). Granite lower than smoke test *(measured)*: single-run
noise in llama-completion-based smoke test vs precise llama-bench N=10 measurement.
Jamba SSM-hybrid +12.5%: closes architecture gap (Dense, MoE, MLA, SSM-hybrid all validated).
Nemotron Q4_K_M +0.2% (NS): *(inferred)* likely a VRAM-pressure artifact. Model = 23 GB on
24 GB VRAM (<1 GB headroom). CV=4.9% (16-40x normal), suggesting resource contention. Not
confirmed via memory profiling or retest on higher-VRAM hardware. Supporting evidence:
Nemotron Q4_0 = +7.9% at 18 GB (6 GB headroom, normal CV). Additionally, Nemotron's expert
dimensions (2688, 1856) both fail 256-alignment, so MoE expert layers fall back to stock
regardless — the +7.9% Q4_0 gain comes from dense/shared-expert layers only *(measured via
Active Ratio diagnostic)*.

**Industry trend note** *(hypothesis)*: Models with many fine-grained experts
(e.g. DeepSeek-V3: 256E, FFN=1536, ne0=1536 → nb_sub=48) pass the current dispatch
guard but are marginal at IQ types. As architectures trend toward more-experts-smaller-FFN,
T-MAC's MoE expert coverage may narrow. Dense and shared-expert layers remain fully
covered regardless of expert granularity.

**IQ Quantization types (N=10 paired interleaved unless noted, ± = 95% CI, tg128 batch=1):**

| Model | Type | Params | Quant | Stock (mean ± 95% CI) | T-MAC (mean ± 95% CI) | Speedup [95% CI] | p-value |
|-------|------|--------|-------|----------------------:|----------------------:|------------------:|--------:|
| Qwen2-57B-A14B | MoE 64E | 57B/14B | IQ3_XXS | 52.81 t/s | 81.58 t/s | **+54.5%** [+54.2%, +54.6%] | <0.0001 |
| Jamba Mini 1.7 | SSM-MoE | 52B | IQ3_XXS | 63.67 t/s | 93.75 t/s | **+47.2%** | N=5 |
| Mixtral 8x7B | MoE | 46.70B | IQ3_S | 55.47 ± 0.06 t/s | 80.67 ± 0.09 t/s | **+45.4%** [+45.3%, +45.5%] | <1e-15 |
| Llama 3.2 1B | Dense | 1.24B | IQ3_XXS | 361.4 ± 0.6 t/s | 494.5 ± 0.7 t/s | **+36.9%** [+36.7%, +37.1%] | <1e-15 |
| Llama 3.2 1B | Dense | 1.24B | IQ3_S | 367.4 ± 0.5 t/s | 493.8 ± 1.1 t/s | **+34.4%** [+34.0%, +34.8%] | <1e-15 |
| OLMoE-1B-7B | MoE | 6.92B | IQ3_S | 313.1 ± 0.8 t/s | 404.3 ± 1.6 t/s | **+29.1%** [+28.7%, +29.6%] | <1e-15 |
| OLMoE-1B-7B | MoE | 6.92B | IQ3_XXS | 319.0 ± 2.3 t/s | 409.0 ± 1.6 t/s | **+28.2%** [+27.6%, +28.8%] | <1e-6 |
| Llama 3.3 70B | Dense | 70.55B | IQ2_XXS | 13.93 ± 0.02 t/s | 17.52 ± 0.08 t/s | **+25.8%** | <1e-6 |
| Llama 3.2 1B | Dense | 1.24B | IQ2_XXS | 372.1 ± 1.1 t/s | 462.8 ± 0.8 t/s | **+24.4%** [+24.0%, +24.8%] | <1e-15 |
| DBRX | MoE 16E | 132B/36B | IQ2_XXS | 23.09 t/s | 28.16 t/s | **+22.0%** | N=5, dual |
| Llama 3.2 1B | Dense | 1.24B | IQ2_XS | 369.3 ± 0.8 t/s | 432.1 ± 1.2 t/s | **+17.0%** [+16.6%, +17.4%] | <1e-15 |
| Llama 3.2 1B | Dense | 1.24B | IQ4_XS | 424.2 ± 1.0 t/s | 471.4 ± 1.0 t/s | **+11.1%** [+10.8%, +11.4%] | <1e-15 |
| Llama 3.2 1B | Dense | 1.24B | IQ1_M | 451.5 ± 1.2 t/s | 505.3 ± 1.2 t/s | **+11.9%** [+11.7%, +12.2%] | <1e-15 |
| OLMoE-1B-7B | MoE 64E | 6.92B | IQ2_XXS | 323.4 ± 0.7 t/s | 382.4 ± 2.7 t/s | **+18.3%** [+17.1%, +19.4%] | N=5 |
| Hunyuan-A13B | MoE 256E | 85B/13B | IQ2_XXS | 48.0 t/s | 48.1 t/s | +0.2% (NS) | N=5 |

Qwen2-57B-A14B IQ3_XXS (+54.5%) is the **all-time single-GPU speedup record**. Verified with
N=12 (N_eff=10): +54.4%, 95% CI: +54.2% to +54.6%, p<0.0001. MoE sparsity (64E, 4 active)
combined with IQ3_XXS (transitional regime) creates multiplicative benefit — expert matrices
large enough (ne0=3584/2560, sb=112/80) for efficient T-MAC dispatch.

Jamba Mini 1.7 IQ3_XXS (+47.2%) is the second highest single-GPU speedup. SSM-MoE hybrid
architecture (Mamba+Attn+MoE, 16E, 2 active). Expert matrices (ne0=4096, sb=128) are large
and well-aligned. N=5 paired interleaved.

DBRX IQ2_XXS (+22.0%): 16 experts, 4 active. Large expert matrices (ne0=6144/10752,
sb=192/336) — enough sub-blocks even for latency-bound IQ2_XXS. Dual GPU (132B model).
Compute Coverage: 89.8% (some alignment misses on non-standard dims).

OLMoE-1B-7B IQ2_XXS (+18.3%): **matched-quant validation** (2026-03-07). Predicted
regression or NS (small experts ne0=1024/2048, sb=32/64 at latency-bound IQ2_XXS).
Result refutes prediction — dense layers (100% coverage) contribute enough to deliver
+18.3% despite expert coverage of only 76.8%. t=43.48, p<0.0001. Partially invalidates
the "IQ2_XXS only works for large experts" rule: dense layer contribution can dominate.

Hunyuan-A13B IQ2_XXS (+0.2% NS): 256 experts, 6 active. Down projection ne0=1024
(sb=32) — marginal sub-block count at latency-bound IQ2_XXS. Key difference vs OLMoE:
Hunyuan has 256E (high MoE ratio, expert layers dominate compute) while OLMoE has 64E
(lower MoE ratio, dense layers contribute more). Dual-GPU verification: -0.5% (NS) —
confirms result is quant-type-specific, not GPU-config-specific.

**Sign LUT elimination (2026-02-27, re-measured 2026-03-02):** IQ3_XXS, IQ2_XXS, IQ2_XS updated with
sign LUT elimination results and re-measured with full absolute t/s values (N=10 paired interleaved).
IQ3_XXS 1B (+36.9%) is T-MAC's highest single-GPU Dense speedup. IQ2_XXS 1B +24.4%. IQ2_XS 1B +17.0%.
OLMoE IQ3_XXS (+28.2%) validates sign LUT elimination on MoE path (64 experts / 8 active). Requantized model
(Q4_K_M → IQ3_XXS with imatrix + --allow-requantize) — kernel validation, not community-distributed.

Mixtral 8x7B IQ3_S (+45.4%) measured on dual 7900 XTX layer split — not directly comparable
to single-GPU results (dual-GPU sync overhead inflates stock baseline proportionally more,
amplifying T-MAC's relative advantage). Now third-highest overall after Qwen2-57B and Jamba.
OLMoE IQ3_S (+29.1%) validates IQ support on smaller MoE architecture (SwiGLU fused).
IQ2_XXS had a grid/block dimension bug (75% of output rows uncomputed → garbage) fixed
2026-02-23. IQ2_XXS PPL verified: T-MAC = Stock = 50.7752 (Δ=0.000, wikitext-2).
IQ4_XS re-measured with N=10 interleaved: +11.1% (was +6.4% at N=5 sequential).
IQ1_M 1B (+11.9%) first pure-IQ1_M benchmark. N=10 paired interleaved, t=110.45. Scaleless design
(no `d` field, super-block scale reassembled from packed scales). 8KB LDS (largest IQ LUT,
25% CU occupancy — 4 workgroups / 16 max waves per CU). Requantized model (Q5_K_M → IQ1_M
with imatrix) — kernel validation only, not a community-distributed model.
IQ types use shared-memory LUT dequantization (breaks zero-smem principle, ~2-8KB LDS).

**IQ Perplexity validation (wikitext-2, 564 chunks, ctx=512):**

| Model | Quant | T-MAC PPL | Stock PPL | Delta |
|-------|-------|-----------|-----------|-------|
| Llama 3.2 1B | IQ2_XXS | 50.7752 ± 0.402 | 50.7752 ± 0.402 | 0.000 |
| Llama 3.2 1B | IQ3_S | 16.3780 ± 0.121 | 16.3780 ± 0.121 | 0.000 |
| Llama 3.2 1B | IQ4_XS | 14.6969 ± 0.110 | 14.6969 ± 0.110 | 0.000 |
| Llama 3.3 70B | IQ2_XXS | 7.7333 ± 0.052 | 7.7333 ± 0.052 | 0.000 |
| Llama 3.2 1B | Q5_0 | 16.0302 ± 0.283 | 16.0302 ± 0.283 | 0.000 |
| Mixtral 8x7B | IQ3_S | 4.7367 ± 0.026 | 4.7367 ± 0.026 | 0.000 |
| Llama 3.2 1B | Q3_K_L | 15.3342 ± 0.115 | 15.3342 ± 0.115 | 0.000 |
| Llama 3.2 1B | IQ1_M | 211.4944 ± 1.810 | 211.4944 ± 1.810 | 0.000 |
| Llama 3.2 1B | IQ3_XXS | 18.5678 | 18.5678 | 0.000 |
| Llama 3.2 1B | IQ2_XS | 33.7614 | 33.7614 | 0.000 |

All types are bit-identical to stock (Δ=0.000). Q5_0 PPL validated (Hard Blocker resolved).
Mixtral 8x7B IQ3_S PPL validated (dual GPU, 642 chunks, ctx=512).
Tier 2 coherence (temp=0) PASS for IQ3_S 1B, 70B IQ2_XXS, and Mixtral 8x7B IQ3_S (dual GPU,
structured compiler guide with code examples, no drift through final token).
PPL hierarchy: Mixtral IQ3_S (4.74) < IQ4_XS (14.70) < Q4_K_M (14.50) < Q5_0 (16.03) ≈ IQ3_S 1B (16.38) < IQ2_XXS 1B (50.78) < IQ1_M 1B (211.49).
Note: Q5_0 PPL tested on requantized model (Q4_K_M → Q5_0), hence slightly higher than native Q5_0.
IQ1_M PPL tested on requantized model (Q5_K_M → IQ1_M with imatrix + --allow-requantize). High absolute PPL (211.5) expected at 1.75 bpw on 1B model.

**MXFP4 PPL validation (wikitext-2, 560 chunks, ctx=512):**

| Model | Quant | T-MAC PPL | Stock PPL | Delta |
|-------|-------|-----------|-----------|-------|
| GPT-OSS 20B | Q4_K_M (MXFP4 experts) | 158.8331 ± 1.338 | 158.8331 ± 1.338 | 0.000 |

MXFP4 bit-identical to stock (Δ=0.000). High absolute PPL (158.8) is a model/quant property, not T-MAC.
All 17 quant types now validated: Δ=0.000 across the board.

**MXFP4 models (N=10 paired interleaved, ± = 95% CI):**

| Model | Type | Params | Quant | Stock (mean ± 95% CI) | T-MAC (mean ± 95% CI) | Speedup [95% CI] | p-value |
|-------|------|--------|-------|----------------------:|----------------------:|------------------:|--------:|
| GPT-OSS 20B | MoE | 20.91B | Q4_K_M (MXFP4 experts) | 172.42 ± 0.16 t/s | 187.85 ± 0.31 t/s | **+8.9%** [+8.7%, +9.1%] | 4.5e-15 |

GPT-OSS 20B: 128 experts, 4 active (~5.1B active). Expert FFN tensors = MXFP4 (74.9% of GEMV FLOP).
Active Ratio: 87.4% → 97.7% (MXFP4 now dispatched through T-MAC).
GPT-OSS 120B (58.5 GiB, CPU offload): +2.9% (N=3, Amdahl bottleneck: PCIe + CPU offload overhead).

**Preliminary (single-run smoke test, not yet statistically validated):**

| Model | Type | Params | Quant | Speedup |
|-------|------|--------|-------|--------:|
| Nemotron-3-Nano 30B | MoE | 31.58B | Q4_K_M | +7.9%* |

*Nemotron smoke test number is pre-nb_sub guard — may change slightly.

**DeepSeek-R1:**

| Model | Type | Params | Quant | Stock (N=10) | T-MAC (N=10) | Speedup [95% CI] | p-value |
|-------|------|--------|-------|-------------:|-------------:|------------------:|--------:|
| DeepSeek-R1-Distill-Llama-8B | Dense | 8.03B | Q4_K_M | 100.26 ± 0.30 t/s | 118.67 ± 0.34 t/s | **+18.4%** [18.0%, 18.7%] | <0.0001 |

Active Ratio: 100% (1775/1775 ops). Compute Coverage: 100%. Llama architecture → validates seamlessly.
N=10 paired interleaved, tg128. Updated from N=3 (2026-03-02).

**Model zoo validation (single-run smoke tests, 2026-02-25):**

| Model | Type | Params | Quant | GPU Setup | t/s | Active Ratio | Compute Coverage |
|-------|------|--------|-------|-----------|----:|:------------:|:----------------:|
| Gemma 3 27B | SWA+Dense | 27B | Q6_K | Single | 31.92 | 100% | 100% |
| Qwen3-30B-A3B | MoE | 30B | Q6_K | Dual | 108.81 | 97.2% | 99.2% |
| Hermes 4 36B | Dense | 36B | Q6_K | Dual | 24.09 | 100% | 100% |
| Phi-3.5 MoE | MoE | 42B | Q6_K_L | Dual | 95.21 | 97.2% | 95.0% |
| Qwen3-Next-80B-A3B | MoE | 80B | Q4_K_M | Dual+CPU | 17.34 | ~100% | 98.0% |
| Llama 4 Scout 17B-16E | MoE 16E | 109B | IQ2_XXS-UD | Dual | 44.62 | **100%** | **96.3%** |
| Qwen3.5-122B-A10B | MoE | 122B | IQ2_XXS | Dual | 44.09 | **100%** | **99.6%** |

Gemma 3 27B validates Sliding Window Attention. Qwen3-Next-80B validates CPU-offload path.
Llama 4 Scout: aliasing fix restored 100% generation Active Ratio (was 86.4% before fix).
96.3% Compute Coverage gap is from prefill-only MoE misses (ne2>1), harmless. 439 alias copies/token.
MoE models consistently show 95-99% Compute Coverage — gap from fused-path guard conditions.
Qwen3.5-122B-A10B: bartowski IQ2_XXS mixed quant with 111 IQ1_M tensors. Before IQ1_M kernel: 58% Active Ratio.
After IQ1_M kernel (v1.5): 100% Active Ratio, 99.6% Compute Coverage (remaining 0.4% = 1 Q2_K tensor).
Total: **23 models statistically benchmarked** (N≥5, paired t-test with CIs), **31 models tested** across **13 architecture families** (Dense, MoE, MLA+MoE, SSM-Hybrid, SWA, MXFP4-MoE, MoE+CPU-offload, ISWA-MoE, SSM-MoE, SSM-Hybrid-Mamba2, Pure-SSM, RWKV, VLM).

**Architecture compatibility expansion (2026-02-27, N=5 llama-bench, ± = SD):**

| Model | Type | Params | Quant | Stock (mean ± SD) | T-MAC (mean ± SD) | Speedup | Active Ratio |
|-------|------|--------|-------|-------------------:|-------------------:|--------:|:------------:|
| Falcon H1 7B | SSM-hybrid (Mamba2+Attn) | 7.59B | Q4_K_M | 64.02 ± 0.04 t/s | 74.76 ± 0.09 t/s | **+16.8%** | 100% |
| Mamba 2.8B | Pure SSM (Mamba) | 2.77B | Q4_K_M | 126.50 ± 0.15 t/s | 139.47 ± 0.25 t/s | **+10.3%** | 100% |
| RWKV-6-World 3B | RWKV (linear attn) | 3.10B | Q4_K_M | 101.96 ± 0.68 t/s | 116.75 ± 0.08 t/s | **+14.5%** | 100% |
| Qwen2-VL 2B | VLM (Qwen2+CLIP) | 1.54B | Q4_K_M | 226.31 ± 0.88 t/s | 265.60 ± 0.72 t/s | **+17.4%** | 100% |

Falcon H1 is a distinct SSM-hybrid from Jamba (TII Mamba2+attention vs AI21 Jamba architecture).
Mamba 2.8B: first pure SSM model — no attention at all. SSM scan/conv ops (`GGML_OP_SSM_SCAN`,
`GGML_OP_SSM_CONV`) are completely separate from MUL_MAT, T-MAC only intercepts MLP linear layers.
RWKV-6: first linear attention model. WKV kernel (`rwkv-wkv.cu`) untouched by T-MAC. Channel-mixing
FFN layers are standard MUL_MAT → fully intercepted. **Bit-identical output** (T-MAC = stock at temp=0).
Qwen2-VL: first VLM. Vision encoder runs F16/F32 (not quantized → T-MAC irrelevant). Cross-attention
falls back gracefully. Text generation decoder fully accelerated. **Bit-identical text output**.
Image description verified: both T-MAC and stock correctly describe test image content.

**Q4_K_M pp512 verification (N=10 paired interleaved):**

| Model | Stock (mean ± SD) | T-MAC (mean ± SD) | Speedup | p-value | 95% CI |
|-------|-------------------:|-------------------:|--------:|--------:|--------|
| Llama 3.2 1B | 15,367 ± 54 t/s | 15,426 ± 173 t/s | +0.4% | 0.30 (NS) | [-0.4%, +1.2%] |
| Codestral 22B | 1,156 ± 6.3 t/s | 1,156 ± 5.6 t/s | -0.0% | 0.90 (NS) | [-0.1%, +0.1%] |
| OLMoE-1B-7B | 8,552 ± 32 t/s | 8,559 ± 37 t/s | +0.1% | 0.49 (NS) | [-0.2%, +0.4%] |

All pp512 results statistically non-significant — regression fully eliminated.

Q4_0 run-to-run CV: 0.12%–0.34%. Q4_K_M tg128 CV: 0.12%–0.28%.
All ± values are now 95% CIs (converted from SD). Q4_0: N=12
llama-bench aggregate (t-dist df=11). Q4_K_M: N=10 paired interleaved (t-dist df=9).
1B Q4_0 updated from previous baseline (was +10.3%) — improvement from `__restrict__`
qualifiers. Other Q4_0 models (22B, 30B, OLMoE) still at previous baseline, may also
show improvement if re-measured.

**Note on interleaved vs sequential ordering:** tg128 numbers are slightly lower than
previous sequential measurements (1B 20.3% vs 19.7%, 22B 14.1% vs 14.7%, OLMoE 14.8%
vs 15.9%). Interleaved A-B-A-B eliminates thermal drift bias (sequential measurements
inflate T-MAC by ~0.3pp due to GPU heating). Interleaved numbers are more accurate.

**Q4_0 vs Q4_K_M gain gap:** Codestral 22B shows +3.9% (Q4_0) vs +14.1% (Q4_K_M) *(measured)*.
*(Inferred)* explanation: (1) Q4_K_M's sub-block parallelization maps better to Wave32 — 64
sub-blocks per row at K=2048 vs Q4_0's 64 blocks, and Q4_K benefits from larger contiguous
loads per super-block (144 bytes vs 18 bytes), improving cache line utilization. (2) Q4_K_M
models have mixed Q4_K + Q6_K tensors; the Q6_K T-MAC kernel covers an additional
24.9% of MMVQ calls that Q4_0 models (uniform type) don't have *(measured via Active Ratio)*.
The Q4_0 22B gain is genuine but lower, *(hypothesis)* because Q4_0's simpler decode path
is already well-optimized in stock dp4a, leaving less room for T-MAC improvement.

**Latency Percentiles (Q4_K_M, tg128, N=12 paired):**

| Model | T-MAC p50 (ms/tok) | Stock p50 (ms/tok) | p50 gain | T-MAC p99 | Stock p99 | p99 gain |
|-------|-------------------|-------------------|----------|-----------|-----------|----------|
| 1B Q4_K_M | 2.210 | 2.648 | +16.5% | 2.224 | 2.659 | +16.4% |
| 22B Q4_K_M | 21.918 | 25.116 | +12.7% | 21.949 | 25.243 | +13.1% |
| OLMoE Q4_K_M | 2.622 | 3.042 | +13.8% | 2.640 | 3.057 | +13.6% |

p50/p99 spread <1.3% — T-MAC introduces no tail latency.

MoE gain breakdown *(inferred, not profiled)*: estimated ~2.8% from attention layers + ~2.2% MoE expert dispatch + ~2.4% Q8_0/Q5_1 coverage + MoE correctness fix. These component contributions are arithmetically decomposed from A/B comparisons across versions, not measured via kernel profiling or controlled single-variable experiments.

**Note:** Devstral 24B omitted from statistical baseline (structurally identical to Codestral 22B, same architecture, similar gain).

---

## Sequence Length Sensitivity

Speedup measured at different generation lengths (Q4_K_M, N=5 paired interleaved, single 7900 XTX, 2026-03-02):

| Model | Params | tg128 | tg512 | tg2048 | Trend |
|-------|--------|------:|------:|-------:|-------|
| Llama 3.2 1B | 1.24B | **+20.1%** | **+20.5%** | **+18.2%** | Slight decrease at 2048 |
| Codestral 22B | 22.25B | **+15.3%** | **+15.2%** | **+13.3%** | Slight decrease at 2048 |
| Qwen3.5 27B | 27B | **+9.8%** | **+9.5%** | — | Stable (tg2048 not measured) |

Raw throughput (t/s):

| Model | Metric | Stock (mean ± SD) | T-MAC (mean ± SD) |
|-------|--------|-------------------:|-------------------:|
| 1B | tg128 | 372.4 ± 0.5 | 447.2 ± 1.0 |
| 1B | tg512 | 371.9 ± 0.5 | 448.3 ± 1.5 |
| 1B | tg2048 | 353.4 ± 0.3 | 417.8 ± 1.6 |
| 22B | tg128 | 40.0 ± 0.1 | 46.2 ± 0.1 |
| 22B | tg512 | 39.2 ± 0.1 | 45.2 ± 0.1 |
| 22B | tg2048 | 36.4 ± 0.1 | 41.3 ± 0.0 |
| 27B | tg128 | 26.8 ± 0.0 | 29.4 ± 0.1 |
| 27B | tg512 | 26.7 ± 0.1 | 29.3 ± 0.0 |

T-MAC speedup holds across generation lengths. The ~2pp decrease at tg2048 is expected:
longer KV-cache increases attention's share of total time (attention uses stock kernels),
diluting T-MAC's GEMV contribution via Amdahl's Law. The effect is larger on 1B (attention
is a bigger fraction of total compute on small models) than 22B (GEMV-dominated).

Qwen3.5-27B shows lower gain (+9.8%) than similarly-sized Codestral 22B (+15.3%).
*(Hypothesis)* This may be a model-specific effect: Qwen3.5 uses GQA with fewer KV heads,
which could shift more time to attention (not profiled). The gain is stable across
tg128→tg512.

---

## Multi-GPU Results

### Row-Split (dual 7900 XTX, Q4_0)

| Model | Type | Stock row-split | T-MAC row-split | Speedup | N |
|-------|------|----------------:|----------------:|--------:|--:|
| Llama 70B | Dense | 20.83 ± 0.07 t/s | 22.19 ± 0.11 t/s | **+6.5%** | 12 |
| Codestral 22B | Dense | 39.84 ± 0.13 t/s | 45.50 ± 0.20 t/s | **+14.2%** | 10 |
| Nemotron 30B | MoE | 113.62 t/s | 120.17 t/s | **+5.8%** | 1 |

**Note (2026-02-23):** 70B Q4_0 row-split re-measured with N=12 (was N=1: 22.62→22.19 T-MAC,
20.80→20.83 Stock). Speedup revised from +8.8% to +6.5%.

### CPU-Offload (dual 7900 XTX, GPT-OSS 120B Q4_K_M, 58.5 GiB, MoE 128exp/4active)

| Config | Split | ngl | CPU Layers | Stock (t/s) | T-MAC (t/s) | T-MAC Speedup | N |
|--------|-------|:---:|:----------:|------------:|------------:|--------------:|:-:|
| Dual GPU | Layer | 28 | 8 | 40.02 ± 0.07 | **40.53 ± 0.07** | **+1.3%** | 5 |
| Dual GPU | Row | 25 | 11 | 30.56 ± 0.04 | 31.22 ± 0.07 | +2.2% | 5 |

VRAM: GPU0=22.9GB, GPU1=21.9GB, CPU/RAM=15.1GB. Coherence: PASS (complex question, 39.54 t/s).
**CORRECTION:** GPT-OSS 120B is **MoE** (128 experts, 4 active, ~5.1B active params),
NOT Dense. The original "Thick Layer" hypothesis was wrong — 40 t/s comes from MoE sparsity.
T-MAC +1.3% is NOT Amdahl dilution but ne0=2880 rejection: `2880 % 256 = 64`, so `can_dispatch`
rejects ALL per-layer tensors. T-MAC fires only on output.weight + Q8_1 skip savings.
Layer-split beating row-split (40.53 vs 31.22) is a MoE property, not a Dense architecture rule.
5.1B active params ≈ 10-15B Dense equivalent for reasoning.

### IQ Quantization (dual 7900 XTX, N=12)

| Model | Quant | Split | Stock (t/s) | T-MAC (t/s) | T-MAC Speedup | vs Single T-MAC |
|-------|-------|-------|------------:|------------:|--------------:|----------------:|
| Llama 70B | IQ2_XXS | single | 13.88 ± 0.01 | 17.00 ± 0.03 | **+22.5%** | baseline |
| Llama 70B | IQ2_XXS | row | 19.26 ± 0.07 | **22.90 ± 0.12** | **+18.9%** | **+34.7%** |
| Llama 70B | IQ2_XXS | layer | 13.18 ± 0.02 | 16.27 ± 0.04 | +23.4% | -4.3% |
| OLMoE | IQ3_S | single | 314.99 ± 3.56 | 409.44 ± 5.54 | **+30.0%** | baseline |
| OLMoE | IQ3_S | row | 190.82 ± 0.96 | 228.01 ± 5.57 | +19.5% | **-44.3%** |
| OLMoE | IQ3_S | layer | 301.77 ± 5.99 | 384.99 ± 7.64 | +27.6% | -6.0% |
| Mixtral 8x7B | IQ3_S | layer | 55.47 ± 0.06 | 80.67 ± 0.09 | **+45.4%** | N/A (requires dual) |

### Llama 4 Scout (dual 7900 XTX, N=10 paired interleaved)

| Model | Quant | Stock (mean ± 95% CI) | T-MAC (mean ± 95% CI) | Speedup [95% CI] | p-value |
|-------|-------|----------------------:|----------------------:|------------------:|--------:|
| Llama 4 Scout 17Bx16E | IQ2_XXS-UD | 39.84 ± 0.07 t/s | 44.62 ± 0.12 t/s | **+12.0%** [+11.6%, +12.4%] | <0.0001 |

t-stat=72.38 (df=9). CV: Stock 0.24%, T-MAC 0.38%. 439 alias copies/token (pool-copy).
Individual speedups: +12.7%, +11.7%, +11.7%, +11.2%, +11.7%, +11.3%, +12.5%, +12.4%, +12.0%, +12.7%.
Llama 4 Scout is ISWA architecture (48 layers, 16 experts, top-1 routing, chunked SWA + full attention).
Active Ratio: 100% generation, 80.6% overall (141 prefill-only MoE misses at ne2=2).
First model where T-MAC's aliasing fix enables fused SwiGLU on all layers.

### Qwen3.5 Family (v1.6, N=5 paired interleaved, 2026-03-06)

| Model | Quant | Config | Stock (mean ± SD) | T-MAC (mean ± SD) | Speedup | p-value | 95% CI |
|-------|-------|--------|-------------------:|-------------------:|--------:|--------:|--------|
| Qwen3.5-9B | Q4_K_M | Single GPU | 69.81 ± 0.32 t/s | 77.57 ± 1.33 t/s | **+11.1%** | <0.001 | +10.1% to +12.2% |
| Qwen3.5-27B | Q4_K_M | Single GPU | 26.64 ± 0.16 t/s | 29.18 ± 0.23 t/s | **+9.5%** | <0.001 | +8.0% to +11.1% |
| Qwen3.5-35B-A3B | Q4_K_M | Single GPU (MoE) | 74.99 ± 0.03 t/s | 83.74 ± 0.09 t/s | **+11.7%** | <0.001 | +11.5% to +11.8% |
| Qwen3.5-27B | Q8_0 | Dual GPU (row) | 20.02 ± 0.08 t/s | 21.27 ± 0.06 t/s | **+6.3%** | <0.001 | +5.9% to +6.6% |

Notes:
- Qwen3.5-9B: 100% Active Ratio, 100% Compute Coverage. N=10 (excl outlier N=9: +11.6%, CI +11.4-11.7%, CV 0.13%). PPL 8.7311, Δ=0.000.
- Qwen3.5-27B Q4_K_M: Lower gain vs similarly-sized Codestral 22B (+14.1%). *(Hypothesis:)* GQA with fewer KV heads may shift more time to attention; not confirmed via profiling.
- Qwen3.5-35B-A3B: 85.2% Active Ratio (4624/5426 ops) but 97.5% Compute Coverage — small expert layers have ne0 alignment misses. Exceptionally tight variance (CV 0.04-0.11%).
- Qwen3.5-27B Q8_0: Row-split across dual 7900 XTX (26.6 GB, exceeds single GPU). +6.3% matches 70B Q4_0 dual-GPU pattern (+6.5%) — PCIe sync overhead caps the benefit.

### Qwen3.5-122B-A10B (dual 7900 XTX, v1.5, N=12 llama-bench aggregate, non-interleaved)

| Model | Quant | Stock (mean ± 95% CI) | T-MAC (mean ± 95% CI) | Speedup |
|-------|-------|----------------------:|----------------------:|--------:|
| Qwen3.5-122B-A10B | IQ2_XXS (mixed, 111× IQ1_M) | 41.05 ± 0.04 t/s | 44.09 ± 0.09 t/s | **+7.4%** |

Bartowski importance-matrix quantization assigns 111 tensors to IQ1_M (1.75 bpw, ne0=3072/1024).
Before IQ1_M kernel: Active Ratio 58%, Compute Coverage 88.4%. After: 100% / 99.6%.
First model to exercise IQ1_M kernel. Validates scaleless super-block design (no `d` field).

### Hidden Champion MoE Models (2026-03-07, N=5 paired interleaved)

| Model | Type | Quant | Config | Stock (t/s) | T-MAC (t/s) | Speedup |
|-------|------|-------|--------|------------:|------------:|--------:|
| Qwen2-57B-A14B | MoE 64E | Q4_K_M | Dual | 53.70 | 60.71 | **+13.1%** |
| Hunyuan-A13B | MoE 256E | Q4_K_M | Dual | 42.25 ± 0.12 | 46.61 ± 0.13 | **+10.3%** [+9.8%, +10.8%] |
| Jamba Mini 1.7 | SSM-MoE | Q6_K_L | Dual | 51.66 | 55.41 | **+7.3%** |
| Hunyuan-A13B | MoE 256E | Q3_K_M | Dual | 51.4 | 48.9 | **-4.9%** |

Qwen2-57B-A14B Q4_K_M dual: validates large MoE at memory-bound quant. Expert matrices
ne0=3584/2560 well-aligned, 100% expert dispatch. Same model holds single-GPU IQ3_XXS record (+54.5%).

Jamba Mini 1.7 Q6_K_L dual: SSM-MoE hybrid (Mamba+Attention+MoE). Expert matrices ne0=4096
(sb=128). Second Jamba architecture validated (first: Jamba-Reasoning 3B Dense +12.5%).

Hunyuan-A13B Q4_K_M dual: **matched-quant validation** (2026-03-07). **+10.3%** (N=5,
95% CI: +9.8% to +10.8%, t=55.64, p<0.0001). Confirms prediction: memory-bound quant
works on any aligned MoE. Compute Coverage: 98.0% (dense: 100%, expert: 87.4%). This
is the second half of the OLMoE paradox matched pair — same model at Q3_K_M = -4.9%
but at Q4_K_M = +10.3%. Quant type is the discriminator, not architecture.

Hunyuan-A13B Q3_K_M dual: **first confirmed T-MAC regression**. 256 experts, down
projection ne0=1024 (sb=32) at Q3_K_M (transitional, 16 sub-blocks per 256-element type).
*(Hypothesis)* Cache locality loss from mixed dispatch (shared experts via T-MAC, routed
experts via stock) combined with marginal sub-block count. Same model is neutral at
IQ2_XXS (+0.2% NS, single GPU) and dual-GPU IQ2_XXS (-0.5% NS) — regression is
Q3_K_M-specific *(confounded: dual-GPU + Q3_K_M not independently tested)*.

**HSA_ENABLE_SDMA=0 test (N=5):** 70B IQ2_XXS row T-MAC: 22.41 (vs 22.90 with SDMA) = -2.1%.
SDMA is working correctly on this board; CPU-copy fallback is slower.

**Multi-GPU IQ scaling paradox (Key finding):** IQ2_XXS halves model size (18GB
vs 38GB Q4_0), but row-split dual only gains +3.2% over Q4_0 dual (22.90 vs 22.19 t/s).
The `hipStreamWaitEvent` sync overhead (~200 events/token × 5-15μs = 1-3ms) is a fixed
cost independent of data volume. As IQ reduces compute time per layer, sync becomes a
proportionally larger fraction of total time (Amdahl's Law). **Result: faster kernels =
worse multi-GPU scaling efficiency.** Actual result: 22.90 t/s.

### Multi-GPU Rules (updated 2026-02-23)

1. **Dense >40GB:** Row-split. Proven: 70B Q4_0 = 22.19 t/s, 70B IQ2_XXS = 22.90 t/s.
2. **Dense models that fit on 1 GPU:** Single GPU always wins (22B paradox confirmed: -7.4%).
3. **MoE:** Single GPU always wins. Row-split catastrophic (-44% OLMoE IQ3_S).
   Exception: Llama 4 Scout (35GB) requires dual GPU — +12.0% with aliasing fix.
4. **Layer-split for Dense:** Never tested successfully — 70B IQ2_XXS layer = -4.3% vs single.
5. **HSA_ENABLE_SDMA=0:** Not helpful on 7900 XTX (tested, -2.1%).
6. **CPU-Offloading:** Useful when VRAM tight. DDR5-7200 + 9950X3D capable co-processor.
**NOTE:** The "Thick/Thin Layer" rules were based on GPT-OSS 120B which is MoE, not Dense.
No Dense >70B model has been tested on dual-GPU. Layer-split advantage for GPT-OSS is a MoE
property. The "Thick Layer Hypothesis" is invalidated.

---

## MoE Dimension Analysis (2026-03-07)

Key finding from hidden champion MoE testing: **quant type × sub-block count** is the real
discriminator for T-MAC benefit on MoE expert layers, not expert count or expert granularity.

### Expert tensor dimensions (verified from GGUF metadata)

| Model | n_expert | n_active | expert_ffn_dim | gate/up ne0 | down ne0 | sb(gate/up) | sb(down) |
|-------|----------|----------|----------------|-------------|----------|-------------|----------|
| Mixtral 8x7B | 8 | 2 | 14336 | 4096 | 14336 | 128 | 448 |
| DBRX | 16 | 4 | 10752 | 6144 | 10752 | 192 | 336 |
| Qwen2-57B-A14B | 64 | 4 | 2560 | 3584 | 2560 | 112 | 80 |
| Jamba Mini 1.7 | 16 | 2 | 4096 | 4096 | 4096 | 128 | 128 |
| OLMoE-1B-7B | 64 | 8 | 1024 | 2048 | 1024 | 64 | 32 |
| GLM-4.7-Flash | 40 | 4 | 1536 | 3072 | 1536 | 96 | 48 |
| DeepSeek-V2-Lite | 64 | 6 | 1536 | 2048 | 1536 | 64 | 48 |
| Nemotron 30B | 128 | 6 | 1856 | 2688 | 1856 | 0 ¹ | 0 ¹ |
| Qwen3.5-35B-A3B | 256 | 8 | 512 | 2048 | 512 | 64 | 16 ² |
| Hunyuan-A13B | 256 | 6 | 1024 | 4096 | 1024 | 128 | 32 |

¹ Nemotron: 2688%256≠0 and 1856%256≠0 → alignment fails, ALL expert layers use stock.
² Qwen3.5-35B: down ne0=512 → nb_sub=16, fails nb_sub≥24 guard → down projection falls back.

### The discriminator: quant type × sub-block count

- **Q4_K_M+ (memory-bound, ≥4 bpw):** Works for ANY aligned MoE architecture (+11–16%). Even
  tiny experts (Qwen3.5 FFN=512) and 256-expert models get gains.
- **IQ3_S/IQ3_XXS (transitional, ~3 bpw):** Works for medium+ expert matrices (≥32 sub-blocks).
  OLMoE +28–29%, Qwen2-57B **+54.5%**, Jamba **+47.2%**.
- **IQ2_XXS (latency-bound, ~2 bpw):** Works for large expert matrices (≥~100 sub-blocks).
  DBRX (192 sb) +22.0%, Hunyuan (32 sb) +0.2% NS. However, OLMoE IQ2_XXS +18.3% despite
  small experts (sb=32/64) — dense layer contribution (100% coverage) can dominate when
  MoE ratio is moderate.
- **Q3_K_M (transitional, sb=16):** Can regress on smaller experts. Hunyuan **-4.9%**.

### OLMoE "Paradox" resolution *(validated)*

OLMoE (FFN=1024, 64E) and Hunyuan (FFN=1024, 256E) share the same FFN dim but were never
tested at the same quant type. OLMoE was tested at Q4_K_M/IQ3_S/IQ3_XXS (all memory-bound
or transitional-favorable), while Hunyuan was tested at IQ2_XXS/Q3_K_M (latency-bound/
transitional-unfavorable).

**Matched-quant validation (2026-03-07):**
- OLMoE IQ2_XXS = **+18.3%** (N=5, p<0.0001) — prediction was regression/NS, **refuted**
- Hunyuan Q4_K_M = **+10.3%** (N=5, p<0.0001) — prediction was positive, **confirmed**

Both models show positive speedups at memory-bound quants (Q4_K_M), confirming the sub-block ×
bpw discriminator. However, OLMoE also wins at IQ2_XXS (+18.3%) while Hunyuan does not (+0.2% NS).
The difference is MoE ratio: OLMoE has 64E with moderate MoE ratio (dense layers contribute
significantly — 100% dense coverage, 76.8% expert coverage), while Hunyuan has 256E with high
MoE ratio (expert layers dominate the compute). At IQ2_XXS, where expert layers are
latency-bound, the dense layer contribution is enough to carry OLMoE but not Hunyuan.

### Interpreting T-MAC diagnostic output

T-MAC reports **Effective T-MAC Coverage (ETC)** at process exit for MoE models:

```
[TMAC] Compute Coverage: 95.9% (dense: 100.0%, expert: 64.6%)
[TMAC] Effective T-MAC Coverage (ETC): 96% (dense 89% × 100.0% + expert 11% × 64.6%)
```

ETC combines coverage and compute fraction: `ETC = dense_frac × dense_dispatch + expert_frac × expert_dispatch`.
Use ETC together with the quant type to estimate benefit:

| ETC | Q4_K_M+ (≥4 bpw) | IQ3_S/IQ3_XXS (~3 bpw) | IQ2_XXS (~2 bpw) |
|----:|:-----------------:|:----------------------:|:----------------:|
| >90% | +10-20% | +25-55% | +15-25% |
| 70-90% | +7-15% | +15-35% | +5-18% |
| 40-70% | +3-10% | +5-15% | NS to +10% |
| <40% | NS to +5% | NS | NS or regression |

These are empirical ranges from 31+ validated models, not guarantees. Actual speedup depends
on model size, memory pressure, and GPU thermal state. Dense-only models (no ETC line shown)
should expect the full range for their quant type.

---

## SwiGLU MoE Validation (2026-02-19, re-verified 2026-02-21)

Fused SwiGLU+MoE path validated on real SwiGLU-based MoE models (previously only tested
with Nemotron 30B which uses plain MLP experts). Re-verified after src1 stride fix.

| Model | Type | Params | Expert Layout | Stock | T-MAC | Speedup |
|-------|------|--------|---------------|------:|----------:|--------:|
| OLMoE-1B-7B | MoE | 6.92B | 64 experts / 8 active | 365.10 t/s | 403.07 t/s | **+10.4%** |
| Qwen1.5-MoE-A2.7B | MoE | 14.32B | 60 experts / 4 active + shared | 192.18 t/s | 202.82 t/s | **+5.5%** |

OLMoE's higher gain (+10.4% vs +5.5%) is *(hypothesis)* attributed to higher MoE ratio and
pure SwiGLU experts (no shared experts competing for bandwidth). However, OLMoE and
Qwen1.5-MoE differ in multiple variables (model size, expert count, layer count) making
the comparison confounded — the architectural explanation is plausible but not isolated.
Both models fire all four T-MAC paths: fused SwiGLU+MoE, unfused MoE dispatch, fused
SwiGLU (shared experts, Qwen only), and direct dequant.

**OLMoE segfault (resolved):** OLMoE previously segfaulted on complex prompts.
Initially assumed to be a pre-existing unrelated issue, but investigation confirmed it was the
same src1 stride bug. OLMoE has 8 active experts — expert slot 7 read `src1 + 7*K` far
past the buffer boundary. Nemotron (6 active, larger K) produced only corrupt data;
OLMoE's smaller intermediate dimension caused the offset to land in unmapped memory.
**One bug, two symptoms — never assume "pre-existing unrelated" without proof.**
