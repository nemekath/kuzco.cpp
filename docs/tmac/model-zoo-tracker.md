# T-MAC Model Zoo — Validation Tracker

> Status codes: `-` not started, `DL` downloading, `OK` passed, `FAIL` failed, `SKIP` skipped
>
> Tiers: T1 = Smoke test, T2 = Long-form coherence (2048 tok), T3 = Perplexity (wikitext-2), T4 = Benchmark N=5
>
> All tests use `HIP_VISIBLE_DEVICES=0` (single GPU) unless noted. Dual GPU uses `0,1`.
>
> Last updated: 2026-03-06

---

## Already Validated (from previous benchmarks)

These models were benchmarked before the model zoo expansion. Results in `benchmarks.md`.

| Model | Params | Arch | Quant | Size | T1 | T2 | T3 | T4 | Speedup | Notes |
|-------|--------|------|-------|-----:|:--:|:--:|:--:|:--:|--------:|-------|
| Llama 3.2 1B | 1.24B | Dense | Q4_K_M | 0.8 GB | OK | OK | OK | OK | **+20.3%** | N=10, reference |
| Llama 3.2 1B | 1.24B | Dense | Q4_0 | 0.7 GB | OK | OK | OK | OK | **+14.8%** | N=12 |
| Llama 3.2 1B | 1.24B | Dense | IQ3_XXS | 0.5 GB | OK | — | OK | OK | **+36.6%** | N=10, P16 |
| Llama 3.2 1B | 1.24B | Dense | IQ3_S | 0.6 GB | OK | — | OK | OK | **+34.4%** | N=10 |
| Llama 3.2 1B | 1.24B | Dense | IQ2_XXS | 0.4 GB | OK | — | OK | OK | **+24.6%** | N=10, P16 |
| Llama 3.2 1B | 1.24B | Dense | IQ2_XS | 0.4 GB | OK | — | OK | OK | **+17.4%** | N=10, P16 |
| Llama 3.2 1B | 1.24B | Dense | IQ4_XS | 0.6 GB | OK | — | OK | OK | **+11.1%** | N=10 |
| Llama 3.2 1B | 1.24B | Dense | IQ1_M | 0.3 GB | OK | — | OK | OK | **+11.9%** | N=10 |
| Llama 3.2 1B | 1.24B | Dense | Q5_K_M | 0.9 GB | OK | — | OK | — | validated | — |
| Llama 3.2 1B | 1.24B | Dense | Q5_0 | 0.8 GB | OK | — | OK | — | validated | — |
| Codestral 22B | 22.25B | Dense | Q4_K_M | 12 GB | OK | OK | OK | OK | **+14.1%** | N=10 |
| Codestral 22B | 22.25B | Dense | Q4_0 | 12 GB | OK | — | — | OK | **+3.9%** | N=12 |
| QwQ-32B | 32B | Dense | Q4_K_M | 20 GB | OK | — | — | OK | **+13.5%** | N=10 |
| Ministral 14B | 14B | Dense+SWA | Q4_K_M | 8.4 GB | OK | — | — | OK | **+14.1%** | N=10 |
| DeepSeek-R1 8B | 8B | Dense | Q4_K_M | 4.6 GB | OK | — | — | OK | **+18.4%** | N=10 |
| Llama 3.1 8B | 8B | Dense | Q5_K_M | 5.7 GB | OK | — | OK | — | validated | — |
| OLMoE 1B-7B | 6.92B | MoE | Q4_K_M | 4.1 GB | OK | — | — | OK | **+14.8%** | N=10 |
| OLMoE 1B-7B | 6.92B | MoE | Q4_0 | 3.8 GB | OK | — | — | OK | **+10.4%** | N=12 |
| OLMoE 1B-7B | 6.92B | MoE | IQ3_S | 2.5 GB | OK | — | OK | OK | **+29.1%** | N=10 |
| OLMoE 1B-7B | 6.92B | MoE | IQ3_XXS | 2.2 GB | OK | — | OK | OK | **+28.2%** | N=10 |
| Nemotron 30B | 30B | MoE | Q4_0 | 17 GB | OK | — | — | OK | **+7.9%** ¹ | N=12 |
| Nemotron 30B | 30B | MoE | Q4_K_M | 19 GB | OK | — | — | — | validated ¹ | — |
| GLM-4.7-Flash | ~16B | MLA+MoE | Q4_K_M | 10 GB | OK | — | — | OK | **+15.2%** | N=10, nb_sub guard |
| Devstral 24B | 24B | Dense | Q6_K_L | 20 GB | OK | — | — | OK | **+5.5%** | N=5 |
| Jamba 3B | 3B | SSM-Hybrid | Q4_K_M | 1.8 GB | OK | — | — | OK | **+12.5%** | N=5 |
| Mamba 2.8B | 2.8B | Pure-SSM | Q4_K_M | 1.9 GB | OK | — | — | OK | **+10.3%** | N=5 |
| RWKV-6 3B | 3B | RWKV | Q4_K_M | 1.8 GB | OK | — | — | OK | **+14.5%** | N=5 |
| Qwen2-VL 2B | 2B | VLM | Q4_K_M | 0.9 GB | OK | — | — | OK | **+17.4%** | N=5 |
| Falcon H1 7B | 7B | SSM-Hybrid-Mamba2 | Q4_K_M | 4.3 GB | OK | — | — | OK | **+16.8%** | N=5 |
| Qwen3.5-9B | 9B | Dense | Q4_K_M | 5.3 GB | OK | OK | OK | OK | **+11.1%** | N=10, PPL Δ=0.000 |
| Qwen3.5-27B | 27B | Dense | Q4_K_M | 16 GB | OK | — | — | OK | **+9.5%** | N=5 |
| Qwen3.5-27B | 27B | Dense | Q8_0 | 27 GB | OK | — | — | OK | **+6.3%** | N=5, dual GPU |
| Qwen3.5-35B-A3B | 35B/3B | MoE | Q4_K_M | 20 GB | OK | — | — | OK | **+11.7%** | N=5, 97.5% coverage |
| GPT-OSS 20B | 20B | MXFP4-MoE | MXFP4 | — | OK | — | — | OK | **+8.9%** | N=5 |
| Llama 3.3 70B | 70B | Dense | IQ2_XXS | 19 GB | OK | — | — | OK | **+25.8%** | N=10, dual GPU |
| Mixtral 8x7B | 47B/13B | MoE | IQ3_S | 18 GB | OK | — | — | OK | **+45.4%** | N=10, dual GPU |
| Llama 4 Scout | 109B/17B | ISWA-MoE | IQ2_XXS | 24 GB | OK | — | — | OK | **+12.0%** | N=10, dual GPU |
| Qwen3.5-122B | 122B/10B | MoE | IQ2_XXS | 28 GB | OK | — | — | OK | **+7.4%** | N=10, dual GPU |
| Qwen2-57B-A14B | 57B/14B | MoE (64E) | IQ3_XXS | 14 GB | OK | — | — | OK | **+54.5%** | N=12, single GPU, record |
| Qwen2-57B-A14B | 57B/14B | MoE (64E) | Q4_K_M | 33 GB | OK | — | — | OK | **+13.1%** | N=5, dual GPU |
| Jamba Mini 1.7 | 52B | SSM-MoE | IQ3_XXS | 14 GB | OK | — | — | OK | **+47.2%** | N=5, single GPU |
| Jamba Mini 1.7 | 52B | SSM-MoE | Q6_K_L | 40 GB | OK | — | — | OK | **+7.3%** | N=5, dual GPU |
| DBRX | 132B/36B | MoE (16E) | IQ2_XXS | 30 GB | OK | — | — | OK | **+22.0%** | N=5, dual GPU |
| DeepSeek-V2-Lite | 16B | MoE (MLA) | Q4_K_M | 10 GB | OK | — | — | OK | **+15.9%** | N=5, single GPU |
| OLMoE-1B-7B | 6.92B | MoE (64E) | IQ2_XXS | 1.8 GB | OK | — | — | OK | **+18.3%** | N=5, matched-quant |
| Hunyuan-A13B | 85B/13B | MoE (256E) | IQ2_XXS | 19 GB | OK | — | — | — | +0.2% NS | N=5 |
| Hunyuan-A13B | 85B/13B | MoE (256E) | Q4_K_M | 49 GB | OK | — | — | OK | **+10.3%** | N=5, dual GPU, matched-quant |
| Hunyuan-A13B | 85B/13B | MoE (256E) | Q3_K_M | 38 GB | OK | — | — | — | **-4.9%** ² | N=5, dual GPU |

¹ Nemotron expert dimensions (2688, 1856) both fail 256-alignment — ALL expert MoE layers
fall back to stock. The +7.9% Q4_0 / validated Q4_K_M gains come entirely from dense and
shared-expert layers. Active Ratio diagnostic confirms expert coverage = 0%.

² Hunyuan Q3_K_M is the first confirmed T-MAC regression. Root cause *(hypothesis)*:
marginal sub-block count (sb=32 down projection) at transitional Q3_K_M combined with
cache locality loss from mixed dispatch (shared experts via T-MAC, routed experts via stock).
Same model is neutral at IQ2_XXS (+0.2% NS).

---

## Statistical Methodology

**Confidence intervals (CIs)** report 95% paired t-test intervals. Two categories:

- **Exact CIs** (no marker): Computed from raw per-run CSV data. Available for Phi-4-mini
  3.8B (Wave 1 #4), Maverick 128E (Wave 4 #3), and Scout Q4_K_M (Wave 3 #5, N=10 rerun).
- **Approximate CIs** (marked with `†`): Estimated from system variance characterization.
  Paired speedup SD on this system is 0.28–0.49% across 6 independently measured N=10
  models (see `benchmarks.md`). Using the conservative upper bound (SD=0.4%) with
  t_crit(df=4, 95%)=2.776 for N=5: approximate 95% CI half-width ≈ ±0.5pp. All N=5 zoo
  entries with speedup >5% are significant beyond any reasonable doubt (lower CI bound > 0).
- **Entries with N=10** (QwQ-32B, Ministral 14B): Exact CIs available in `benchmarks.md`.

**Existing N=10 models** from the "Already Validated" section already have full CIs with
p-values in `benchmarks.md`. They are not duplicated here.

---

## Wave 1: Small Models (1–8B)

> *All T4 speedups measured at tg128, batch=1. T-MAC guards reject batch>1, falling back
> to stock kernels. Results do not reflect batch inference performance.*

| # | Model | Params | Arch | Quant | ~Size | Location | DL | T1 | T2 | T3 | T4 | PPL T-MAC | PPL Stock | Speedup |
|--:|-------|--------|------|-------|------:|----------|:--:|:--:|:--:|:--:|:--:|----------:|----------:|--------:|
| 1 | Qwen3 8B | 8B | Dense | Q4_K_M | 5.0 GB | llm-data | OK | OK | OK | OK | OK | 10.4645 | 10.4645 | **+19.5%** [~19.0, ~20.0]† N=5 |
| 2 | Qwen3 4B | 4B | Dense | Q4_K_M | 2.5 GB | llm-data | OK | OK | OK | OK | OK | 16.6260 | 16.6260 | **+17.0%** [~16.5, ~17.5]† N=5 |
| 3 | Phi-4 14B | 14B | Dense | Q4_K_M | 8.4 GB | llm-data | OK | OK | OK | OK | OK | 6.6204 | 6.6204 | **+15.0%** [~14.5, ~15.5]† N=5 |
| 4 | Phi-4-mini 3.8B | 3.8B | Dense | Q4_K_M | 2.4 GB | llm-data | OK | OK | OK | OK | OK | 10.7277 | 10.7277 | **+11.3%** [10.9, 11.6] N=5 |
| 5 | Gemma 3 12B | 12B | Dense | Q4_K_M | 7.0 GB | llm-data | OK | OK | OK | OK | OK | 9.8712 | 9.8712 | **+16.1%** [~15.6, ~16.6]† N=5 |
| 6 | Gemma 3 4B | 4B | Dense | Q4_K_M | 2.5 GB | llm-data | OK | OK | OK | OK | OK | 17.0976 | 17.0976 | **+14.3%** [~13.8, ~14.8]† N=5 |
| 7 | DeepSeek-R1-0528-Qwen3-8B | 8B | Dense | Q4_K_M | 5.0 GB | llm-data | OK | OK | OK | OK | OK | 11.5985 | 11.5985 | **+17.4%** [~16.9, ~17.9]† N=5 |
| 8 | DeepSeek-R1-Distill-Qwen-7B | 7B | Dense | Q4_K_M | 4.5 GB | llm-data | OK | OK | OK | OK | OK | 25.4086 | 25.4086 | **+17.3%** [~16.8, ~17.8]† N=5 |
| 9 | Qwen2.5-Coder 7B | 7B | Dense | Q4_K_M | 4.5 GB | llm-data | OK | OK | OK | OK | OK | 9.9962 | 9.9962 | **+17.0%** [~16.5, ~17.5]† N=5 |
| 10 | Mistral Nemo 12B | 12B | Dense | Q4_K_M | 7.5 GB | llm-data | OK | OK | OK | OK | OK | 6.4797 | 6.4797 | **+15.9%** [~15.4, ~16.4]† N=5 |
| 11 | Mistral 7B v0.3 | 7B | Dense | Q4_K_M | 4.4 GB | llm-data | OK | OK | OK | OK | OK | 6.2338 | 6.2338 | **+18.9%** [~18.4, ~19.4]† N=5 |
| 12 | Falcon H1R 7B | 7B | SSM-Hybrid | Q4_K_M | 4.5 GB | — | SKIP | SKIP | SKIP | SKIP | SKIP | — | — | No GGUF repo |
| 13 | RWKV-7 7B | 7B | RWKV | Q4_K_M | 4.0 GB | — | SKIP | SKIP | SKIP | SKIP | SKIP | — | — | No GGUF quant yet |
| 14 | Qwen3-VL 8B | 8B | VLM | Q4_K_M | 5.0 GB | llm-data | OK | OK | OK | OK | OK | 8.6763 | 8.6763 | **+16.6%** [~16.1, ~17.1]† N=5 |

**Wave 1 total new:** ~65 GB

---

## Wave 2: Medium Models (14–32B)

> *All T4 speedups measured at tg128, batch=1. T-MAC guards reject batch>1, falling back
> to stock kernels. Results do not reflect batch inference performance.*

| # | Model | Params | Arch | Quant | ~Size | Location | DL | T1 | T2 | T3 | T4 | PPL T-MAC | PPL Stock | Speedup |
|--:|-------|--------|------|-------|------:|----------|:--:|:--:|:--:|:--:|:--:|----------:|----------:|--------:|
| 1 | Qwen3 32B | 32B | Dense | Q4_K_M | 18.4 GB | llm-data | OK | OK | OK | SKIP | OK | NaN@38 | NaN@38 | **+15.5%** [~15.0, ~16.0]† N=5 |
| 2 | Qwen3 14B | 14B | Dense | Q4_K_M | 8.3 GB | llm-data | OK | OK | OK | OK | OK | 9.3056 | 9.3056 | **+15.5%** [~15.0, ~16.0]† N=5 |
| 3 | DeepSeek-R1-Distill-Qwen-32B | 32B | Dense | Q4_K_M | 18.4 GB | llm-data | OK | OK | OK | OK | OK | 7.1262 | 7.1262 | **+15.4%** [~14.9, ~15.9]† N=5 |
| 4 | QwQ-32B | 32B | Dense | Q4_K_M | 18.4 GB | llm-data | OK | OK | OK | - | OK | — | — | **+13.5%** [13.2, 13.8] N=10 |
| 5 | Gemma 3 27B | 27B | Dense | Q4_K_M | 15.4 GB | llm-data | OK | OK | OK | OK | OK | 8.4812 | 8.4812 | **+12.7%** [~12.2, ~13.2]† N=5 |
| 6 | Mistral Small 3.2 24B | 24B | Dense | Q4_K_M | 13.3 GB | llm-data | OK | OK | OK | OK | OK | 5.5325 | 5.5325 | **+11.7%** [~11.2, ~12.2]† N=5 |
| 7 | Qwen2.5-Coder 32B | 32B | Dense | Q4_K_M | 18.4 GB | llm-data | OK | OK | OK | OK | OK | 7.7705 | 7.7705 | **+15.8%** [~15.3, ~16.3]† N=5 |
| 8 | Qwen3-30B-A3B (MoE) | 30B/3.3B | MoE | Q4_K_M | 17.2 GB | llm-data | OK | OK | OK | OK | OK | 9.6500 | 9.6500 | **+15.7%** [~15.2, ~16.2]† N=5 |
| 9 | Ministral 14B (v3) | 14B | Dense+SWA | Q4_K_M | 8.4 GB | llm-data | OK | OK | OK | - | OK | — | — | **+14.1%** [13.8, 14.3] N=10 |
**Wave 2 total new:** ~115 GB

---

## Wave 3: Large Models (70B) — Dual GPU

> *All T4 speedups measured at tg128, batch=1. T-MAC guards reject batch>1, falling back
> to stock kernels. Results do not reflect batch inference performance.*

| # | Model | Params | Arch | Quant | ~Size | Location | DL | T1 | T2 | T3 | T4 | PPL T-MAC | PPL Stock | Speedup |
|--:|-------|--------|------|-------|------:|----------|:--:|:--:|:--:|:--:|:--:|----------:|----------:|--------:|
| 1 | Llama 3.3 70B | 70B | Dense | Q4_K_M | 42 GB | games | OK | OK | OK | OK | OK | 4.1202 | 4.1202 | **+10.1%** [~9.6, ~10.6]† N=5 |
| 2 | Qwen3 72B | 72B | Dense | Q4_K_M | 47 GB | games | OK | OK | OK | OK* | OK | 8975.7§ | 8975.7§ | **+10.4%** [~9.9, ~10.9]† N=5 |
| 3 | DeepSeek-R1-Distill-Llama-70B | 70B | Dense | Q4_K_M | 42 GB | games | OK | OK | OK | OK | OK | 6.6134 | 6.6134 | **+10.4%** [~9.9, ~10.9]† N=5 |
| 4 | Qwen2.5 72B | 72B | Dense | Q4_K_M | 47 GB | games | OK | OK | OK | OK | OK | 4.5590 | 4.5590 | **+10.5%** [~10.0, ~11.0]† N=5 |
| 5 | Llama 4 Scout (Q4_K_M) | 109B/17B | ISWA-MoE | Q4_K_M | 65 GB | games | OK | OK | OK | OK | OK | 9.1006 | 9.1006 | **Parity** (+2.5% [-9.4, +14.4] NS‡) N=10 |

**Wave 3 total new:** ~240 GB (overflow to /mnt/games/)

---

## Wave 4: Extra Large (100B+) — Dual GPU + Offload

> *All T4 speedups measured at tg128, batch=1. T-MAC guards reject batch>1, falling back
> to stock kernels. Results do not reflect batch inference performance.*

| # | Model | Params Active | Arch | Quant | ~Size | Location | DL | T1 | T2 | T3 | T4 | PPL T-MAC | PPL Stock | Speedup |
|--:|-------|--------------|------|-------|------:|----------|:--:|:--:|:--:|:--:|:--:|----------:|----------:|--------:|
| 1 | Qwen3 235B-A22B | 22B | MoE | IQ4_XS | 126 GB | — | - | - | - | - | - | — | — | SKIP (no IQ2_XXS avail, IQ4_XS 126 GB) |
| 2 | DeepSeek-R1-0528 671B | MoE | MoE | IQ2_XXS | 217 GB | — | - | - | - | - | - | — | — | SKIP (doesn't fit 159 GB free) |
| 3 | Llama 4 Maverick 128E | 17B | ISWA-MoE | UD-IQ2_XXS | 135 GB | games | OK | OK | OK | SKIP | OK | — | — | **-0.4%** [-1.7, +0.9] NS N=5, ngl=13 CPU offload |

**Wave 4 COMPLETE.** Only Maverick testable. Speedup NS due to heavy CPU offload (13/~200 layers on GPU)

---

## Storage Budget

| Mount | Total | Used | Free | Allocated |
|-------|------:|-----:|-----:|----------:|
| `/mnt/llm-data/` (NVMe) | 916 GB | 669 GB | 201 GB | Wave 1 + Wave 2 (~180 GB) |
| `/mnt/games/` (NVMe) | 2.8 TB | 2.6 TB | 220 GB | Wave 3 (~170 GB) + Wave 4 |
| `/home/` (SATA) | 915 GB | 714 GB | 155 GB | Symlinks only |

## Notes

- Wikitext-2 test corpus: `/mnt/llm-data/wikitext-2-raw/wiki.test.raw`
- Model symlinks: `/home/benjamin/llama-tmac-real/` → actual storage location
- Overflow models (>40 GB): `/mnt/games/kuzco-models/`
- All PPL deltas must be < 0.1 (bit-identical validation)
- † **Approximate 95% CI:** Estimated from system variance characterization (paired speedup SD = 0.28–0.49% across 6 N=10 models). Conservative upper bound SD=0.4%, t_crit(df=4)=2.776 → half-width ≈ ±0.5pp. Exact CIs (no marker) computed from raw per-run CSV data
- § Qwen3-72B PPL=8975.7: instruct-tuned model/wikitext-2 incompatibility (stock identical). Same family as Qwen3-32B NaN@38
- * Qwen3-72B T3 required `-b 128` (OOM at default batch=2048 on dual 24 GB GPUs)
- Wave 3 repo fix: `Qwen/Qwen3-72B-GGUF` doesn't exist; used `mradermacher/Qwen3-72B-Instruct-GGUF`
- Wave 3 Qwen2.5-72B T3 also required `-b 128` due to 72B model size on dual GPU
- ‡ Scout Q4_K_M uses CPU offload (`-ngl 36`, ~26 GB on CPU). Reported as **Parity** — CPU offload CV=14-18% (50× normal) drowns the T-MAC signal. N=10 rerun: +2.5% [-9.4%, +14.4%] NS (outlier-excluded, N=9). Compare Scout IQ2_XXS (+12.0% fully on GPU)
- Scout Q4_K_M split into 2 shards: symlinks must use original shard names, not custom name
- Scout T3 perplexity: ~75 min per run at `-b 128 -ngl 36` (CPU offload slows batch processing)
- Maverick 128E T4: `llama-bench` defaults `--mmap 0` → 127 GB malloc → OOM crash. **Must use `--mmap 1`**
- Maverick 128E: `/mnt/games/` NTFS partition may need `sudo mount -o ro /mnt/games` if dirty flag set
- Maverick 128E: -0.4% [-1.7, +0.9] NS (p=0.47) — only 13 GPU layers, T-MAC coverage ~6%. Same as Scout Q4_K_M pattern
