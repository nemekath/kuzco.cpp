# Raw Benchmark Data

This directory contains raw benchmark data from T-MAC performance measurements.
All data was collected on the system described below.

## System Configuration

| Property | Value |
|----------|-------|
| GPU | AMD Radeon RX 7900 XTX (gfx1100) |
| VRAM | 24 GB GDDR6 |
| CPU | AMD Ryzen 9 9950X3D |
| RAM | 64 GB DDR5 |
| ROCm | 7.1.1 (hip 7.1.52802) |
| OS | Ubuntu 24.04, kernel 6.17.0-14-generic |
| kuzco.cpp | commit 5b9abcc7b |

## CSV Format

All files use the same schema:

```csv
# date=...,rocm=...,hip=...,gpu=...,kernel=...,commit=...
section,model,metric,run,variant,tokens_per_sec
```

- **section**: Test category (e.g., `iq-absolute`, `q4km-baseline`)
- **model**: GGUF model filename (without path)
- **metric**: `tg128` (128-token generation) or `pp512` (512-token prefill)
- **run**: Run number within the interleaved A-B sequence
- **variant**: `tmac` or `stock`
- **tokens_per_sec**: Measured throughput

## Methodology

- **Interleaved A-B**: Runs alternate tmac/stock to cancel thermal drift
- **N=10**: 12 total runs, first 2 discarded as warmup (N_eff=10)
- **Stock baseline**: Same binary with `GGML_HIP_NO_TMAC=1`
- **Statistical test**: Paired t-test, p < 0.05, 95% CI

## Reproducing

Use `scripts/reproduce-benchmarks.sh` to generate your own data on your hardware.
See [docs/tmac/benchmarks.md](../../docs/tmac/benchmarks.md) for aggregated results
with confidence intervals and p-values.

## Files

| File | Contents | N |
|------|----------|---|
| `iq-types-n10-20260302.csv` | IQ3_XXS, IQ3_S, IQ2_XXS, IQ4_XS, IQ1_M (Llama 1B) | 10 |
| `iq2xs-rerun-n10-20260302.csv` | IQ2_XXS, IQ2_XS rerun with P16 sign LUT elimination | 10 |
| `phi-4-mini-3.8B-Q4_K_M-n5-20260302.csv` | Phi-4-mini model zoo validation | 5 |
