# Contributing to kuzco.cpp

kuzco.cpp is a specialized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
that adds custom GEMV kernels for AMD RDNA3 GPUs. Contributions are welcome.

## AI-Assisted Development

This project is developed with AI assistance (Claude, Gemini). We are transparent
about this because we believe it's the honest thing to do. The code is held to the
same standards as any hand-written code: it must be correct, well-tested, and
well-documented.

Every claim in this repository is backed by reproducible benchmarks with raw data.
If you find a result you can't reproduce, please open an issue.

## What Contributions Are Welcome

**Yes, please:**
- Bug reports (especially on RDNA3 hardware we haven't tested)
- Performance measurements on other RDNA3 SKUs (7900 XT, 7800 XT, W7900)
- New quantization type kernels (follow the pattern in `tmac.cu`)
- Benchmark reproductions (confirming or contradicting our published numbers)
- Documentation improvements

**Probably not:**
- NVIDIA GPU support (T-MAC uses AMD-specific intrinsics like `__builtin_amdgcn_ubfe`)
- Changes to upstream llama.cpp code (submit those [upstream](https://github.com/ggml-org/llama.cpp))
- Batch > 1 GEMM kernels (T-MAC targets single-token generation only)

## Reporting Bugs

### Is it a T-MAC bug or an upstream bug?

Quick test: run with `GGML_HIP_NO_TMAC=1`. If the bug persists, it's upstream —
please report it at [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues).

If the bug only occurs *without* `GGML_HIP_NO_TMAC=1` (i.e., with T-MAC active),
it's ours. Please include:

1. **GPU model** and `rocminfo | grep gfx`
2. **ROCm version** (`apt show hip-runtime-amd 2>/dev/null | grep Version`)
3. **Model file** (name, quantization type, source)
4. **Command** that reproduces the issue
5. **Expected vs actual** output
6. Output of `GGML_TMAC_LOG_MISSES=1` run (shows dispatch decisions)

## Development Setup

```bash
git clone https://github.com/nemekath/kuzco.cpp
cd kuzco.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build -j$(nproc)

# Smoke test
echo "" | HIP_VISIBLE_DEVICES=0 ./build/bin/llama-completion \
  -m your-model-Q4_K_M.gguf -p "2+2=" -n 20 -ngl 99

# Regression test
HIP_VISIBLE_DEVICES=0 scripts/tmac-regression.sh --quick
```

## Code Organization

| File | What to modify |
|------|----------------|
| `ggml/src/ggml-cuda/tmac.cu` | Kernel implementations and quant type traits |
| `ggml/src/ggml-cuda/tmac.cuh` | Guards, dispatch, type support list |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Dispatch sites (only if adding new fusion patterns) |
| `ggml/src/ggml-cuda/mmvq.cu` | Fused SwiGLU dispatch (only if changing fusion logic) |

**Adding a new quantization type:**
1. Add `block_*_internal` struct in `tmac.cu` with `static_assert` for size
2. Add `tmac_quant_traits<GGML_TYPE_*>` specialization
3. Add `tmac_*_dot()` or `tmac_*_subblock_dot()` function
4. Add case to `tmac_dispatch_simple()` and `tmac_dispatch_fused()` in `tmac.cuh`
5. Add type to `ggml_cuda_tmac_is_supported_type()` in `tmac.cuh`
6. Run `scripts/tmac-regression.sh` with a model using the new type
7. Verify perplexity delta = 0.000 on wikitext-2

## Testing Requirements

Before submitting:

1. **Build succeeds:** `cmake --build build -j$(nproc)` with zero warnings in tmac.cu/tmac.cuh
2. **Smoke test passes:** Correct output on at least one model
3. **No regressions:** `scripts/tmac-regression.sh --quick` passes (N=2, ~5 min)
4. **Perplexity unchanged:** For new quant types or accumulation changes, verify
   PPL delta = 0.000 on wikitext-2

For performance claims, use N>=5 paired interleaved runs with `scripts/tmac-regression.sh`.

## Commit Style

```
type(scope): short description

Longer explanation if needed.
```

Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`.
Scope: `tmac`, `dispatch`, `bench`, `build`, `ci`.

Example: `perf(tmac): v9.2 fuse bias into GEMV kernel (+20.3% on 1B Q4_K_M)`

## License

By contributing, you agree that your contributions will be licensed under the
MIT License, same as the rest of the project.
