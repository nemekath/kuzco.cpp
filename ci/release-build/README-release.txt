kuzco.cpp — Prebuilt Release for AMD RDNA3
==========================================

Requirements:
  - AMD RDNA3 GPU (RX 7900 XTX / XT / W7900 or similar gfx1100/1101/1102)
  - ROCm 6.x or 7.x runtime (hip-runtime-amd package)
  - Ubuntu 22.04 or compatible (glibc 2.35+)
  - libomp (ships with ROCm — if missing, add ROCm LLVM to your path:
    export LD_LIBRARY_PATH="/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH")

Quick Start:
  tar xzf kuzco.cpp-*.tar.gz
  cd kuzco.cpp-*/
  export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"
  HIP_VISIBLE_DEVICES=0 ./bin/llama-cli -m model.gguf -ngl 99

System-Wide Install:
  sudo cp bin/* /usr/local/bin/
  sudo cp lib/* /usr/local/lib/
  sudo ldconfig

Local Install (no root):
  export PATH="$(pwd)/bin:$PATH"
  export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"

Usage Notes:
  - Set HIP_VISIBLE_DEVICES=0 if you have an integrated GPU (avoids segfault)
  - T-MAC kernels activate automatically on RDNA3 — no configuration needed
  - To disable T-MAC (stock fallback): export GGML_HIP_NO_TMAC=1

Binaries:
  llama-cli         Interactive chat / text generation
  llama-server      OpenAI-compatible HTTP server
  llama-bench       Throughput benchmarking (pp/tg)
  llama-quantize    Convert / re-quantize GGUF models
  llama-perplexity  Perplexity evaluation on datasets
  llama-completion  Single-shot text completion

Libraries:
  libllama.so       Core inference library
  libggml.so        Tensor computation framework
  libggml-base.so   Base ggml operations
  libggml-hip.so    HIP/ROCm backend (includes T-MAC kernels)
  libggml-cpu.so    CPU backend fallback
  libmtmd.so        Multimodal support (required by llama-cli)

Source & Issues: https://github.com/nemekath/kuzco.cpp
