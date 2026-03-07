#pragma once
#include "common.cuh"

#ifdef GGML_HIP_TMAC

// ═══════════════════════════════════════════════════════════════════════
// T-MAC Dispatch Site Index
// ═══════════════════════════════════════════════════════════════════════
// Site 1/6: ggml-cuda.cu  — Unfused non-split    (ne2 GQA loop)
// Site 2/6: ggml-cuda.cu  — Unfused split         (row-split callback)
// Site 3/6: ggml-cuda.cu  — MoE unfused           (src1 stride)
// Site 4/6: mmvq.cu       — Fused SwiGLU          (non-split + MoE)
// Site 5/6: ggml-cuda.cu  — Fused SwiGLU split    (row-split callback)
// Site 6/6: ggml-cuda.cu  — Bias-fused non-split  (ne2 loop + bias)
//
// Each site has unique context (ne2 loops, MoE stride, split callbacks,
// bias extraction) that must remain at the call site.  Central dispatch
// is handled by tmac_dispatch_simple() and tmac_dispatch_fused() below.
//
// Why not a single tmac_try_dispatch() entry point?
// Each site has genuinely different pre-dispatch logic: site 1 iterates ne2
// for GQA broadcasting, site 2/5 are split-tensor callbacks with pre-sliced
// pointers, site 3 computes conditional src1 stride for MoE, site 4 extracts
// fused gate pointers, site 6 extracts bias from a fused ADD node.  A unified
// entry would need 15+ parameters or opaque context structs, adding indirection
// that obscures the dispatch logic rather than simplifying it.
// ═══════════════════════════════════════════════════════════════════════

// Shared enablement check: auto-detect RDNA3, with GGML_HIP_NO_TMAC=1 opt-out.
// Evaluated per-call (uses TLS device via ggml_cuda_get_device()), so mixed-arch
// multi-GPU setups are correct: RDNA3 GPU returns true, non-RDNA3 returns false.
// RDNA 3.5 (gfx1150/gfx1151) is included — GGML_CUDA_CC_IS_RDNA3 covers both generations.
static inline bool ggml_cuda_tmac_enabled() {
    static const bool disabled = (getenv("GGML_HIP_NO_TMAC") != nullptr);
    if (disabled) { return false; }
    return GGML_CUDA_CC_IS_RDNA3(
        ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

// Type-check: is this a quant type T-MAC supports?
static inline bool ggml_cuda_tmac_is_supported_type(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q5_1
        || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_MXFP4
        || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K
        || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ3_S
        || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS
        || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S
        || type == GGML_TYPE_IQ1_M;
}

// Combined guard: enabled (RDNA3 auto-detect) + supported type + dimension constraints (ne0 = K, input dim)
// The RDNA3 check in enabled() prevents T-MAC intrinsics (__builtin_amdgcn_ubfe) from
// running on NVIDIA GPUs or older AMD architectures where they would crash.
// Q4_K uses 256-element super-blocks requiring ne0 % 256 == 0 (always true for valid Q4_K tensors).
static inline bool ggml_cuda_tmac_can_dispatch(ggml_type type, int64_t ne0) {
    if (!ggml_cuda_tmac_enabled()) return false;
    if (!ggml_cuda_tmac_is_supported_type(type)) return false;
    if (type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K
        || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S
        || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS
        || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S
        || type == GGML_TYPE_IQ1_M) {
        if (ne0 % 256 != 0 || ne0 < 256) return false;
        // Sub-block warp efficiency guard: Q4_K has 8 sub-blocks per 256-element super-block,
        // Q6_K has 16. With Wave32, nb_sub < 24 means <75% warp lane utilization — the kernel
        // launch overhead exceeds any bandwidth savings on such small GEMVs.
        // Rejects: Q4_K ne0=256 (nb_sub=8), ne0=512 (nb_sub=16). Accepts ne0>=768 (nb_sub>=24).
        // Rejects: Q6_K ne0=256 (nb_sub=16). Accepts ne0>=512 (nb_sub>=32).
        const int64_t sub_blocks_per_sb =
            (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K
                || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S
                || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS
                || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S
                || type == GGML_TYPE_IQ1_M) ? 8 : 16;
        const int64_t nb_sub = (ne0 / 256) * sub_blocks_per_sb;
        if (nb_sub < 24) return false;
        return true;
    }
    // Q4_0/Q8_0/Q5_0/Q5_1/MXFP4: 32-element blocks.
    return ne0 % 32 == 0 && ne0 >= 128;
}

// One-shot debug log — replaces repeated static bool + GGML_LOG_DEBUG boilerplate.
// Usage: TMAC_LOG_ONCE("[TMAC] message %s\n", arg)
#define TMAC_LOG_ONCE(...) do { \
    static bool _tmac_logged = false; \
    if (!_tmac_logged) { \
        GGML_LOG_DEBUG(__VA_ARGS__); \
        _tmac_logged = true; \
    } \
} while (0)

// ── Experiment safeguard ──────────────────────────────────────────────
// P14 ran for weeks on Q4_K_M models without realizing the experiment
// code only existed in the !is_sub_block_parallel branch (Q4_0/Q8_0/Q5_1).
// Result: NULL EXPERIMENT — measured noise, reported it as signal.
//
// Use TMAC_EXPERIMENT_LOG in host-side dispatch to confirm activation.
// Use static_assert in kernel templates to constrain quant type scope.
//
// Host-side (in ggml_cuda_tmac_*() wrappers or tmac_dispatch_*()):
//   TMAC_EXPERIMENT_LOG("cache_policy", type);
//
// Kernel-side (inside #ifdef EXPERIMENT blocks):
//   static_assert(QType == GGML_TYPE_Q4_0 || QType == GGML_TYPE_Q8_0,
//       "Experiment X only implemented for Q4_0/Q8_0");
// ──────────────────────────────────────────────────────────────────────
#define TMAC_EXPERIMENT_LOG(name, type) \
    TMAC_LOG_ONCE("[TMAC] Experiment '%s' active for %s\n", (name), ggml_type_name(type))

// One-time warning when MoE expert layers fall back to stock due to alignment/warp-efficiency.
// Fires exactly once per process (thread-safe via static bool). Tells users which guard
// rejected the expert tensors so they know dense/shared layers still use T-MAC.
static inline void tmac_warn_expert_fallback(ggml_type type, int64_t ne0) {
    static bool warned = false;
    if (warned) return;
    warned = true;
    const char * reason = "unknown";
    if (ne0 % 256 != 0 || ne0 < 256) {
        reason = "alignment (ne0 % 256 != 0)";
    } else {
        // Must be warp efficiency guard
        reason = "warp efficiency (nb_sub < 24)";
    }
    GGML_LOG_WARN("[TMAC] MoE expert layers using stock kernel: %s "
                  "(type=%s, ne0=%lld). Dense/shared layers still use T-MAC.\n",
                  reason, ggml_type_name(type), (long long)ne0);
}

// T-MAC v7.1: unfused GEMV for Q4_0 × F32 (v7.5: + MoE expert dispatch)
void ggml_cuda_tmac_q4_0_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,        // MoE expert routing (nullptr = non-MoE)
    int64_t expert_stride = 0,            // bytes between experts in src0
    int n_expert_used = 1,                // number of expert slots to dispatch
    int64_t src1_expert_stride = 0,       // floats between expert activations in src1
    const float * bias = nullptr          // bias vector for fused MUL_MAT+ADD (v9.2)
);

// T-MAC: unfused GEMV for Q8_0 x F32 (same Warp-per-Row architecture)
void ggml_cuda_tmac_q8_0_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: unfused GEMV for Q5_1 x F32 (same Warp-per-Row architecture)
void ggml_cuda_tmac_q5_1_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC v7.2: fused SwiGLU GEMV — dual projection + GLU activation in one pass
//             (v7.5: + MoE expert dispatch)
void ggml_cuda_tmac_q4_0_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

void ggml_cuda_tmac_q8_0_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

void ggml_cuda_tmac_q5_1_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for Q5_0 x F32 (symmetric 5-bit, same architecture as Q5_1)
void ggml_cuda_tmac_q5_0_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

void ggml_cuda_tmac_q5_0_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for Q4_K x F32 (256-element super-blocks)
void ggml_cuda_tmac_q4_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for Q4_K x F32 (256-element super-blocks)
void ggml_cuda_tmac_q4_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for Q5_K x F32 (256-element super-blocks, 5-bit K-quants)
void ggml_cuda_tmac_q5_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for Q5_K x F32 (256-element super-blocks, 5-bit K-quants)
void ggml_cuda_tmac_q5_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for Q6_K x F32 (256-element super-blocks, 6-bit quants)
void ggml_cuda_tmac_q6_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for Q6_K x F32 (256-element super-blocks, 6-bit quants)
void ggml_cuda_tmac_q6_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ4_XS x F32 (256-element super-blocks, LUT-indexed 4-bit quants)
void ggml_cuda_tmac_iq4_xs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for IQ4_XS x F32 (256-element super-blocks, LUT-indexed)
void ggml_cuda_tmac_iq4_xs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ3_S x F32 (256-element super-blocks, LUT-indexed 3-bit + signs)
void ggml_cuda_tmac_iq3_s_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for IQ3_S x F32 (256-element super-blocks, LUT-indexed 3-bit)
void ggml_cuda_tmac_iq3_s_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ2_XXS x F32 (256-element super-blocks, dual-LUT 2-bit)
void ggml_cuda_tmac_iq2_xxs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for IQ2_XXS x F32 (256-element super-blocks, dual-LUT 2-bit)
void ggml_cuda_tmac_iq2_xxs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ3_XXS x F32 (3-bit LUT-indexed, 98 bytes/block, 256 elements)
void ggml_cuda_tmac_iq3_xxs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);
void ggml_cuda_tmac_iq3_xxs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ2_XS x F32 (2.3-bit LUT-indexed, 74 bytes/block, 256 elements)
void ggml_cuda_tmac_iq2_xs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);
void ggml_cuda_tmac_iq2_xs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ2_S x F32 (2.5-bit LUT-indexed, 82 bytes/block, 256 elements)
void ggml_cuda_tmac_iq2_s_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);
void ggml_cuda_tmac_iq2_s_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ1_M x F32 (1.75-bit LUT-indexed, 56 bytes/block, 256 elements)
void ggml_cuda_tmac_iq1_m_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1,
    int64_t src1_expert_stride = 0, const float * bias = nullptr
);
void ggml_cuda_tmac_iq1_m_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for MXFP4 x F32 (OCP Microscaling 4-bit, 17 bytes/block)
void ggml_cuda_tmac_mxfp4_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for MXFP4 x F32 (OCP Microscaling 4-bit)
void ggml_cuda_tmac_mxfp4_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for IQ4_NL x F32 (32-element blocks, non-linear LUT dequant)
void ggml_cuda_tmac_iq4_nl_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for IQ4_NL x F32 (32-element blocks, non-linear LUT)
void ggml_cuda_tmac_iq4_nl_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// T-MAC: unfused GEMV for Q3_K x F32 (256-element super-blocks, 3-bit K-quants)
void ggml_cuda_tmac_q3_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
);

// T-MAC: fused SwiGLU GEMV for Q3_K x F32 (256-element super-blocks, 3-bit K-quants)
void ggml_cuda_tmac_q3_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr, int64_t expert_stride = 0, int n_expert_used = 1
);

// ═══════════════════════════════════════════════════════════════════════
// Active Ratio Counter — Silent Failure Detection
// ═══════════════════════════════════════════════════════════════════════
// Tracks how many quantized GEMV ops go through T-MAC vs stock fallback.
// Counts ALL quantized GEMV batch=1 ops regardless of type (including Q2_K, etc.).
// Coverage < 50%: model uses unsupported quant types.
// Coverage < 90%: ne0 alignment rejection on supported types.
// Reported via GGML_LOG at process exit (atexit). Zero overhead (~1ns per GEMV op).

struct tmac_active_ratio {
    int64_t tmac_ops;       // ops dispatched through T-MAC
    int64_t total_ops;      // stock fallback GEMV ops (ALL quantized types, not just T-MAC-eligible)
    int64_t tmac_elements;  // FLOP-weighted: M*K elements processed by T-MAC
    int64_t total_elements; // FLOP-weighted: M*K elements processed by stock fallback
    int64_t alias_copies;   // fused ops where src1/dst aliasing required temp copy
    int64_t prefill_misses; // misses from prefill (ne2>1 batched MoE) — not a coverage gap
    int64_t expert_tmac_elements;  // FLOP-weighted: expert MoE elements via T-MAC
    int64_t expert_stock_elements; // FLOP-weighted: expert MoE elements via stock
};

static inline tmac_active_ratio & tmac_get_counters() {
    static tmac_active_ratio counters = {};
    return counters;
}

static inline void tmac_count_hit(int64_t elements = 1)  {
    auto & c = tmac_get_counters();
    c.tmac_ops++;
    c.tmac_elements += elements;
}
static inline void tmac_count_miss(int64_t elements = 1) {
    auto & c = tmac_get_counters();
    c.total_ops++;
    c.total_elements += elements;
}
static inline void tmac_count_miss_prefill() {
    tmac_get_counters().prefill_misses++;
}
// Expert-specific counters for dense/expert split in Active Ratio report.
// Called at MoE dispatch sites (Site 3/6, Site 4/6) to track expert layer coverage separately.
static inline void tmac_count_expert_hit(int64_t elements) {
    tmac_get_counters().expert_tmac_elements += elements;
}
static inline void tmac_count_expert_miss(int64_t elements) {
    tmac_get_counters().expert_stock_elements += elements;
}
// Diagnostic: log WHY a miss happened (first occurrence per unique type+ne0 pair).
// Enable via GGML_TMAC_LOG_MISSES=1 env var. Negligible overhead when disabled (static branch).
static inline void tmac_log_miss(ggml_type type, int64_t ne0, const char * site) {
    // thread_local: multi-GPU dispatch uses multiple host threads (one per device).
    // Static shared state would be a data race / UB. Each thread gets its own dedup set.
    thread_local bool checked = false, enabled = false;
    if (!checked) { enabled = (getenv("GGML_TMAC_LOG_MISSES") != nullptr); checked = true; }
    if (!enabled) return;

    // Deduplicate: only log first miss per (type, ne0, site) triple
    struct miss_key { ggml_type t; int64_t n; const char * s; };
    thread_local miss_key seen[64];
    thread_local int n_seen = 0;
    for (int i = 0; i < n_seen; i++) {
        if (seen[i].t == type && seen[i].n == ne0 && seen[i].s == site) return;
    }
    if (n_seen < 64) { seen[n_seen++] = {type, ne0, site}; }

    const char * reason = "unknown";
    if (!ggml_cuda_tmac_is_supported_type(type)) {
        reason = "unsupported_type";
    } else if (!ggml_cuda_tmac_can_dispatch(type, ne0)) {
        // Determine which sub-guard failed
        bool is_256 = (type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K
            || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S
            || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS
            || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S
            || type == GGML_TYPE_IQ1_M);
        if (is_256) {
            if (ne0 % 256 != 0 || ne0 < 256) reason = "ne0_align_256";
            else {
                int sb = (type==GGML_TYPE_Q4_K||type==GGML_TYPE_Q5_K
                    ||type==GGML_TYPE_IQ4_XS||type==GGML_TYPE_IQ3_S
                    ||type==GGML_TYPE_IQ2_XXS||type==GGML_TYPE_IQ3_XXS
                    ||type==GGML_TYPE_IQ2_XS||type==GGML_TYPE_IQ2_S
                    ||type==GGML_TYPE_IQ1_M) ? 8 : 16;
                int64_t nb_sub = (ne0/256)*sb;
                if (nb_sub < 24) reason = "nb_sub_warp_eff";
                else reason = "can_dispatch_other";
            }
        } else {
            if (ne0 % 32 != 0 || ne0 < 128) reason = "ne0_align_32";
            else reason = "can_dispatch_other";
        }
    } else {
        reason = "dispatch_guard (ne1/ne2/fusion)";
    }
    GGML_LOG_WARN("[TMAC MISS] site=%s type=%s ne0=%lld reason=%s\n",
        site, ggml_type_name(type), (long long)ne0, reason);
}
static inline void tmac_count_alias_copy() {
    tmac_get_counters().alias_copies++;
}

static inline void tmac_report_active_ratio() {
    const auto & c = tmac_get_counters();
    const int64_t total = c.tmac_ops + c.total_ops;
    if (total == 0) return;

    // Generation-only ratio: subtract prefill misses (ne2>1 MoE batch ops)
    // which are expected and harmless. Prefill is matrix-matrix, not GEMV.
    const int64_t gen_misses = c.total_ops - c.prefill_misses;
    const int64_t gen_total  = c.tmac_ops + gen_misses;
    const double gen_ratio = gen_total > 0 ? 100.0 * c.tmac_ops / gen_total : 100.0;

    const double ratio = 100.0 * c.tmac_ops / total;
    if (gen_ratio >= 99.0 && ratio < 90.0) {
        // High generation coverage but low overall due to prefill misses — informational
        GGML_LOG_INFO("[TMAC] Active Ratio: %.1f%% generation, %.1f%% overall (%lld/%lld ops, %lld prefill-only)\n",
            gen_ratio, ratio, (long long)c.tmac_ops, (long long)total, (long long)c.prefill_misses);
    } else if (gen_ratio < 50.0) {
        GGML_LOG_WARN("[TMAC] *** GEMV COVERAGE WARNING: %.1f%% (%lld/%lld ops) ***\n",
            gen_ratio, (long long)c.tmac_ops, (long long)gen_total);
        GGML_LOG_WARN("[TMAC] T-MAC is NOT accelerating most GEMV ops for this model.\n");
        GGML_LOG_WARN("[TMAC] Likely cause: unsupported quant type (Q2_K, Q3_K, etc.)\n");
        GGML_LOG_WARN("[TMAC]   or ne0 alignment (ne0 %% 256 != 0 for Q4_K/Q6_K/IQ types).\n");
        GGML_LOG_WARN("[TMAC] Run: scripts/model-card.sh <model.gguf> for tensor type diagnosis.\n");
    } else if (gen_ratio < 90.0) {
        GGML_LOG_WARN("[TMAC] *** ACTIVE RATIO WARNING: %.1f%% (%lld/%lld ops) ***\n",
            gen_ratio, (long long)c.tmac_ops, (long long)gen_total);
        GGML_LOG_WARN("[TMAC] Some GEMV ops falling through to stock kernels.\n");
        GGML_LOG_WARN("[TMAC] Likely cause: ne0 alignment (ne0 %% 256 != 0 for Q4_K/Q6_K/IQ).\n");
        GGML_LOG_WARN("[TMAC] Run: scripts/model-card.sh <model.gguf> for diagnosis.\n");
    } else {
        GGML_LOG_INFO("[TMAC] Active Ratio: %.1f%% (%lld/%lld quantized GEMV ops)\n",
            gen_ratio, (long long)c.tmac_ops, (long long)gen_total);
    }
    // FLOP-weighted ratio: reflects actual compute coverage, not just call count.
    // Addresses finding that 87.4% op ratio masked 16% FLOP coverage on GPT-OSS 120B.
    const int64_t total_elem = c.tmac_elements + c.total_elements;
    if (total_elem > 0) {
        const double elem_ratio = 100.0 * c.tmac_elements / total_elem;
        // Show dense/expert split when model has MoE expert layers
        const int64_t expert_total = c.expert_tmac_elements + c.expert_stock_elements;
        if (expert_total > 0) {
            const double expert_ratio = 100.0 * c.expert_tmac_elements / expert_total;
            const int64_t dense_tmac  = c.tmac_elements - c.expert_tmac_elements;
            const int64_t dense_total = total_elem - expert_total;
            const double dense_ratio  = dense_total > 0 ? 100.0 * dense_tmac / dense_total : 100.0;
            GGML_LOG_INFO("[TMAC] Compute Coverage: %.1f%% (dense: %.1f%%, expert: %.1f%%)\n",
                elem_ratio, dense_ratio, expert_ratio);
            // Effective T-MAC Coverage (ETC): weighted by dense/expert compute fractions.
            // Predicts speedup direction: >70% = full benefit, 40-70% = attenuated, <30% = NS/regression.
            const double dense_frac  = dense_total > 0 ? (double)dense_total / total_elem : 0.0;
            const double expert_frac = (double)expert_total / total_elem;
            const double etc = dense_frac * (dense_ratio / 100.0) + expert_frac * (expert_ratio / 100.0);
            GGML_LOG_INFO("[TMAC] Effective T-MAC Coverage (ETC): %.0f%% (dense %.0f%% × %.1f%% + expert %.0f%% × %.1f%%)\n",
                etc * 100.0, dense_frac * 100.0, dense_ratio, expert_frac * 100.0, expert_ratio);
        } else {
            GGML_LOG_INFO("[TMAC] Compute Coverage: %.1f%% (by elements, FLOP-weighted)\n", elem_ratio);
        }
    }
    // Aliasing stats: how many fused ops needed src1→temp copy.
    // High alias_copies count is normal for models like Llama 4 where the graph
    // allocator reuses src1 buffer for dst on most fused GLU layers.
    if (c.alias_copies > 0) {
        GGML_LOG_INFO("[TMAC] Alias copies: %lld fused ops (src1/dst pool-copy, sub-us each)\n",
            (long long)c.alias_copies);
    }
}

static inline void tmac_reset_counters() {
    auto & c = tmac_get_counters();
    c.tmac_ops = 0;
    c.total_ops = 0;
    c.tmac_elements = 0;
    c.total_elements = 0;
    c.alias_copies = 0;
    c.prefill_misses = 0;
    c.expert_tmac_elements = 0;
    c.expert_stock_elements = 0;
}

// Auto-report at process exit (registered once on first dispatch)
static inline void tmac_register_atexit_report() {
    static bool registered = false;
    if (!registered) {
        atexit(tmac_report_active_ratio);
        registered = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Central type dispatch — new quant type = add 2 case lines + traits
// ═══════════════════════════════════════════════════════════════════════

// Central type dispatch for all unfused T-MAC GEMV paths.
// Handles both non-MoE (ids=nullptr) and MoE (ids!=nullptr).
static inline void tmac_dispatch_simple(
    ggml_backend_cuda_context & ctx,
    ggml_type type,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1,
    int64_t src1_expert_stride = 0,
    const float * bias = nullptr
) {
    tmac_count_hit(ne0 * ne00);
    tmac_register_atexit_report();
    switch (type) {
        case GGML_TYPE_Q4_0:
            ggml_cuda_tmac_q4_0_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_Q8_0:
            ggml_cuda_tmac_q8_0_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_Q5_1:
            ggml_cuda_tmac_q5_1_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_Q5_0:
            ggml_cuda_tmac_q5_0_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_Q4_K:
            ggml_cuda_tmac_q4_K_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_Q5_K:
            ggml_cuda_tmac_q5_K_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_Q6_K:
            ggml_cuda_tmac_q6_K_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ4_XS:
            ggml_cuda_tmac_iq4_xs_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ3_S:
            ggml_cuda_tmac_iq3_s_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ2_XXS:
            ggml_cuda_tmac_iq2_xxs_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ3_XXS:
            ggml_cuda_tmac_iq3_xxs_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ2_XS:
            ggml_cuda_tmac_iq2_xs_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ2_S:
            ggml_cuda_tmac_iq2_s_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ1_M:
            ggml_cuda_tmac_iq1_m_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_MXFP4:
            ggml_cuda_tmac_mxfp4_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_Q3_K:
            ggml_cuda_tmac_q3_K_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        case GGML_TYPE_IQ4_NL:
            ggml_cuda_tmac_iq4_nl_simple(ctx, src0, src1, dst, ne00, ne0, stream,
                ids, expert_stride, n_expert_used, src1_expert_stride, bias); break;
        default:
            GGML_ABORT("T-MAC: unsupported type %s", ggml_type_name(type));
    }
}

// Central type dispatch for all fused SwiGLU T-MAC GEMV paths.
static inline void tmac_dispatch_fused(
    ggml_backend_cuda_context & ctx,
    ggml_type type,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids = nullptr,
    int64_t expert_stride = 0,
    int n_expert_used = 1
) {
    tmac_count_hit(ne0 * ne00 * 2);  // fused: UP + GATE = 2x element count
    tmac_register_atexit_report();
    switch (type) {
        case GGML_TYPE_Q4_0:
            ggml_cuda_tmac_q4_0_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_Q8_0:
            ggml_cuda_tmac_q8_0_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_Q5_1:
            ggml_cuda_tmac_q5_1_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_Q5_0:
            ggml_cuda_tmac_q5_0_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_Q4_K:
            ggml_cuda_tmac_q4_K_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_Q5_K:
            ggml_cuda_tmac_q5_K_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_Q6_K:
            ggml_cuda_tmac_q6_K_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ4_XS:
            ggml_cuda_tmac_iq4_xs_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ3_S:
            ggml_cuda_tmac_iq3_s_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ2_XXS:
            ggml_cuda_tmac_iq2_xxs_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ3_XXS:
            ggml_cuda_tmac_iq3_xxs_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ2_XS:
            ggml_cuda_tmac_iq2_xs_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ2_S:
            ggml_cuda_tmac_iq2_s_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ1_M:
            ggml_cuda_tmac_iq1_m_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_MXFP4:
            ggml_cuda_tmac_mxfp4_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_Q3_K:
            ggml_cuda_tmac_q3_K_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        case GGML_TYPE_IQ4_NL:
            ggml_cuda_tmac_iq4_nl_fused(ctx, src0_up, src0_gate, src1, dst,
                ne00, ne0, glu_op, stream, ids, expert_stride, n_expert_used); break;
        default:
            GGML_ABORT("T-MAC fused: unsupported type %s", ggml_type_name(type));
    }
}

#else // !GGML_HIP_TMAC — stubs so dispatch sites compile (can_dispatch returns false → never called)

[[maybe_unused]] static inline bool ggml_cuda_tmac_enabled()                            { return false; }
[[maybe_unused]] static inline bool ggml_cuda_tmac_is_supported_type(ggml_type)         { return false; }
[[maybe_unused]] static inline bool ggml_cuda_tmac_can_dispatch(ggml_type, int64_t)     { return false; }

// Active Ratio stubs — never called (enabled() returns false), but compiler needs declarations
[[maybe_unused]] static inline void tmac_count_hit(int64_t = 1)                         { }
[[maybe_unused]] static inline void tmac_count_miss(int64_t = 1)                        { }
[[maybe_unused]] static inline void tmac_count_miss_prefill()                           { }
[[maybe_unused]] static inline void tmac_log_miss(ggml_type, int64_t, const char *)     { }
[[maybe_unused]] static inline void tmac_count_alias_copy()                             { }
[[maybe_unused]] static inline void tmac_count_expert_hit(int64_t)                      { }
[[maybe_unused]] static inline void tmac_count_expert_miss(int64_t)                     { }
[[maybe_unused]] static inline void tmac_warn_expert_fallback(ggml_type, int64_t)       { }
[[maybe_unused]] static inline void tmac_register_atexit_report()                       { }

#define TMAC_LOG_ONCE(...) ((void)0)

// Dispatch stubs — needed because C++ resolves names even in dead branches.
// can_dispatch() returning false guarantees these are never called at runtime.
[[maybe_unused]] static inline void tmac_dispatch_simple(
    ggml_backend_cuda_context &, ggml_type,
    const void *, const float *, float *,
    int64_t, int64_t, cudaStream_t,
    const int32_t * = nullptr, int64_t = 0, int = 1, int64_t = 0,
    const float * = nullptr) {
    GGML_ABORT("T-MAC not enabled (GGML_HIP_TMAC=OFF)");
}

[[maybe_unused]] static inline void tmac_dispatch_fused(
    ggml_backend_cuda_context &, ggml_type,
    const void *, const void *,
    const float *, float *,
    int64_t, int64_t,
    ggml_glu_op, cudaStream_t,
    const int32_t * = nullptr, int64_t = 0, int = 1) {
    GGML_ABORT("T-MAC not enabled (GGML_HIP_TMAC=OFF)");
}

#endif // GGML_HIP_TMAC
