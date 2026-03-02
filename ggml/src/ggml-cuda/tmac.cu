#include "common.cuh"
#include "tmac.cuh"

// P8 Q6_K weight repacking — REJECTED (2026-02-23).
// 0% gain on 22B, +1.1% noise on 1B, +2.5GB VRAM cost for dual buffers.
// Root cause: kernel is memory-bound (VALU/MEM=0.37x), repacking doesn't reduce
// bytes fetched. Compiler already generates efficient interleaved decode via v_bfe_u32.

// ═════════════════════════════════════════════════════════════════════
// Block struct definitions (packed, matching ggml quantization layout)
//
// Why internal structs instead of upstream block_q4_0 etc. from ggml-common.h?
// T-MAC kernels use byte-level pointer arithmetic to access fields within
// weight blocks (e.g., casting row pointers to block_q4_0_internal* and
// indexing by block number). This requires guaranteed packed layout with no
// padding — __attribute__((packed)) enforces this. The upstream types in
// ggml-common.h may have different alignment on different platforms. The
// static_asserts below verify size parity as a safety net.
// ═════════════════════════════════════════════════════════════════════

// Q4_0 block: 2 bytes scale (half) + 16 bytes quantized (32 x 4-bit weights)
struct __attribute__((packed)) block_q4_0_internal {
    half d;
    uint8_t qs[16];
};

// Q8_0 block: 2 bytes scale (half) + 32 bytes quantized (32 x int8)
struct __attribute__((packed)) block_q8_0_internal {
    half d;
    int8_t qs[32];
};

// Q5_1 block: 2 bytes d (half) + 2 bytes m (half) + 4 bytes qh + 16 bytes qs = 24 bytes
struct __attribute__((packed)) block_q5_1_internal {
    half d;
    half m;
    uint8_t qh[4];  // 32 high bits packed
    uint8_t qs[16];  // 32 x 4-bit low nibbles (same packing as Q4_0)
};

// Q5_0 block: 2 bytes d (half) + 4 bytes qh + 16 bytes qs = 22 bytes
// Symmetric variant of Q5_1: dequant = d * (val - 16), no min term.
// Same qh/qs bit packing as Q5_1, only the scale formula differs.
struct __attribute__((packed)) block_q5_0_internal {
    half d;
    uint8_t qh[4];  // 32 high bits packed
    uint8_t qs[16];  // 32 x 4-bit low nibbles (same packing as Q5_1)
};

// Q4_K super-block: 256 elements = 8 sub-blocks of 32, asymmetric dequant.
// Layout: d (2B) + dmin (2B) + scales[12] (packed 6-bit scale+min per sub-block) + qs[128] (4-bit quants)
struct __attribute__((packed)) block_q4_K_internal {
    half d;              // super-block scale for quantized scales
    half dmin;           // super-block scale for quantized mins
    uint8_t scales[12];  // packed 6-bit sub-block scales and mins
    uint8_t qs[128];     // 256 × 4-bit quants (low nibble = even sub-block, high = odd)
};

// Q5_K super-block: 256 elements = 8 sub-blocks of 32, asymmetric dequant.
// Layout identical to Q4_K + 32 bytes for high bits (qh). Each element is 5-bit:
// value = (qs nibble) + (qh bit << 4), giving range 0-31 instead of Q4_K's 0-15.
// qh layout is interleaved: bit j of qh[l] = high bit for sub-block j, element l.
struct __attribute__((packed)) block_q5_K_internal {
    half d;              // super-block scale for quantized scales
    half dmin;           // super-block scale for quantized mins
    uint8_t scales[12];  // packed 6-bit sub-block scales and mins (SAME format as Q4_K)
    uint8_t qh[32];      // high bits: 1 bit per element, interleaved across sub-blocks
    uint8_t qs[128];     // 256 × 4-bit quants (low nibble = even sub-block, high = odd)
};

// Q6_K super-block: 256 elements = 16 sub-blocks of 16, 6-bit quants with 8-bit scales.
// Layout: ql[128] (lower 4 bits, 2 per byte) + qh[64] (upper 2 bits, 4 per byte)
//       + scales[16] (int8, one per sub-block) + d (half, super-block scale)
struct __attribute__((packed)) block_q6_K_internal {
    uint8_t ql[128];    // lower 4 bits (2 elements/byte: low nib + high nib)
    uint8_t qh[64];     // upper 2 bits (4 elements/byte, 2 bits each)
    int8_t  scales[16]; // one 8-bit scale per 16-element sub-block
    half    d;          // super-block scale
};

// IQ4_XS super-block: 256 elements = 8 sub-blocks of 32, 4-bit LUT-indexed quants.
// Layout: d (fp16, 2B) + scales_h (uint16, 2B) + scales_l[4] (4B) + qs[128] (128B) = 136B
// Each 4-bit value is an index into kvalues_iq4nl[16] — non-linear quantization.
// Scales: 6-bit per sub-block, packed as 4-bit low (scales_l) + 2-bit high (scales_h).
struct __attribute__((packed)) block_iq4_xs_internal {
    half     d;            // super-block scale
    uint16_t scales_h;     // high 2 bits of 8 sub-block scales (2 bits each, 16 bits total)
    uint8_t  scales_l[4];  // low 4 bits of 8 sub-block scales (4 bits each, packed 2 per byte)
    uint8_t  qs[128];      // 256 × 4-bit LUT indices (2 per byte)
};

// IQ3_S super-block: 256 elements = 8 sub-blocks of 32, 3-bit LUT-indexed quants.
// Layout: d (fp16, 2B) + qs[64] (64B) + qh[8] (8B) + signs[32] (32B) + scales[4] (4B) = 110B
// Each qs byte → grid index (+ 1 high bit from qh) → 4 packed uint8 values from iq3s_grid[512].
// Signs: 1 bit per element, negates the grid-decoded value.
struct __attribute__((packed)) block_iq3_s_internal {
    half    d;           // super-block scale
    uint8_t qs[64];      // lower 8 bits of 9-bit grid indices (8 per sub-block, each decodes to 4 values)
    uint8_t qh[8];       // high bit of 9-bit grid indices (8 bits per sub-block, 1 per grid lookup)
    uint8_t signs[32];   // sign bits for each element (256 bits = 32B)
    uint8_t scales[4];   // 4-bit sub-block scales (packed 2 per byte, 8 sub-groups)
};

// IQ2_XXS super-block: 256 elements, extreme 2-bit compression.
// Layout: d (fp16, 2B) + qs[32] (64B as uint16_t[32]) = 66B
// Each group of 8 elements is encoded as: 2 uint16 for indices + embedded signs.
struct __attribute__((packed)) block_iq2_xxs_internal {
    half     d;         // super-block scale
    uint16_t qs[32];    // packed 2-bit indices with embedded scale/sign info
};

// IQ3_XXS super-block: 256 elements, 3-bit LUT-indexed with embedded scales/signs.
// Layout: d (fp16, 2B) + qs[96] (96B: first 64B = grid indices, last 32B = scales_and_signs) = 98B
// 8 sub-blocks of 32 elements. Each sub-block: 8 grid lookups × 4 values = 32 elements.
// Signs via ksigns_iq2xs[7-bit key], scales embedded in upper 4 bits of aux32.
struct __attribute__((packed)) block_iq3_xxs_internal {
    half    d;           // super-block scale
    uint8_t qs[96];      // first 64B: 8-bit grid indices; last 32B: packed scales + signs
};

// IQ2_XS super-block: 256 elements, 2.3125 bpw.
// Layout: d (fp16, 2B) + qs[32] (64B as uint16_t[32]) + scales[8] (8B) = 74B
// 8 sub-blocks of 32 elements. 9-bit grid index + 7-bit sign key packed per uint16_t.
// Separate 4-bit per-sub-block scales (packed 2 per byte).
struct __attribute__((packed)) block_iq2_xs_internal {
    half     d;           // super-block scale
    uint16_t qs[32];      // 9-bit grid index (lower) + 7-bit sign key (upper)
    uint8_t  scales[8];   // 4-bit scales for 8 sub-blocks (packed 2 per byte)
};

// IQ2_S super-block: 256 elements, 2.5625 bpw.
// Layout: d (fp16, 2B) + qs[64] (64B) + qh[8] (8B) + signs[32] (= qs+64, interleaved) + scales[8] = 82B
// 8 sub-blocks of 32 elements. 10-bit grid index from qs[8-bit] + qh[2-bit].
// Separate sign bytes (1 bit per element in groups of 8). Separate 4-bit scales.
struct __attribute__((packed)) block_iq2_s_internal {
    half    d;            // super-block scale
    uint8_t qs[64];       // lower 8 bits of 10-bit grid indices
    uint8_t qh[8];        // high 2 bits of 10-bit grid indices (2 bits per sub-block-pair)
    uint8_t scales[8];    // 4-bit scales for 8 sub-blocks (packed 2 per byte)
};

// Compile-time check: internal block structs must match upstream layout (ggml-common.h)
static_assert(sizeof(block_q4_0_internal) == sizeof(block_q4_0),
    "T-MAC Q4_0 block size mismatch with upstream");
static_assert(sizeof(block_q8_0_internal) == sizeof(block_q8_0),
    "T-MAC Q8_0 block size mismatch with upstream");
static_assert(sizeof(block_q5_1_internal) == sizeof(block_q5_1),
    "T-MAC Q5_1 block size mismatch with upstream");
static_assert(sizeof(block_q5_0_internal) == sizeof(block_q5_0),
    "T-MAC Q5_0 block size mismatch with upstream");
static_assert(sizeof(block_q4_K_internal) == sizeof(block_q4_K),
    "T-MAC Q4_K block size mismatch with upstream");
static_assert(sizeof(block_q5_K_internal) == sizeof(block_q5_K),
    "T-MAC Q5_K block size mismatch with upstream");
static_assert(sizeof(block_q6_K_internal) == sizeof(block_q6_K),
    "T-MAC Q6_K block size mismatch with upstream");
static_assert(sizeof(block_iq4_xs_internal) == sizeof(block_iq4_xs),
    "T-MAC IQ4_XS block size mismatch with upstream");
static_assert(sizeof(block_iq3_s_internal) == sizeof(block_iq3_s),
    "T-MAC IQ3_S block size mismatch with upstream");
static_assert(sizeof(block_iq2_xxs_internal) == sizeof(block_iq2_xxs),
    "T-MAC IQ2_XXS block size mismatch with upstream");
static_assert(sizeof(block_iq3_xxs_internal) == sizeof(block_iq3_xxs),
    "T-MAC IQ3_XXS block size mismatch with upstream");
static_assert(sizeof(block_iq2_xs_internal) == sizeof(block_iq2_xs),
    "T-MAC IQ2_XS block size mismatch with upstream");
static_assert(sizeof(block_iq2_s_internal) == sizeof(block_iq2_s),
    "T-MAC IQ2_S block size mismatch with upstream");

// IQ1_M: 256-element super-blocks, 1.75 bpw, iq1s_grid_gpu[2048] LUT.
// Unique: no explicit `d` field — super-block scale encoded in top 4 bits of scales.
struct __attribute__((packed)) block_iq1_m_internal {
    uint8_t qs[QK_K/8];      // 32 bytes — lower 8 bits of 11-bit grid indices
    uint8_t qh[QK_K/16];     // 16 bytes — upper 3 bits + 1 delta-shift bit (4 bits per group)
    uint8_t scales[QK_K/32]; // 8 bytes — packed 3-bit sub-scales + embedded fp16 super-block scale
};
static_assert(sizeof(block_iq1_m_internal) == sizeof(block_iq1_m),
    "T-MAC IQ1_M block size mismatch with upstream");

// MXFP4 block: 1 byte E8M0 exponent + 16 bytes (32 × 4-bit nibbles) = 17 bytes.
// OCP Microscaling Format: dequant = kvalues_mxfp4[nibble] × 2^(e-127) × 0.5
// Same interleaved nibble layout as Q4_0: low nibble = element j, high = element j+16.
struct __attribute__((packed)) block_mxfp4_internal {
    uint8_t e;          // E8M0 scale exponent (shared across 32 elements)
    uint8_t qs[16];     // 32 × 4-bit nibbles (interleaved: low=0-15, high=16-31)
};
static_assert(sizeof(block_mxfp4_internal) == sizeof(block_mxfp4),
    "T-MAC MXFP4 block size mismatch with upstream");

// Q3_K super-block: 256 elements = 16 sub-blocks of 16, 3-bit quants (2-bit qs + 1-bit hmask).
// Symmetric with zero-point: val = qs_2bit - (hmask_clear ? 4 : 0), range -4 to 3.
// Scale: 6-bit signed (scale - 32), same 12-byte packing as Q4_K but no min term.
struct __attribute__((packed)) block_q3_K_internal {
    uint8_t hmask[32];   // high bit mask: 1 per element, hmask[elem%32] bit (elem/32)
    uint8_t qs[64];      // low 2 bits: 4 per byte, packed in 128-element halves
    uint8_t scales[12];  // 6-bit packed scales (16 values, no mins)
    half d;              // super-block scale
};
static_assert(sizeof(block_q3_K_internal) == sizeof(block_q3_K),
    "T-MAC Q3_K block size mismatch with upstream");

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define TMAC_WARPS_PER_BLOCK 8
#define TMAC_ROWS_PER_BLOCK  TMAC_WARPS_PER_BLOCK  // 1 warp per row

// ═════════════════════════════════════════════════════════════════════
// Device utility functions
// ═════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        // HIP maps __shfl_xor_sync → __shfl_xor (mask ignored on AMD).
        // The 0xffffffff mask is for CUDA source compatibility (common.cuh convention).
        val += __shfl_xor_sync(0xffffffff, val, offset, WARP_SIZE);
    }
    return val;
}

// RDNA3: Extract 4-bit nibble from uint32. Single v_bfe_u32 on AMD.
__device__ __forceinline__ uint32_t tmac_bfe4(uint32_t src, uint32_t bit_offset) {
#ifdef __HIP_PLATFORM_AMD__
    return __builtin_amdgcn_ubfe(src, bit_offset, 4);
#else
    return (src >> bit_offset) & 0xF;
#endif
}

// RDNA3: Extract 2-bit field from uint32. Single v_bfe_u32 on AMD.
// Used by Q6_K to extract the upper 2 bits of the 6-bit quantized value.
__device__ __forceinline__ uint32_t tmac_bfe2(uint32_t src, uint32_t bit_offset) {
#ifdef __HIP_PLATFORM_AMD__
    return __builtin_amdgcn_ubfe(src, bit_offset, 2);
#else
    return (src >> bit_offset) & 0x3;
#endif
}

// RDNA3: Extract 1-bit field from uint32. Single v_bfe_u32 on AMD.
// Used by Q5_K to extract the high bit of the 5-bit quantized value.
__device__ __forceinline__ uint32_t tmac_bfe1(uint32_t src, uint32_t bit_offset) {
#ifdef __HIP_PLATFORM_AMD__
    return __builtin_amdgcn_ubfe(src, bit_offset, 1);
#else
    return (src >> bit_offset) & 0x1;
#endif
}

// ═════════════════════════════════════════════════════════════════════
// Q4_K scale decoder — extracts 6-bit scale and min for sub-block j
// Mirrors get_scale_min_k4() from convert.cu: sub-blocks 0-3 use simple
// 6-bit mask, sub-blocks 4-7 use 4+2 bit split across bytes 8-11.
// ═════════════════════════════════════════════════════════════════════

__device__ __forceinline__ void tmac_q4_K_decode_scales(
    int j, const uint8_t * __restrict__ scales, uint8_t & sc, uint8_t & mn
) {
    if (j < 4) {
        sc = scales[j]     & 63;
        mn = scales[j + 4] & 63;
    } else {
        sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
        mn = (scales[j+4] >>  4) | ((scales[j-0] >> 6) << 4);
    }
}

// Q4_K sub-block dot: processes one 32-element sub-block within a 256-element super-block.
// Computes: dot_val = sum(quant[i] * act[i]) and sum_a = sum(act[i])
// where quant[i] is the 4-bit value for element (j*32 + i) within the super-block.
__device__ __forceinline__ void tmac_q4_K_subblock_dot(
    const block_q4_K_internal * __restrict__ blk,
    int j,                          // sub-block index 0-7
    const float4 * __restrict__ av, // 8 x float4 activations for this sub-block
    float & dot_val,
    float & sum_a
) {
    const int pair     = j >> 1;
    const int nib_base = (j & 1) * 4;

    #pragma unroll
    for (int f = 0; f < 8; f++) {
        uint32_t q4;
        memcpy(&q4, blk->qs + 32 * pair + f * 4, sizeof(q4));
        float4 a = av[f];

        dot_val += (float)tmac_bfe4(q4, nib_base)      * a.x;
        dot_val += (float)tmac_bfe4(q4, nib_base + 8)  * a.y;
        dot_val += (float)tmac_bfe4(q4, nib_base + 16) * a.z;
        dot_val += (float)tmac_bfe4(q4, nib_base + 24) * a.w;

        sum_a += a.x + a.y + a.z + a.w;
    }
}

// Q5_K sub-block dot: processes one 32-element sub-block within a 256-element super-block.
// Identical to Q4_K except each 4-bit nibble is extended by 1 high bit from qh.
// qh layout is interleaved: bit j of each qh byte = high bit for sub-block j.
// All 32 qh bytes are read (same as other sub-blocks), but only bit j is extracted.
__device__ __forceinline__ void tmac_q5_K_subblock_dot(
    const block_q5_K_internal * __restrict__ blk,
    int j,                          // sub-block index 0-7
    const float4 * __restrict__ av, // 8 x float4 activations for this sub-block
    float & dot_val,
    float & sum_a
) {
    const int pair     = j >> 1;
    const int nib_base = (j & 1) * 4;

    #pragma unroll
    for (int f = 0; f < 8; f++) {
        uint32_t q4;
        memcpy(&q4, blk->qs + 32 * pair + f * 4, sizeof(q4));
        uint32_t qh4;
        memcpy(&qh4, blk->qh + f * 4, sizeof(qh4));
        float4 a = av[f];

        // 5-bit value = low 4 bits (from qs) + high 1 bit (from qh) * 16
        dot_val += (float)(tmac_bfe4(q4, nib_base)      + (tmac_bfe1(qh4, j)      << 4)) * a.x;
        dot_val += (float)(tmac_bfe4(q4, nib_base + 8)  + (tmac_bfe1(qh4, j + 8)  << 4)) * a.y;
        dot_val += (float)(tmac_bfe4(q4, nib_base + 16) + (tmac_bfe1(qh4, j + 16) << 4)) * a.z;
        dot_val += (float)(tmac_bfe4(q4, nib_base + 24) + (tmac_bfe1(qh4, j + 24) << 4)) * a.w;

        sum_a += a.x + a.y + a.z + a.w;
    }
}

// Q6_K sub-block dot: processes one 16-element sub-block within a 256-element super-block.
// Computes: dot_val = sum(q6[i] * act[i]) and sum_a = sum(act[i])
// where q6[i] is the raw 6-bit value (0-63) for element (sb*16 + i).
// Caller applies: d * scales[sb] * (dot_val - 32 * sum_a)
__device__ __forceinline__ void tmac_q6_K_subblock_dot(
    const block_q6_K_internal * __restrict__ blk,
    int sb,                          // sub-block index 0-15
    const float4 * __restrict__ av,  // 4 x float4 (16 elements)
    float & dot_val,
    float & sum_a
) {
    // ── Original interleaved ql/qh layout ──
    // Q6_K ql/qh layout is interleaved — each half (128 elements) uses:
    //   ql[0..63]: lower 4 bits, 2 rows share bytes (low/high nibble)
    //   qh[0..31]: upper 2 bits, 4 rows share bytes (2 bits each)
    // Sub-block mapping: sb → (half, row, col) determines ql_off, qh_off, qh_sh, nib_base.
    const int half     = sb >> 3;          // 0 for sb 0-7, 1 for sb 8-15
    const int sb_local = sb & 7;
    const int row      = sb_local >> 1;    // 0-3: which pair of ql rows / qh bit pair
    const int col      = sb_local & 1;     // 0-1: which 16-byte column within the row

    const int ql_off   = 64 * half + (row & 1) * 32 + 16 * col;
    const int qh_off   = 32 * half + 16 * col;
    const int qh_sh    = row * 2;          // shift within each qh byte: 0, 2, 4, 6
    const int nib_base = (row >= 2) ? 4 : 0;  // low nibble (rows 0-1) vs high nibble (rows 2-3)

    #pragma unroll
    for (int f = 0; f < 4; f++) {
        uint32_t ql4;
        memcpy(&ql4, blk->ql + ql_off + f * 4, sizeof(ql4));
        uint32_t qh4;
        memcpy(&qh4, blk->qh + qh_off + f * 4, sizeof(qh4));
        float4 a = av[f];

        dot_val += (float)(tmac_bfe4(ql4, nib_base)      | (tmac_bfe2(qh4, qh_sh)      << 4)) * a.x;
        dot_val += (float)(tmac_bfe4(ql4, nib_base + 8)  | (tmac_bfe2(qh4, qh_sh + 8)  << 4)) * a.y;
        dot_val += (float)(tmac_bfe4(ql4, nib_base + 16) | (tmac_bfe2(qh4, qh_sh + 16) << 4)) * a.z;
        dot_val += (float)(tmac_bfe4(ql4, nib_base + 24) | (tmac_bfe2(qh4, qh_sh + 24) << 4)) * a.w;

        sum_a += a.x + a.y + a.z + a.w;
    }
}

// Q3_K scale decoder — extracts signed 6-bit scale for sub-block j (0-15).
// Same 12-byte packing as Q4_K, but 16 individual scales (no sc/mn pairs).
// Returns (unsigned_6bit - 32) for signed scale, matching upstream dequant.
__device__ __forceinline__ int tmac_q3_K_decode_scale(
    int j, const uint8_t * __restrict__ scales
) {
    int us;
    if (j < 4) {
        us = (scales[j] & 0xF) | (((scales[j + 8] >> 0) & 3) << 4);
    } else if (j < 8) {
        us = (scales[j] & 0xF) | (((scales[j + 4] >> 2) & 3) << 4);
    } else if (j < 12) {
        us = (scales[j - 8] >> 4) | (((scales[j] >> 4) & 3) << 4);
    } else {
        us = (scales[j - 8] >> 4) | (((scales[j - 4] >> 6) & 3) << 4);
    }
    return us - 32;
}

// Q3_K sub-block dot: processes one 16-element sub-block within a 256-element super-block.
// 3-bit quants: 2-bit low from qs + 1-bit high from hmask.
// val = qs_2bit - (hmask_clear ? 4 : 0). Range: -4 to 3.
// Computes: dot_val = sum(qs_2bit * act) and sum_a = sum(act where hmask CLEAR).
// Caller applies: d * (scale - 32) * (dot_val - 4 * sum_a).
__device__ __forceinline__ void tmac_q3_K_subblock_dot(
    const block_q3_K_internal * __restrict__ blk,
    int j,
    const float4 * __restrict__ av,
    float & dot_val,
    float & sum_a
) {
    // Sub-block j (0-15) addressing within Q3_K's interleaved 2-bit packing:
    // 256 elements in two 128-element halves, each with 4 groups of 32 at different bit shifts.
    const int half_idx = j / 8;           // 0 or 1 (which 128-element half)
    const int group    = (j % 8) / 2;     // 0-3 (which 32-element group, determines bit shift)
    const int sub      = j % 2;           // 0 or 1 (which 16-element sub-group within 32)
    const int qs_off   = 32 * half_idx + 16 * sub;
    const int shift    = 2 * group;
    const int hm_off   = 16 * sub;
    const uint8_t hm_bit = 1 << (4 * half_idx + group);

    // Process 16 elements as 4 groups of 4 (matching 4 float4 activation vectors)
    for (int k = 0; k < 4; k++) {
        const float4 a = av[k];
        const int base = qs_off + k * 4;
        const int hbase = hm_off + k * 4;

        // Extract 2-bit values from qs
        const float v0 = (float)((blk->qs[base + 0] >> shift) & 3);
        const float v1 = (float)((blk->qs[base + 1] >> shift) & 3);
        const float v2 = (float)((blk->qs[base + 2] >> shift) & 3);
        const float v3 = (float)((blk->qs[base + 3] >> shift) & 3);

        dot_val += v0 * a.x + v1 * a.y + v2 * a.z + v3 * a.w;

        // Accumulate activations where hmask is CLEAR (need -4 correction)
        if (!(blk->hmask[hbase + 0] & hm_bit)) sum_a += a.x;
        if (!(blk->hmask[hbase + 1] & hm_bit)) sum_a += a.y;
        if (!(blk->hmask[hbase + 2] & hm_bit)) sum_a += a.z;
        if (!(blk->hmask[hbase + 3] & hm_bit)) sum_a += a.w;
    }
}

// IQ4_XS scale decoder — extracts 6-bit scale for sub-block j (0-7).
// Low 4 bits from scales_l (packed 2 per byte), high 2 bits from scales_h.
// Scale is offset by 32 (unsigned 6-bit → signed: result - 32).
__device__ __forceinline__ int tmac_iq4_xs_decode_scale(
    int j, const uint8_t * __restrict__ scales_l, uint16_t scales_h
) {
    const int low  = (scales_l[j >> 1] >> ((j & 1) * 4)) & 0x0F;
    const int high = ((scales_h >> (j * 2)) & 0x03) << 4;
    return (low | high) - 32;
}

// IQ4_XS sub-block dot: processes one 32-element sub-block.
// Each nibble is an index into the shared-memory LUT (kvalues_iq4nl).
// Computes: dot_val = sum(lut[nibble[i]] * act[i]) for i in 0..31
// Caller applies: d * scale * dot_val
__device__ __forceinline__ float tmac_iq4_xs_subblock_dot(
    const block_iq4_xs_internal * __restrict__ blk,
    int j,                           // sub-block index 0-7
    const float4 * __restrict__ av,  // 8 x float4 activations for this sub-block
    const int8_t * __restrict__ lut  // shared memory LUT (kvalues_iq4nl, 16 entries)
) {
    float dot = 0.0f;
    const uint8_t * qs = blk->qs + j * 16;  // 16 bytes for this sub-block

    // IQ4_XS nibble layout (SPLIT, not interleaved like Q4_0):
    //   Low nibbles of bytes 0..15 → elements 0..15  (first half of sub-block)
    //   High nibbles of bytes 0..15 → elements 16..31 (second half of sub-block)
    // Activations: av[0..3] = elements 0..15, av[4..7] = elements 16..31

    // First half: low nibbles → elements 0..15
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        const float4 a = av[f];
        dot += (float)lut[qs[f * 4 + 0] & 0xF] * a.x;
        dot += (float)lut[qs[f * 4 + 1] & 0xF] * a.y;
        dot += (float)lut[qs[f * 4 + 2] & 0xF] * a.z;
        dot += (float)lut[qs[f * 4 + 3] & 0xF] * a.w;
    }
    // Second half: high nibbles → elements 16..31
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        const float4 a = av[f + 4];
        dot += (float)lut[qs[f * 4 + 0] >> 4] * a.x;
        dot += (float)lut[qs[f * 4 + 1] >> 4] * a.y;
        dot += (float)lut[qs[f * 4 + 2] >> 4] * a.z;
        dot += (float)lut[qs[f * 4 + 3] >> 4] * a.w;
    }
    return dot;
}

// IQ3_S scale decoder — extracts 4-bit scale for sub-block j (0-7).
// Scale formula: 1 + 2 * raw_scale (maps 0-15 → 1,3,5,...,31)
__device__ __forceinline__ int tmac_iq3_s_decode_scale(
    int j, const uint8_t * __restrict__ scales
) {
    const int raw = (scales[j >> 1] >> ((j & 1) * 4)) & 0x0F;
    return 1 + 2 * raw;
}

// IQ3_S sub-block dot: processes one 32-element sub-block.
// Each element is decoded via: qs byte → 9-bit grid index (with qh high bit) →
// iq3s_grid[idx] → 4 packed uint8 values, then sign-applied from signs[].
// Computes: dot = sum(sign[i] * grid_value[i] * act[i]) for i in 0..31
__device__ __forceinline__ float tmac_iq3_s_subblock_dot(
    const block_iq3_s_internal * __restrict__ blk,
    int j,                            // sub-block index 0-7
    const float4 * __restrict__ av,   // 8 x float4 activations for this sub-block
    const uint32_t * __restrict__ lut // shared memory LUT (iq3s_grid, 512 uint32 entries)
) {
    float dot = 0.0f;
    const uint8_t * qs = blk->qs + j * 8;      // 8 grid indices for this sub-block
    const uint8_t qh = blk->qh[j];             // 8 high bits
    const uint8_t * signs = blk->signs + j * 4; // 32 sign bits (4 bytes)

    #pragma unroll
    for (int g = 0; g < 8; g++) {
        // 9-bit grid index: 8 bits from qs + 1 bit from qh
        const int idx = (int)qs[g] | (((qh >> g) & 1) << 8);
        const uint32_t grid_val = lut[idx];

        // Extract 4 uint8 values from grid_val
        const float gv0 = (float)((grid_val >>  0) & 0xFF);
        const float gv1 = (float)((grid_val >>  8) & 0xFF);
        const float gv2 = (float)((grid_val >> 16) & 0xFF);
        const float gv3 = (float)((grid_val >> 24) & 0xFF);

        // Apply sign bits: 1 bit per element, 4 elements per grid lookup
        // signs[g/2] has 8 bits, upper/lower nibble for even/odd g
        const uint8_t sb = signs[g >> 1];
        const int soff = (g & 1) * 4;
        const float s0 = ((sb >> (soff + 0)) & 1) ? -1.0f : 1.0f;
        const float s1 = ((sb >> (soff + 1)) & 1) ? -1.0f : 1.0f;
        const float s2 = ((sb >> (soff + 2)) & 1) ? -1.0f : 1.0f;
        const float s3 = ((sb >> (soff + 3)) & 1) ? -1.0f : 1.0f;

        // Each grid lookup corresponds to 4 consecutive elements = 1 float4
        const float4 a = av[g];
        dot += s0 * gv0 * a.x;
        dot += s1 * gv1 * a.y;
        dot += s2 * gv2 * a.z;
        dot += s3 * gv3 * a.w;
    }
    return dot;
}

// P16: Compute ksigns_iq2xs[key] via parity — eliminates sign LUT from shared memory.
// Identity: ksigns[key] = key ^ (__popc(key) & 1) << 7  (bit 7 = parity of bits 0-6).
// Matches upstream vecdotq.cuh:unpack_ksigns(). Replaces LDS read with 4 VALU ops.
__device__ __forceinline__ uint8_t tmac_compute_ksigns(uint32_t key) {
    const uint32_t p = __popc(key) & 1u;
    return (uint8_t)(key ^ (p << 7));
}

// IQ2_XXS sub-block dot: processes one 32-element sub-block.
// Layout: 4 grid indices (1 byte each from q2) + 4 × 7-bit sign keys + 4-bit scale,
// all packed into 8 bytes (2 × int32 from qs[]).
// Scale: 4-bit from aux32 bits [28:31], formula: (ls * sumi + sumi/2) / 4.
//
// P16 Phase 5: grid values pre-converted to float32 in LDS (stride-9 layout).
// Eliminates per-access bit-extract (v_bfe_u32) + int→float (v_cvt_f32_u32).
// Each grid access is now a direct ds_load_b32 of a float value.
__device__ __forceinline__ float tmac_iq2_xxs_subblock_dot(
    const block_iq2_xxs_internal * __restrict__ blk,
    int j,                             // sub-block index 0-7
    const float4 * __restrict__ av,    // 8 x float4 activations for this sub-block
    const float * __restrict__ grid_f32  // shared memory: stride-9 float32 grid
) {
    const uint16_t * qs_sub = blk->qs + j * 4;
    uint32_t q2;
    memcpy(&q2, qs_sub, 4);
    uint32_t aux32;
    memcpy(&aux32, qs_sub + 2, 4);
    const uint8_t * aux8 = (const uint8_t *)&q2;

    float sumi = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int base = aux8[k] * 9;  // stride-9 offset into float32 grid
        const uint8_t signs = tmac_compute_ksigns((aux32 >> (7 * k)) & 0x7F);

        const float4 a0 = av[k * 2];
        const float4 a1 = av[k * 2 + 1];

        // Direct float32 reads from LDS — no bit-extract, no int→float conversion
        sumi += ((signs & 0x01) ? -grid_f32[base + 0] : grid_f32[base + 0]) * a0.x;
        sumi += ((signs & 0x02) ? -grid_f32[base + 1] : grid_f32[base + 1]) * a0.y;
        sumi += ((signs & 0x04) ? -grid_f32[base + 2] : grid_f32[base + 2]) * a0.z;
        sumi += ((signs & 0x08) ? -grid_f32[base + 3] : grid_f32[base + 3]) * a0.w;
        sumi += ((signs & 0x10) ? -grid_f32[base + 4] : grid_f32[base + 4]) * a1.x;
        sumi += ((signs & 0x20) ? -grid_f32[base + 5] : grid_f32[base + 5]) * a1.y;
        sumi += ((signs & 0x40) ? -grid_f32[base + 6] : grid_f32[base + 6]) * a1.z;
        sumi += ((signs & 0x80) ? -grid_f32[base + 7] : grid_f32[base + 7]) * a1.w;
    }

    const int ls = aux32 >> 28;
    return sumi * (float)(2 * ls + 1) / 8.0f;
}

// IQ3_XXS sub-block dot: processes one 32-element sub-block.
// Layout: 8 grid indices (1 byte each from qs[0..63]) + embedded scales/signs in qs[64..95].
// Each grid index → iq3xxs_grid[idx] → uint32 = 4 packed uint8 values.
// Scale: 4-bit from aux32 bits [28:31], formula: d * (0.5 + scale) * 0.5.
__device__ __forceinline__ float tmac_iq3_xxs_subblock_dot(
    const block_iq3_xxs_internal * __restrict__ blk,
    int j,                             // sub-block index 0-7
    const float4 * __restrict__ av,    // 8 x float4 activations for this sub-block
    const uint32_t * __restrict__ grid_lut   // shared memory: iq3xxs_grid[256]
) {
    const uint8_t * qs = blk->qs + j * 8;  // 8 grid indices for this sub-block
    const uint8_t * scales_and_signs = blk->qs + QK_K/4;  // after the 64 index bytes
    uint32_t aux32;
    memcpy(&aux32, scales_and_signs + 4 * j, sizeof(uint32_t));

    float sumi = 0.0f;

    #pragma unroll
    for (int l = 0; l < 4; l++) {
        const uint32_t grid1 = grid_lut[qs[2*l+0]];
        const uint32_t grid2 = grid_lut[qs[2*l+1]];
        const uint8_t signs = tmac_compute_ksigns((aux32 >> (7*l)) & 0x7F);

        const float4 a = av[l * 2];
        const float4 a2 = av[l * 2 + 1];

        // Grid1 → 4 values (elements 0-3), Grid2 → 4 values (elements 4-7)
        sumi += ((signs & 1) ? -1.0f : 1.0f) * (float)((grid1 >>  0) & 0xFF) * a.x;
        sumi += ((signs & 2) ? -1.0f : 1.0f) * (float)((grid1 >>  8) & 0xFF) * a.y;
        sumi += ((signs & 4) ? -1.0f : 1.0f) * (float)((grid1 >> 16) & 0xFF) * a.z;
        sumi += ((signs & 8) ? -1.0f : 1.0f) * (float)((grid1 >> 24) & 0xFF) * a.w;
        sumi += ((signs & 16)  ? -1.0f : 1.0f) * (float)((grid2 >>  0) & 0xFF) * a2.x;
        sumi += ((signs & 32)  ? -1.0f : 1.0f) * (float)((grid2 >>  8) & 0xFF) * a2.y;
        sumi += ((signs & 64)  ? -1.0f : 1.0f) * (float)((grid2 >> 16) & 0xFF) * a2.z;
        sumi += ((signs & 128) ? -1.0f : 1.0f) * (float)((grid2 >> 24) & 0xFF) * a2.w;
    }

    // Scale: 4-bit from bits 28-31. Formula: sumi * (0.5 + scale) * 0.5
    const int scale = aux32 >> 28;
    return sumi * (0.5f + (float)scale) * 0.5f;
}

// IQ2_XS sub-block dot: processes one 32-element sub-block.
// Layout: 4 uint16 from qs[] per sub-block. Each uint16: lower 9 bits = grid index,
// upper 7 bits = sign key. TWO 4-bit scales per sub-block (lo/hi nibble of scales[j]).
// Grid: iq2xs_grid[512] -> uint64 = 8 packed uint8 values.
__device__ __forceinline__ float tmac_iq2_xs_subblock_dot(
    const block_iq2_xs_internal * __restrict__ blk,
    int j,                             // sub-block index 0-7
    const float4 * __restrict__ av,    // 8 x float4 activations for this sub-block
    const uint64_t * __restrict__ grid_lut   // shared memory: iq2xs_grid[512]
) {
    const uint16_t * qs = blk->qs + j * 4;  // 4 uint16 for this sub-block
    // Two scales per sub-block: low nibble for first 16 elements, high for last 16
    const float db0 = (0.5f + (float)(blk->scales[j] & 0xf)) * 0.25f;
    const float db1 = (0.5f + (float)(blk->scales[j] >>  4)) * 0.25f;

    float sumi = 0.0f;

    #pragma unroll
    for (int l = 0; l < 4; l++) {
        const float dl = (l < 2) ? db0 : db1;
        const uint64_t grid_val = grid_lut[qs[l] & 511];  // lower 9 bits
        const uint8_t signs = tmac_compute_ksigns(qs[l] >> 9);  // upper 7 bits

        const float4 a0 = av[l * 2];
        const float4 a1 = av[l * 2 + 1];

        float sub = 0.0f;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            const float gv = (float)((grid_val >> (b * 8)) & 0xFF);
            const float s  = ((signs >> b) & 1) ? -1.0f : 1.0f;
            if (b < 4) {
                const float act = (b == 0) ? a0.x : (b == 1) ? a0.y : (b == 2) ? a0.z : a0.w;
                sub += s * gv * act;
            } else {
                const float act = (b == 4) ? a1.x : (b == 5) ? a1.y : (b == 6) ? a1.z : a1.w;
                sub += s * gv * act;
            }
        }
        sumi += dl * sub;
    }

    return sumi;
}

// IQ2_S sub-block dot: processes one 32-element sub-block.
// Layout: 4 bytes from qs[] (lower 8 bits of index) + 2 bits from qh[] (high bits)
// -> 10-bit grid index -> iq2s_grid[1024] -> uint64 = 8 packed uint8 values.
// Sign bits: 4 bytes from signs array (= qs + 32 from block start), 1 bit per element.
// TWO 4-bit scales per sub-block (lo/hi nibble of scales[j]).
__device__ __forceinline__ float tmac_iq2_s_subblock_dot(
    const block_iq2_s_internal * __restrict__ blk,
    int j,                             // sub-block index 0-7
    const float4 * __restrict__ av,    // 8 x float4 activations for this sub-block
    const uint64_t * __restrict__ grid_lut   // shared memory: iq2s_grid[1024]
) {
    const uint8_t * qs = blk->qs + j * 4;          // 4 lower-8 index bytes
    const uint8_t * signs = blk->qs + 32 + j * 4;  // signs at qs + QK_K/8
    // Two scales per sub-block: low nibble for first 16 elements, high for last 16
    const float db0 = (0.5f + (float)(blk->scales[j] & 0xf)) * 0.25f;
    const float db1 = (0.5f + (float)(blk->scales[j] >>  4)) * 0.25f;

    float sumi = 0.0f;

    #pragma unroll
    for (int l = 0; l < 4; l++) {
        const float dl = (l < 2) ? db0 : db1;
        // 10-bit grid index: 8 bits from qs + 2 high bits from qh[j]
        // qh[j] byte encodes 4 pairs of high bits (2 bits per l):
        //   l=0: bits 0-1, l=1: bits 2-3, l=2: bits 4-5, l=3: bits 6-7
        const int idx = qs[l] | (((int)blk->qh[j] << (8 - 2*l)) & 0x300);
        const uint64_t grid_val = grid_lut[idx];
        const uint8_t sign_byte = signs[l];

        const float4 a0 = av[l * 2];
        const float4 a1 = av[l * 2 + 1];

        float sub = 0.0f;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            const float gv = (float)((grid_val >> (b * 8)) & 0xFF);
            const float s  = ((sign_byte >> b) & 1) ? -1.0f : 1.0f;
            if (b < 4) {
                const float act = (b == 0) ? a0.x : (b == 1) ? a0.y : (b == 2) ? a0.z : a0.w;
                sub += s * gv * act;
            } else {
                const float act = (b == 4) ? a1.x : (b == 5) ? a1.y : (b == 6) ? a1.z : a1.w;
                sub += s * gv * act;
            }
        }
        sumi += dl * sub;
    }

    return sumi;
}

// IQ1_M: 1.75-bit, 256-element super-blocks, iq1s_grid_gpu[2048] LUT.
// Unique among all quant types: no `d` field in block — super-block scale
// is reassembled from top 4 bits of each uint16 in the scales array.
// Grid values are nibble-packed uint32: low nibbles → elements 0-3, high → 4-7.
// Delta correction per 8-element group from qh bit 3.
__device__ __forceinline__ float tmac_iq1_m_subblock_dot(
    const block_iq1_m_internal * __restrict__ blk,
    int j,                             // sub-block index 0-7
    const float4 * __restrict__ av,    // 8 x float4 activations for this sub-block
    const uint32_t * __restrict__ grid_lut  // shared memory: iq1s_grid_gpu[2048]
) {
    // Super-block scale: reassemble fp16 from top 4 bits of each uint16 in scales
    const uint16_t * sc = (const uint16_t *)blk->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
              | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16);

    // Two 3-bit sub-scales per 32-element sub-block (first 16 + last 16 elements)
    const int tmp = sc[j/2] >> (6 * (j % 2));
    const float db0 = d * (float)(2 * ((tmp >> 0) & 7) + 1);
    const float db1 = d * (float)(2 * ((tmp >> 3) & 7) + 1);

    const uint8_t * qs_j = blk->qs + j * 4;
    const uint8_t * qh_j = blk->qh + j * 2;

    float sumi0 = 0.0f, sumi1 = 0.0f;

    #pragma unroll
    for (int l = 0; l < 4; l++) {
        const int qhl = qh_j[l/2] >> (4 * (l % 2));
        const int idx = qs_j[l] | ((qhl & 0x07) << 8);
        const uint32_t grid = grid_lut[idx];
        const float delta = -1.0f + IQ1M_DELTA
                          - (float)(qhl & 0x08) * (2.0f * IQ1M_DELTA / 8.0f);

        const float4 a0 = av[l * 2];
        const float4 a1 = av[l * 2 + 1];

        // IQ1_M grid uint32: byte-interleaved nibbles.
        // Low nibbles at bit positions 0,8,16,24 → elements 0-3 (a0.xyzw)
        // High nibbles at bit positions 4,12,20,28 → elements 4-7 (a1.xyzw)
        float sub = 0.0f;
        sub += ((float)((grid >>  0) & 0xF) + delta) * a0.x;
        sub += ((float)((grid >>  8) & 0xF) + delta) * a0.y;
        sub += ((float)((grid >> 16) & 0xF) + delta) * a0.z;
        sub += ((float)((grid >> 24) & 0xF) + delta) * a0.w;
        sub += ((float)((grid >>  4) & 0xF) + delta) * a1.x;
        sub += ((float)((grid >> 12) & 0xF) + delta) * a1.y;
        sub += ((float)((grid >> 20) & 0xF) + delta) * a1.z;
        sub += ((float)((grid >> 28) & 0xF) + delta) * a1.w;

        if (l < 2) sumi0 += sub;
        else sumi1 += sub;
    }

    return db0 * sumi0 + db1 * sumi1;
}

// ═════════════════════════════════════════════════════════════════════
// Type-specific dot product functions (32-element blocks)
// ═════════════════════════════════════════════════════════════════════

// Q4_0: v_bfe_u32 nibble extraction, zero-point 8, scale d
__device__ __forceinline__ float tmac_q4_0_dot(
    const block_q4_0_internal & blk,
    const float4 av0, const float4 av1, const float4 av2, const float4 av3,
    const float4 av4, const float4 av5, const float4 av6, const float4 av7
) {
    float acc = 0.0f;
    uint32_t q4;

    memcpy(&q4, blk.qs + 0, sizeof(q4));
    acc += ((float)tmac_bfe4(q4,  0) - 8.0f) * av0.x;
    acc += ((float)tmac_bfe4(q4,  4) - 8.0f) * av4.x;
    acc += ((float)tmac_bfe4(q4,  8) - 8.0f) * av0.y;
    acc += ((float)tmac_bfe4(q4, 12) - 8.0f) * av4.y;
    acc += ((float)tmac_bfe4(q4, 16) - 8.0f) * av0.z;
    acc += ((float)tmac_bfe4(q4, 20) - 8.0f) * av4.z;
    acc += ((float)tmac_bfe4(q4, 24) - 8.0f) * av0.w;
    acc += ((float)tmac_bfe4(q4, 28) - 8.0f) * av4.w;

    memcpy(&q4, blk.qs + 4, sizeof(q4));
    acc += ((float)tmac_bfe4(q4,  0) - 8.0f) * av1.x;
    acc += ((float)tmac_bfe4(q4,  4) - 8.0f) * av5.x;
    acc += ((float)tmac_bfe4(q4,  8) - 8.0f) * av1.y;
    acc += ((float)tmac_bfe4(q4, 12) - 8.0f) * av5.y;
    acc += ((float)tmac_bfe4(q4, 16) - 8.0f) * av1.z;
    acc += ((float)tmac_bfe4(q4, 20) - 8.0f) * av5.z;
    acc += ((float)tmac_bfe4(q4, 24) - 8.0f) * av1.w;
    acc += ((float)tmac_bfe4(q4, 28) - 8.0f) * av5.w;

    memcpy(&q4, blk.qs + 8, sizeof(q4));
    acc += ((float)tmac_bfe4(q4,  0) - 8.0f) * av2.x;
    acc += ((float)tmac_bfe4(q4,  4) - 8.0f) * av6.x;
    acc += ((float)tmac_bfe4(q4,  8) - 8.0f) * av2.y;
    acc += ((float)tmac_bfe4(q4, 12) - 8.0f) * av6.y;
    acc += ((float)tmac_bfe4(q4, 16) - 8.0f) * av2.z;
    acc += ((float)tmac_bfe4(q4, 20) - 8.0f) * av6.z;
    acc += ((float)tmac_bfe4(q4, 24) - 8.0f) * av2.w;
    acc += ((float)tmac_bfe4(q4, 28) - 8.0f) * av6.w;

    memcpy(&q4, blk.qs + 12, sizeof(q4));
    acc += ((float)tmac_bfe4(q4,  0) - 8.0f) * av3.x;
    acc += ((float)tmac_bfe4(q4,  4) - 8.0f) * av7.x;
    acc += ((float)tmac_bfe4(q4,  8) - 8.0f) * av3.y;
    acc += ((float)tmac_bfe4(q4, 12) - 8.0f) * av7.y;
    acc += ((float)tmac_bfe4(q4, 16) - 8.0f) * av3.z;
    acc += ((float)tmac_bfe4(q4, 20) - 8.0f) * av7.z;
    acc += ((float)tmac_bfe4(q4, 24) - 8.0f) * av3.w;
    acc += ((float)tmac_bfe4(q4, 28) - 8.0f) * av7.w;

    return acc;
}

// Q8_0: byte extraction, no zero-point, scale d
__device__ __forceinline__ float tmac_q8_0_dot(
    const block_q8_0_internal & blk,
    const float4 av0, const float4 av1, const float4 av2, const float4 av3,
    const float4 av4, const float4 av5, const float4 av6, const float4 av7
) {
    float acc = 0.0f;
    int32_t q32;

    memcpy(&q32, blk.qs + 0, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av0.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av0.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av0.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av0.w;

    memcpy(&q32, blk.qs + 4, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av1.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av1.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av1.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av1.w;

    memcpy(&q32, blk.qs + 8, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av2.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av2.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av2.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av2.w;

    memcpy(&q32, blk.qs + 12, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av3.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av3.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av3.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av3.w;

    memcpy(&q32, blk.qs + 16, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av4.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av4.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av4.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av4.w;

    memcpy(&q32, blk.qs + 20, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av5.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av5.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av5.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av5.w;

    memcpy(&q32, blk.qs + 24, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av6.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av6.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av6.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av6.w;

    memcpy(&q32, blk.qs + 28, sizeof(q32));
    acc += (float)(int8_t)(q32       ) * av7.x;
    acc += (float)(int8_t)(q32 >>  8 ) * av7.y;
    acc += (float)(int8_t)(q32 >> 16 ) * av7.z;
    acc += (float)(int8_t)(q32 >> 24 ) * av7.w;

    return acc;
}

// Q5_1: dual accumulator — val[i] = low_4bits | (high_bit << 4), w[i] = val[i]*d + m
// Returns sum(val[i]*act[i]) via sum_val_act, sum(act[i]) via sum_act (both accumulated).
__device__ __forceinline__ void tmac_q5_1_dot(
    const block_q5_1_internal & blk,
    const float4 av0, const float4 av1, const float4 av2, const float4 av3,
    const float4 av4, const float4 av5, const float4 av6, const float4 av7,
    float & sum_val_act, float & sum_act
) {
    uint32_t qh; memcpy(&qh, blk.qh, sizeof(qh));
    uint32_t q4;

    sum_act += av0.x + av0.y + av0.z + av0.w;
    sum_act += av1.x + av1.y + av1.z + av1.w;
    sum_act += av2.x + av2.y + av2.z + av2.w;
    sum_act += av3.x + av3.y + av3.z + av3.w;
    sum_act += av4.x + av4.y + av4.z + av4.w;
    sum_act += av5.x + av5.y + av5.z + av5.w;
    sum_act += av6.x + av6.y + av6.z + av6.w;
    sum_act += av7.x + av7.y + av7.z + av7.w;

    memcpy(&q4, blk.qs + 0, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >>  0) & 1) << 4)) * av0.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >>  1) & 1) << 4)) * av0.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >>  2) & 1) << 4)) * av0.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >>  3) & 1) << 4)) * av0.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 16) & 1) << 4)) * av4.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 17) & 1) << 4)) * av4.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 18) & 1) << 4)) * av4.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 19) & 1) << 4)) * av4.w;

    memcpy(&q4, blk.qs + 4, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >>  4) & 1) << 4)) * av1.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >>  5) & 1) << 4)) * av1.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >>  6) & 1) << 4)) * av1.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >>  7) & 1) << 4)) * av1.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 20) & 1) << 4)) * av5.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 21) & 1) << 4)) * av5.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 22) & 1) << 4)) * av5.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 23) & 1) << 4)) * av5.w;

    memcpy(&q4, blk.qs + 8, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >>  8) & 1) << 4)) * av2.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >>  9) & 1) << 4)) * av2.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >> 10) & 1) << 4)) * av2.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >> 11) & 1) << 4)) * av2.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 24) & 1) << 4)) * av6.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 25) & 1) << 4)) * av6.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 26) & 1) << 4)) * av6.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 27) & 1) << 4)) * av6.w;

    memcpy(&q4, blk.qs + 12, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >> 12) & 1) << 4)) * av3.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >> 13) & 1) << 4)) * av3.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >> 14) & 1) << 4)) * av3.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >> 15) & 1) << 4)) * av3.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 28) & 1) << 4)) * av7.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 29) & 1) << 4)) * av7.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 30) & 1) << 4)) * av7.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 31) & 1) << 4)) * av7.w;
}

// Q5_0: same bit packing as Q5_1, symmetric dequant: w[i] = d * (val[i] - 16)
// Returns sum(val[i]*act[i]) via sum_val_act, sum(act[i]) via sum_act (both accumulated).
// Caller computes: d * (sum_val_act - 16 * sum_act).
__device__ __forceinline__ void tmac_q5_0_dot(
    const block_q5_0_internal & blk,
    const float4 av0, const float4 av1, const float4 av2, const float4 av3,
    const float4 av4, const float4 av5, const float4 av6, const float4 av7,
    float & sum_val_act, float & sum_act
) {
    uint32_t qh; memcpy(&qh, blk.qh, sizeof(qh));
    uint32_t q4;

    sum_act += av0.x + av0.y + av0.z + av0.w;
    sum_act += av1.x + av1.y + av1.z + av1.w;
    sum_act += av2.x + av2.y + av2.z + av2.w;
    sum_act += av3.x + av3.y + av3.z + av3.w;
    sum_act += av4.x + av4.y + av4.z + av4.w;
    sum_act += av5.x + av5.y + av5.z + av5.w;
    sum_act += av6.x + av6.y + av6.z + av6.w;
    sum_act += av7.x + av7.y + av7.z + av7.w;

    memcpy(&q4, blk.qs + 0, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >>  0) & 1) << 4)) * av0.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >>  1) & 1) << 4)) * av0.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >>  2) & 1) << 4)) * av0.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >>  3) & 1) << 4)) * av0.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 16) & 1) << 4)) * av4.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 17) & 1) << 4)) * av4.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 18) & 1) << 4)) * av4.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 19) & 1) << 4)) * av4.w;

    memcpy(&q4, blk.qs + 4, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >>  4) & 1) << 4)) * av1.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >>  5) & 1) << 4)) * av1.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >>  6) & 1) << 4)) * av1.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >>  7) & 1) << 4)) * av1.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 20) & 1) << 4)) * av5.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 21) & 1) << 4)) * av5.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 22) & 1) << 4)) * av5.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 23) & 1) << 4)) * av5.w;

    memcpy(&q4, blk.qs + 8, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >>  8) & 1) << 4)) * av2.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >>  9) & 1) << 4)) * av2.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >> 10) & 1) << 4)) * av2.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >> 11) & 1) << 4)) * av2.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 24) & 1) << 4)) * av6.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 25) & 1) << 4)) * av6.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 26) & 1) << 4)) * av6.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 27) & 1) << 4)) * av6.w;

    memcpy(&q4, blk.qs + 12, sizeof(q4));
    sum_val_act += (float)(tmac_bfe4(q4,  0) | (((qh >> 12) & 1) << 4)) * av3.x;
    sum_val_act += (float)(tmac_bfe4(q4,  8) | (((qh >> 13) & 1) << 4)) * av3.y;
    sum_val_act += (float)(tmac_bfe4(q4, 16) | (((qh >> 14) & 1) << 4)) * av3.z;
    sum_val_act += (float)(tmac_bfe4(q4, 24) | (((qh >> 15) & 1) << 4)) * av3.w;
    sum_val_act += (float)(tmac_bfe4(q4,  4) | (((qh >> 28) & 1) << 4)) * av7.x;
    sum_val_act += (float)(tmac_bfe4(q4, 12) | (((qh >> 29) & 1) << 4)) * av7.y;
    sum_val_act += (float)(tmac_bfe4(q4, 20) | (((qh >> 30) & 1) << 4)) * av7.z;
    sum_val_act += (float)(tmac_bfe4(q4, 28) | (((qh >> 31) & 1) << 4)) * av7.w;
}

// MXFP4: LUT-based dequant via kvalues_mxfp4[16] (constexpr __device__, L0/K$ resident).
// Same interleaved nibble pattern as Q4_0: low nibble = element j, high = element j+16.
// Returns unscaled dot product; caller applies e8m0_to_fp32(e) * 0.5f.
__device__ __forceinline__ float tmac_mxfp4_dot(
    const block_mxfp4_internal & blk,
    const float4 av0, const float4 av1, const float4 av2, const float4 av3,
    const float4 av4, const float4 av5, const float4 av6, const float4 av7
) {
    float acc = 0.0f;
    uint32_t q4;

    memcpy(&q4, blk.qs + 0, sizeof(q4));
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  0)] * av0.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  4)] * av4.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  8)] * av0.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 12)] * av4.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 16)] * av0.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 20)] * av4.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 24)] * av0.w;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 28)] * av4.w;

    memcpy(&q4, blk.qs + 4, sizeof(q4));
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  0)] * av1.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  4)] * av5.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  8)] * av1.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 12)] * av5.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 16)] * av1.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 20)] * av5.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 24)] * av1.w;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 28)] * av5.w;

    memcpy(&q4, blk.qs + 8, sizeof(q4));
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  0)] * av2.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  4)] * av6.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  8)] * av2.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 12)] * av6.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 16)] * av2.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 20)] * av6.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 24)] * av2.w;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 28)] * av6.w;

    memcpy(&q4, blk.qs + 12, sizeof(q4));
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  0)] * av3.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  4)] * av7.x;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4,  8)] * av3.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 12)] * av7.y;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 16)] * av3.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 20)] * av7.z;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 24)] * av3.w;
    acc += (float)kvalues_mxfp4[tmac_bfe4(q4, 28)] * av7.w;

    return acc;
}

// IQ4_NL: same block layout as Q4_0 (18 bytes, 32 elements, interleaved nibbles).
// Non-linear dequant via kvalues_iq4nl[16] (constexpr __device__, K$ resident).
// Structurally identical to MXFP4 dot but with kvalues_iq4nl LUT.
__device__ __forceinline__ float tmac_iq4_nl_dot(
    const block_q4_0_internal & blk,   // reuses Q4_0 block struct (identical layout)
    const float4 av0, const float4 av1, const float4 av2, const float4 av3,
    const float4 av4, const float4 av5, const float4 av6, const float4 av7
) {
    float acc = 0.0f;
    uint32_t q4;

    memcpy(&q4, blk.qs + 0, sizeof(q4));
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  0)] * av0.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  4)] * av4.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  8)] * av0.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 12)] * av4.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 16)] * av0.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 20)] * av4.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 24)] * av0.w;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 28)] * av4.w;

    memcpy(&q4, blk.qs + 4, sizeof(q4));
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  0)] * av1.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  4)] * av5.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  8)] * av1.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 12)] * av5.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 16)] * av1.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 20)] * av5.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 24)] * av1.w;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 28)] * av5.w;

    memcpy(&q4, blk.qs + 8, sizeof(q4));
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  0)] * av2.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  4)] * av6.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  8)] * av2.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 12)] * av6.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 16)] * av2.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 20)] * av6.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 24)] * av2.w;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 28)] * av6.w;

    memcpy(&q4, blk.qs + 12, sizeof(q4));
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  0)] * av3.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  4)] * av7.x;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4,  8)] * av3.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 12)] * av7.y;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 16)] * av3.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 20)] * av7.z;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 24)] * av3.w;
    acc += (float)kvalues_iq4nl[tmac_bfe4(q4, 28)] * av7.w;

    return acc;
}

// ═════════════════════════════════════════════════════════════════════
// Quant type traits — abstracts block type, dot product, and scaling
//
// dot_scale() returns a single float per block: the fully-scaled
// contribution to the output dot product. For Q5_1 (dual accumulator),
// this merges d*sum(val*act) + m*sum(act) per block — mathematically
// equivalent to separate accumulation via sum associativity.
// ═════════════════════════════════════════════════════════════════════

template <ggml_type QType> struct tmac_quant_traits;

template <> struct tmac_quant_traits<GGML_TYPE_Q4_0> {
    using block_type = block_q4_0_internal;
    static constexpr int elements_per_block = 32;
    static constexpr bool is_sub_block_parallel = false;
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;  // 56 VGPRs -> 1 block/CU
    static constexpr int fused_min_blocks   = 2;  // 96 VGPRs -> 2 blocks/CU

    static __device__ __forceinline__ float dot_scale(
        const block_type & blk,
        float4 a0, float4 a1, float4 a2, float4 a3,
        float4 a4, float4 a5, float4 a6, float4 a7) {
        return tmac_q4_0_dot(blk, a0, a1, a2, a3, a4, a5, a6, a7)
               * __half2float(blk.d);
    }
};

template <> struct tmac_quant_traits<GGML_TYPE_Q8_0> {
    using block_type = block_q8_0_internal;
    static constexpr int elements_per_block = 32;
    static constexpr bool is_sub_block_parallel = false;
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 2;

    static __device__ __forceinline__ float dot_scale(
        const block_type & blk,
        float4 a0, float4 a1, float4 a2, float4 a3,
        float4 a4, float4 a5, float4 a6, float4 a7) {
        return tmac_q8_0_dot(blk, a0, a1, a2, a3, a4, a5, a6, a7)
               * __half2float(blk.d);
    }
};

template <> struct tmac_quant_traits<GGML_TYPE_Q5_1> {
    using block_type = block_q5_1_internal;
    static constexpr int elements_per_block = 32;
    static constexpr bool is_sub_block_parallel = false;
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 2;

    static __device__ __forceinline__ float dot_scale(
        const block_type & blk,
        float4 a0, float4 a1, float4 a2, float4 a3,
        float4 a4, float4 a5, float4 a6, float4 a7) {
        float bva = 0.0f, ba = 0.0f;
        tmac_q5_1_dot(blk, a0, a1, a2, a3, a4, a5, a6, a7, bva, ba);
        return bva * __half2float(blk.d) + ba * __half2float(blk.m);
    }
};

template <> struct tmac_quant_traits<GGML_TYPE_Q5_0> {
    using block_type = block_q5_0_internal;
    static constexpr int elements_per_block = 32;
    static constexpr bool is_sub_block_parallel = false;
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 2;

    static __device__ __forceinline__ float dot_scale(
        const block_type & blk,
        float4 a0, float4 a1, float4 a2, float4 a3,
        float4 a4, float4 a5, float4 a6, float4 a7) {
        float bva = 0.0f, ba = 0.0f;
        tmac_q5_0_dot(blk, a0, a1, a2, a3, a4, a5, a6, a7, bva, ba);
        return __half2float(blk.d) * (bva - 16.0f * ba);
    }
};

// MXFP4: OCP Microscaling 4-bit (17 bytes/block, 32 elements).
// Same geometry as Q4_0 but uses LUT dequant via kvalues_mxfp4[16] (constexpr __device__).
// Scale: E8M0 exponent (uint8_t e), not FP16. Conversion via ggml_cuda_e8m0_to_fp32().
template <> struct tmac_quant_traits<GGML_TYPE_MXFP4> {
    using block_type = block_mxfp4_internal;
    static constexpr int elements_per_block = 32;
    static constexpr bool is_sub_block_parallel = false;
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 2;

    static __device__ __forceinline__ float dot_scale(
        const block_type & blk,
        float4 a0, float4 a1, float4 a2, float4 a3,
        float4 a4, float4 a5, float4 a6, float4 a7) {
        return tmac_mxfp4_dot(blk, a0, a1, a2, a3, a4, a5, a6, a7)
               * ggml_cuda_e8m0_to_fp32(blk.e) * 0.5f;
    }
};

// Q4_K: 256-element super-blocks. No dot_scale — sub-block parallelization
// is inlined directly in the kernel's else branch (if constexpr).
template <> struct tmac_quant_traits<GGML_TYPE_Q4_K> {
    using block_type = block_q4_K_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;    // ~53 VGPRs estimated
    static constexpr int fused_min_blocks   = 1;    // ~69 VGPRs estimated
};

// Q5_K: 256-element super-blocks. Same geometry as Q4_K (8×32) but 5-bit values.
// Reuses Q4_K scale decoder. No dot_scale — sub-block parallelization path.
template <> struct tmac_quant_traits<GGML_TYPE_Q5_K> {
    using block_type = block_q5_K_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// Q6_K: 256-element super-blocks, 6-bit quants with 8-bit scales.
// 16 sub-blocks of 16 elements — finer granularity than Q4_K's 8×32.
template <> struct tmac_quant_traits<GGML_TYPE_Q6_K> {
    using block_type = block_q6_K_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 16;
    static constexpr int sub_block_size = 16;
    static constexpr int sub_block_shift = 4;       // log2(16)
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ4_NL: 32-element blocks with non-linear LUT dequant (kvalues_iq4nl[16]).
// Same block layout as Q4_0 (18 bytes: half d + 16 nibble bytes). No sub-blocks.
// Reuses block_q4_0_internal. Constant memory LUT (K$ resident, no smem).
template <> struct tmac_quant_traits<GGML_TYPE_IQ4_NL> {
    using block_type = block_q4_0_internal;  // identical layout: half d + uint8_t qs[16]
    static constexpr int elements_per_block = 32;
    static constexpr bool is_sub_block_parallel = false;
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 2;

    static __device__ __forceinline__ float dot_scale(
        const block_type & blk,
        float4 a0, float4 a1, float4 a2, float4 a3,
        float4 a4, float4 a5, float4 a6, float4 a7) {
        return tmac_iq4_nl_dot(blk, a0, a1, a2, a3, a4, a5, a6, a7)
               * __half2float(blk.d);
    }
};

// Q3_K: 256-element super-blocks, 3-bit quants (2-bit qs + 1-bit hmask).
// 16 sub-blocks of 16 elements — same geometry as Q6_K but 3-bit values.
// Symmetric with signed 6-bit scales (scale - 32). No dmin term.
template <> struct tmac_quant_traits<GGML_TYPE_Q3_K> {
    using block_type = block_q3_K_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 16;
    static constexpr int sub_block_size = 16;
    static constexpr int sub_block_shift = 4;       // log2(16)
    static constexpr int smem_bytes = 0;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ4_XS: 256-element super-blocks with LUT-indexed 4-bit quants.
// 8 sub-blocks of 32 elements — same geometry as Q4_K but with non-linear
// dequant via kvalues_iq4nl[16] LUT loaded into shared memory.
template <> struct tmac_quant_traits<GGML_TYPE_IQ4_XS> {
    using block_type = block_iq4_xs_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr bool needs_smem = true;
    static constexpr int smem_bytes = 16;           // kvalues_iq4nl[16] × 1 byte
    static constexpr int lut_entries = 16;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ3_S: 256-element super-blocks with LUT-indexed 3-bit quants + signs.
// 8 sub-blocks of 32 elements — each decoded via iq3s_grid[512] (2KB LUT).
template <> struct tmac_quant_traits<GGML_TYPE_IQ3_S> {
    using block_type = block_iq3_s_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr bool needs_smem = true;
    static constexpr int smem_bytes = 512 * 4;      // iq3s_grid[512] × 4 bytes = 2048B
    static constexpr int lut_entries = 512;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ2_XXS: 256-element super-blocks with extreme 2-bit compression.
// 8 sub-blocks of 32 elements — decoded via iq2xxs_grid[256].
// P16 Phase 1: ksigns_iq2xs sign LUT eliminated (computed via parity).
// P16 Phase 5: grid LUT pre-converted to float32 with stride-9 padding (zero bank conflicts).
//   Layout: grid_f32[256][9] — elements 0-7 = float32 grid values, element 8 = padding.
//   Stride-9 guarantees zero LDS bank conflicts: gcd(9, 32) = 1.
//   LDS: 256 × 9 × 4B = 9216B (was 2048B).
template <> struct tmac_quant_traits<GGML_TYPE_IQ2_XXS> {
    using block_type = block_iq2_xxs_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr bool needs_smem = true;
    static constexpr int smem_bytes = 256 * 9 * sizeof(float);  // 9216B (stride-9 float32)
    static constexpr int lut_entries = 256;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ3_XXS: 256-element super-blocks with LUT-indexed 3-bit quants + embedded signs.
// 8 sub-blocks of 32 elements — decoded via iq3xxs_grid[256] (1KB). P16: sign LUT eliminated.
template <> struct tmac_quant_traits<GGML_TYPE_IQ3_XXS> {
    using block_type = block_iq3_xxs_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr bool needs_smem = true;
    static constexpr int smem_bytes = 256 * 4;        // 256 × 4B = 1024B
    static constexpr int lut_entries = 256;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ2_XS: 256-element super-blocks with 512-entry grid + separate scales.
// 8 sub-blocks of 32 elements — decoded via iq2xs_grid[512] (4KB). P16: sign LUT eliminated.
template <> struct tmac_quant_traits<GGML_TYPE_IQ2_XS> {
    using block_type = block_iq2_xs_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr bool needs_smem = true;
    static constexpr int smem_bytes = 512 * 8;        // 512 × 8B = 4096B
    static constexpr int lut_entries = 512;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ2_S: 256-element super-blocks with 1024-entry grid + separate signs + scales.
// 8 sub-blocks of 32 elements — decoded via iq2s_grid[1024] (8KB). No sign LUT needed
// (signs stored directly in block data).
template <> struct tmac_quant_traits<GGML_TYPE_IQ2_S> {
    using block_type = block_iq2_s_internal;
    static constexpr int elements_per_block = 256;  // QK_K
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr bool needs_smem = true;
    static constexpr int smem_bytes = 1024 * 8;     // iq2s_grid (8192B). No sign LUT needed.
    static constexpr int lut_entries = 1024;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// IQ1_M: 256-element super-blocks with 2048-entry grid (shared with IQ1_S).
// 8 sub-blocks of 32 elements — decoded via iq1s_grid_gpu[2048] (8KB uint32 LUT).
// Scale extracted from packed scales array (no separate d field).
template <> struct tmac_quant_traits<GGML_TYPE_IQ1_M> {
    using block_type = block_iq1_m_internal;
    static constexpr int elements_per_block = 256;
    static constexpr bool is_sub_block_parallel = true;
    static constexpr int sub_blocks_per_block = 8;
    static constexpr int sub_block_size = 32;
    static constexpr int sub_block_shift = 3;       // log2(8)
    static constexpr bool needs_smem = true;
    static constexpr int smem_bytes = 2048 * 4;     // iq1s_grid_gpu (8192B). No sign LUT needed.
    static constexpr int lut_entries = 2048;
    static constexpr int unfused_min_blocks = 1;
    static constexpr int fused_min_blocks   = 1;
};

// ═════════════════════════════════════════════════════════════════════
// Shared memory context for IQ LUT pointers.
// Groups all IQ type LUT pointers into a single struct to avoid passing
// 4 separate pointer params (3-4 always nullptr) through call chains.
// ═════════════════════════════════════════════════════════════════════

struct tmac_smem_context {
    const int8_t   * lut_i8    = nullptr;  // IQ4_XS: kvalues_iq4nl
    const float    * lut_f32   = nullptr;  // IQ2_XXS: pre-converted float32 grid (stride-9, P16 Phase 5)
    const uint32_t * lut_u32   = nullptr;  // IQ3_S/IQ3_XXS: iq3s_grid/iq3xxs_grid
    const uint64_t * lut_u64   = nullptr;  // IQ2_XS/IQ2_S: iq2xs_grid/iq2s_grid
    // P16: lut_signs removed — sign bits computed inline via tmac_compute_ksigns()
};

// ═════════════════════════════════════════════════════════════════════
// IQ LUT loader — populates shared memory and returns smem_context.
// Called once at kernel entry. Non-IQ types return an empty context.
// ═════════════════════════════════════════════════════════════════════

template <ggml_type QType>
__device__ __forceinline__ tmac_smem_context tmac_load_iq_lut() {
    using Traits = tmac_quant_traits<QType>;
    tmac_smem_context ctx;

    if constexpr (QType == GGML_TYPE_IQ4_XS) {
        extern __shared__ char smem[];
        if (threadIdx.x < Traits::lut_entries) {
            reinterpret_cast<int8_t *>(smem)[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
        }
        __syncthreads();
        ctx.lut_i8 = reinterpret_cast<const int8_t *>(smem);
    } else if constexpr (QType == GGML_TYPE_IQ3_S) {
        extern __shared__ char smem[];
        for (int i = threadIdx.x; i < Traits::lut_entries; i += 256) {
            reinterpret_cast<uint32_t *>(smem)[i] = iq3s_grid[i];
        }
        __syncthreads();
        ctx.lut_u32 = reinterpret_cast<const uint32_t *>(smem);
    } else if constexpr (QType == GGML_TYPE_IQ2_XXS) {
        // P16 Phase 5: pre-convert grid uint64 → 8×float32 with stride-9 padding.
        // Stride-9 eliminates LDS bank conflicts: gcd(9, 32) = 1.
        // Each thread handles one entry (256 threads = 256 entries, one pass).
        extern __shared__ char smem[];
        float * f32_lut = reinterpret_cast<float *>(smem);
        for (int i = threadIdx.x; i < 256; i += 256) {
            const uint64_t gv = iq2xxs_grid[i];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                f32_lut[i * 9 + b] = (float)((gv >> (b * 8)) & 0xFF);
            }
            f32_lut[i * 9 + 8] = 0.0f;  // stride-9 padding
        }
        // P16 Phase 1: ksigns_iq2xs sign LUT eliminated — computed via parity (tmac_compute_ksigns)
        __syncthreads();
        ctx.lut_f32 = f32_lut;
    } else if constexpr (QType == GGML_TYPE_IQ3_XXS) {
        extern __shared__ char smem[];
        // iq3xxs_grid[256] × uint32 = 1024B
        for (int i = threadIdx.x; i < 256; i += 256) {
            reinterpret_cast<uint32_t *>(smem)[i] = iq3xxs_grid[i];
        }
        __syncthreads();
        ctx.lut_u32 = reinterpret_cast<const uint32_t *>(smem);
    } else if constexpr (QType == GGML_TYPE_IQ2_XS) {
        extern __shared__ char smem[];
        // iq2xs_grid[512] × uint64 = 4096B
        for (int i = threadIdx.x; i < 512; i += 256) {
            reinterpret_cast<uint64_t *>(smem)[i] = iq2xs_grid[i];
        }
        __syncthreads();
        ctx.lut_u64 = reinterpret_cast<const uint64_t *>(smem);
    } else if constexpr (QType == GGML_TYPE_IQ2_S) {
        extern __shared__ char smem[];
        // iq2s_grid[1024] × uint64 = 8192B — no sign LUT needed (signs in block data)
        for (int i = threadIdx.x; i < 1024; i += 256) {
            reinterpret_cast<uint64_t *>(smem)[i] = iq2s_grid[i];
        }
        __syncthreads();
        ctx.lut_u64 = reinterpret_cast<const uint64_t *>(smem);
    } else if constexpr (QType == GGML_TYPE_IQ1_M) {
        extern __shared__ char smem[];
        // iq1s_grid_gpu[2048] × uint32 = 8192B — shared with IQ1_S, no sign LUT needed
        for (int i = threadIdx.x; i < Traits::lut_entries; i += 256) {
            reinterpret_cast<uint32_t *>(smem)[i] = iq1s_grid_gpu[i];
        }
        __syncthreads();
        ctx.lut_u32 = reinterpret_cast<const uint32_t *>(smem);
    }

    return ctx;
}

// ═════════════════════════════════════════════════════════════════════
// Sub-block scaled dot product — unified accumulation for all sub-block types
// Called once in unfused kernel, twice in fused kernel (up + gate).
// Returns the scaled contribution to the row accumulator.
// ═════════════════════════════════════════════════════════════════════

template <ggml_type QType>
__device__ __forceinline__ float tmac_subblock_contribution(
    const typename tmac_quant_traits<QType>::block_type * __restrict__ blk,
    int j,
    const float4 * __restrict__ av,
    const tmac_smem_context & ctx
) {
    if constexpr (QType == GGML_TYPE_Q4_K) {
        uint8_t sc, mn;
        tmac_q4_K_decode_scales(j, blk->scales, sc, mn);
        float dot_val = 0.0f, sum_a = 0.0f;
        tmac_q4_K_subblock_dot(blk, j, av, dot_val, sum_a);
        return __half2float(blk->d)    * (float)sc * dot_val
             - __half2float(blk->dmin) * (float)mn * sum_a;
    } else if constexpr (QType == GGML_TYPE_Q5_K) {
        uint8_t sc, mn;
        tmac_q4_K_decode_scales(j, blk->scales, sc, mn);  // same 6-bit packed format as Q4_K
        float dot_val = 0.0f, sum_a = 0.0f;
        tmac_q5_K_subblock_dot(blk, j, av, dot_val, sum_a);
        return __half2float(blk->d)    * (float)sc * dot_val
             - __half2float(blk->dmin) * (float)mn * sum_a;
    } else if constexpr (QType == GGML_TYPE_Q3_K) {
        const int scale = tmac_q3_K_decode_scale(j, blk->scales);
        float dot_val = 0.0f, sum_a = 0.0f;
        tmac_q3_K_subblock_dot(blk, j, av, dot_val, sum_a);
        return __half2float(blk->d) * (float)scale * (dot_val - 4.0f * sum_a);
    } else if constexpr (QType == GGML_TYPE_Q6_K) {
        float dot_val = 0.0f, sum_a = 0.0f;
        tmac_q6_K_subblock_dot(blk, j, av, dot_val, sum_a);
        return __half2float(blk->d) * (float)blk->scales[j]
             * (dot_val - 32.0f * sum_a);
    } else if constexpr (QType == GGML_TYPE_IQ4_XS) {
        const int scale = tmac_iq4_xs_decode_scale(j, blk->scales_l, blk->scales_h);
        float dot_val = tmac_iq4_xs_subblock_dot(blk, j, av, ctx.lut_i8);
        return __half2float(blk->d) * (float)scale * dot_val;
    } else if constexpr (QType == GGML_TYPE_IQ3_S) {
        const int scale = tmac_iq3_s_decode_scale(j, blk->scales);
        float dot_val = tmac_iq3_s_subblock_dot(blk, j, av, ctx.lut_u32);
        return __half2float(blk->d) * (float)scale * dot_val;
    } else if constexpr (QType == GGML_TYPE_IQ2_XXS) {
        float dot_val = tmac_iq2_xxs_subblock_dot(blk, j, av, ctx.lut_f32);
        return __half2float(blk->d) * dot_val;
    } else if constexpr (QType == GGML_TYPE_IQ3_XXS) {
        float dot_val = tmac_iq3_xxs_subblock_dot(blk, j, av, ctx.lut_u32);
        return __half2float(blk->d) * dot_val;
    } else if constexpr (QType == GGML_TYPE_IQ2_XS) {
        float dot_val = tmac_iq2_xs_subblock_dot(blk, j, av, ctx.lut_u64);
        return __half2float(blk->d) * dot_val;
    } else if constexpr (QType == GGML_TYPE_IQ2_S) {
        float dot_val = tmac_iq2_s_subblock_dot(blk, j, av, ctx.lut_u64);
        return __half2float(blk->d) * dot_val;
    } else if constexpr (QType == GGML_TYPE_IQ1_M) {
        // d already applied inside subblock_dot (extracted from packed scales)
        return tmac_iq1_m_subblock_dot(blk, j, av, ctx.lut_u32);
    }
    return 0.0f;  // unreachable
}

// ═════════════════════════════════════════════════════════════════════
// Template kernel: Unfused Warp-per-Row GEMV
//
// Processes one output row per warp (8 rows per block of 256 threads).
// blockIdx.y provides MoE expert slot dispatch.
//
// Design choice — F32 dequant+FMA instead of dp4a (int8 dot product):
// At batch=1, GEMV is deeply memory-bound (VALU/MEM = 0.37x on RDNA3).
// The VALU unit is idle 63% of execution time waiting for VRAM data.
// dp4a would give 4x compute throughput per instruction, but this is
// invisible when ALU is already idle. T-MAC wins by skipping the
// F32→Q8_1 quantization pass that stock's dp4a path requires. At
// batch>=2 the kernel becomes compute-balanced and dp4a wins — T-MAC
// guards reject ne[1]!=1 for this reason (see can_dispatch()).
// ═════════════════════════════════════════════════════════════════════

template <ggml_type QType, bool HAS_BIAS = false>
__global__ void __launch_bounds__(256, tmac_quant_traits<QType>::unfused_min_blocks)
tmac_gemv_direct(
    float * __restrict__ dst,
    const void * __restrict__ src0,
    const float * __restrict__ src1,
    const int M,
    const int nb,                         // K / elements_per_block
    const int32_t * __restrict__ ids,     // MoE expert routing (nullptr = non-MoE)
    const int64_t expert_stride,          // bytes between experts in src0
    const int dst_expert_stride,          // floats between expert outputs in dst
    const int64_t src1_expert_stride,      // floats between expert activations in src1
    const float * __restrict__ bias       // bias vector (nullptr when HAS_BIAS=false)
) {
    using Traits = tmac_quant_traits<QType>;
    using Block  = typename Traits::block_type;

    const auto ctx = tmac_load_iq_lut<QType>();

    const int expert_slot = blockIdx.y;
    const int expert_id   = ids ? ids[expert_slot] : 0;
    const void  * src0_expert = (const char *)src0 + expert_id * expert_stride;
    const float * src1_expert = src1 + expert_slot * src1_expert_stride;
    float       * dst_expert  = dst + expert_slot * dst_expert_stride;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int m       = blockIdx.x * TMAC_ROWS_PER_BLOCK + warp_id;

    if (m >= M) return;

    const Block * row_w = (const Block *)src0_expert + (long long)m * (long long)nb;
    float thread_sum = 0.0f;
    float kahan_c    = 0.0f;  // Kahan compensation for numerical stability

    if constexpr (!Traits::is_sub_block_parallel) {
        for (int kb = lane; kb < nb; kb += WARP_SIZE) {
            // 32-element blocks (Q4_0, Q8_0, Q5_0, Q5_1): load 8 x float4 activations
            const float4 * av = (const float4 *)(src1_expert + kb * 32);
            const Block blk = row_w[kb];
            const float y = Traits::dot_scale(blk, av[0], av[1], av[2], av[3],
                                                   av[4], av[5], av[6], av[7]) - kahan_c;
            const float t = thread_sum + y;
            kahan_c = (t - thread_sum) - y;
            thread_sum = t;
        }
    } else {
        // Sub-block-level parallelization for 256-element super-blocks.
        // Distributes sub-blocks across warp lanes for full utilization.
        constexpr int SB_PER_BLK = Traits::sub_blocks_per_block;
        constexpr int SB_SIZE    = Traits::sub_block_size;
        constexpr int SB_SHIFT   = Traits::sub_block_shift;

        const int nb_sub = nb * SB_PER_BLK;
        for (int sb = lane; sb < nb_sub; sb += WARP_SIZE) {
            const int bidx = sb >> SB_SHIFT;
            const int j    = sb & (SB_PER_BLK - 1);
            const float4 * __restrict__ av =
                (const float4 *)(src1_expert + bidx * 256 + j * SB_SIZE);
            const float y = tmac_subblock_contribution<QType>(
                &row_w[bidx], j, av, ctx) - kahan_c;
            const float t = thread_sum + y;
            kahan_c = (t - thread_sum) - y;
            thread_sum = t;
        }
    }

    thread_sum = warp_reduce_sum_f32(thread_sum);
    if (lane == 0) {
        float result = thread_sum;
        if constexpr (HAS_BIAS) result += bias[m];
        dst_expert[m] = result;
    }
}

// ═════════════════════════════════════════════════════════════════════
// Template kernel: Fused SwiGLU Warp-per-Row GEMV
//
// Dual projection (up + gate) with shared activation loads and inline
// GLU activation. Halves activation bandwidth for FFN layers.
// ═════════════════════════════════════════════════════════════════════

template <ggml_type QType, ggml_glu_op glu_op>
__global__ void __launch_bounds__(256, tmac_quant_traits<QType>::fused_min_blocks)
tmac_gemv_fused_glu(
    float * __restrict__ dst,
    const void * __restrict__ src0_up,
    const void * __restrict__ src0_gate,
    const float * __restrict__ src1,
    const int M,
    const int nb,
    const int32_t * __restrict__ ids,
    const int64_t expert_stride,
    const int dst_expert_stride
) {
    using Traits = tmac_quant_traits<QType>;
    using Block  = typename Traits::block_type;

    const auto ctx = tmac_load_iq_lut<QType>();

    const int expert_slot = blockIdx.y;
    const int expert_id   = ids ? ids[expert_slot] : 0;
    const void  * up_expert   = (const char *)src0_up   + expert_id * expert_stride;
    const void  * gate_expert = (const char *)src0_gate  + expert_id * expert_stride;
    float       * dst_expert  = dst + expert_slot * dst_expert_stride;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int m       = blockIdx.x * TMAC_ROWS_PER_BLOCK + warp_id;

    if (m >= M) return;

    const Block * row_up   = (const Block *)up_expert   + (long long)m * (long long)nb;
    const Block * row_gate = (const Block *)gate_expert + (long long)m * (long long)nb;

    float sum_up = 0.0f, sum_gate = 0.0f;
    float c_up   = 0.0f, c_gate  = 0.0f;  // Kahan compensation (fixes Llama 4 accumulation drift)

    // Precondition: src1 is always the shared hidden state (ne11==1).
    // Fused SwiGLU only applies to UP+GATE projections which share one activation vector.
    // Per-expert activation offsets only apply to DOWN (handled by unfused path).
    if constexpr (!Traits::is_sub_block_parallel) {
        for (int kb = lane; kb < nb; kb += WARP_SIZE) {
            // 32-element blocks: load 8 x float4 activations, shared between up and gate
            const float4 * av = (const float4 *)(src1 + kb * 32);
            const float4 a0 = av[0], a1 = av[1], a2 = av[2], a3 = av[3];
            const float4 a4 = av[4], a5 = av[5], a6 = av[6], a7 = av[7];

            { const float y = Traits::dot_scale(row_up[kb], a0, a1, a2, a3, a4, a5, a6, a7) - c_up;
              const float t = sum_up + y; c_up = (t - sum_up) - y; sum_up = t; }
            { const float y = Traits::dot_scale(row_gate[kb], a0, a1, a2, a3, a4, a5, a6, a7) - c_gate;
              const float t = sum_gate + y; c_gate = (t - sum_gate) - y; sum_gate = t; }
        }
    } else {
        // Sub-block-level parallelization for 256-element super-blocks.
        // Both up and gate share activation loads, same as 32-element fused path.
        constexpr int SB_PER_BLK = Traits::sub_blocks_per_block;
        constexpr int SB_SIZE    = Traits::sub_block_size;
        constexpr int SB_SHIFT   = Traits::sub_block_shift;

        const int nb_sub = nb * SB_PER_BLK;
        for (int sb = lane; sb < nb_sub; sb += WARP_SIZE) {
            const int bidx = sb >> SB_SHIFT;
            const int j    = sb & (SB_PER_BLK - 1);
            const float4 * __restrict__ av =
                (const float4 *)(src1 + bidx * 256 + j * SB_SIZE);
            { const float y = tmac_subblock_contribution<QType>(
                  &row_up[bidx], j, av, ctx) - c_up;
              const float t = sum_up + y; c_up = (t - sum_up) - y; sum_up = t; }
            { const float y = tmac_subblock_contribution<QType>(
                  &row_gate[bidx], j, av, ctx) - c_gate;
              const float t = sum_gate + y; c_gate = (t - sum_gate) - y; sum_gate = t; }
        }
    }

    sum_up   = warp_reduce_sum_f32(sum_up);
    sum_gate = warp_reduce_sum_f32(sum_gate);

    if (lane == 0) {
        float gate_act;
        if constexpr (glu_op == GGML_GLU_OP_SWIGLU || glu_op == GGML_GLU_OP_SWIGLU_OAI) {
            gate_act = sum_gate / (1.0f + expf(-sum_gate));
        } else if constexpr (glu_op == GGML_GLU_OP_GEGLU || glu_op == GGML_GLU_OP_GEGLU_ERF) {
            gate_act = 0.5f * sum_gate * (1.0f + erff(sum_gate * 0.7071067811865475f));
        } else if constexpr (glu_op == GGML_GLU_OP_GEGLU_QUICK) {
            gate_act = sum_gate / (1.0f + expf(-1.702f * sum_gate));
        } else if constexpr (glu_op == GGML_GLU_OP_REGLU) {
            gate_act = fmaxf(sum_gate, 0.0f);
        }
        dst_expert[m] = sum_up * gate_act;
    }
}

// ═════════════════════════════════════════════════════════════════════
// Fused kernel launch helper — eliminates glu_op switch duplication
// ═════════════════════════════════════════════════════════════════════

template <ggml_type QType>
static void tmac_launch_fused(
    float * dst, const void * src0_up, const void * src0_gate,
    const float * src1, int M, int nb, ggml_glu_op glu_op,
    cudaStream_t stream, const int32_t * ids, int64_t expert_stride,
    int n_expert_used
) {
    dim3 block(256);
    dim3 grid((M + TMAC_ROWS_PER_BLOCK - 1) / TMAC_ROWS_PER_BLOCK, n_expert_used);

    constexpr int smem = tmac_quant_traits<QType>::smem_bytes;

    #define TMAC_LAUNCH_FUSED_GLU(OP) \
        tmac_gemv_fused_glu<QType, OP><<<grid, block, smem, stream>>>( \
            dst, src0_up, src0_gate, src1, M, nb, ids, expert_stride, M)

    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:      TMAC_LAUNCH_FUSED_GLU(GGML_GLU_OP_SWIGLU); break;
        case GGML_GLU_OP_SWIGLU_OAI:  TMAC_LAUNCH_FUSED_GLU(GGML_GLU_OP_SWIGLU_OAI); break;
        case GGML_GLU_OP_GEGLU:       TMAC_LAUNCH_FUSED_GLU(GGML_GLU_OP_GEGLU); break;
        case GGML_GLU_OP_GEGLU_ERF:   TMAC_LAUNCH_FUSED_GLU(GGML_GLU_OP_GEGLU_ERF); break;
        case GGML_GLU_OP_GEGLU_QUICK: TMAC_LAUNCH_FUSED_GLU(GGML_GLU_OP_GEGLU_QUICK); break;
        case GGML_GLU_OP_REGLU:       TMAC_LAUNCH_FUSED_GLU(GGML_GLU_OP_REGLU); break;
        default: GGML_ABORT("T-MAC fused: unsupported GLU op %d", (int)glu_op);
    }

    #undef TMAC_LAUNCH_FUSED_GLU
}


// ═════════════════════════════════════════════════════════════════════
// Template launcher for unfused (simple) GEMV — eliminates copy-paste
// host wrappers that caused the IQ2_XXS grid/block bug.
// Analogous to tmac_launch_fused<QType>() above for the fused path.
// ═════════════════════════════════════════════════════════════════════

template <ggml_type QType>
static void tmac_launch_simple(
    float * dst, const void * src0, const float * src1,
    int M, int nb,
    cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride,
    int n_expert_used, int64_t src1_expert_stride,
    const float * bias
) {
    constexpr int smem = tmac_quant_traits<QType>::smem_bytes;
    dim3 block(256);
    dim3 grid((M + TMAC_ROWS_PER_BLOCK - 1) / TMAC_ROWS_PER_BLOCK, n_expert_used);

    if (bias) {
        tmac_gemv_direct<QType, true><<<grid, block, smem, stream>>>(
            dst, src0, src1, M, nb,
            ids, expert_stride, /*dst_expert_stride=*/M, src1_expert_stride, bias);
    } else {
        tmac_gemv_direct<QType><<<grid, block, smem, stream>>>(
            dst, src0, src1, M, nb,
            ids, expert_stride, /*dst_expert_stride=*/M, src1_expert_stride, nullptr);
    }
}

// ═════════════════════════════════════════════════════════════════════
// Host entry points — ABI-stable wrappers (signatures match tmac.cuh)
// Each wrapper computes M/K/nb from tensor dimensions, then delegates
// to tmac_launch_simple<QType>() or tmac_launch_fused<QType>().
// ═════════════════════════════════════════════════════════════════════

// ── Q4_0 ──

void ggml_cuda_tmac_q4_0_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q4_0>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q4_0_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q4_0>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── Q8_0 ──

void ggml_cuda_tmac_q8_0_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q8_0>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q8_0_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q8_0>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── Q5_1 ──

void ggml_cuda_tmac_q5_1_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q5_1>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q5_1_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q5_1>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── Q5_0 ──

void ggml_cuda_tmac_q5_0_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q5_0>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q5_0_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q5_0>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── Q4_K ── (256-element super-blocks, nb = K / 256)

void ggml_cuda_tmac_q4_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256, NOT 32

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q4_K>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q4_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256, NOT 32

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q4_K>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── Q5_K ── (256-element super-blocks, nb = K / 256)

void ggml_cuda_tmac_q5_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q5_K>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q5_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q5_K>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── Q6_K ── (256-element super-blocks, nb = K / 256)

void ggml_cuda_tmac_q6_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q6_K>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q6_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q6_K>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ4_XS ── (256-element super-blocks, LUT-indexed 4-bit quants)

void ggml_cuda_tmac_iq4_xs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_IQ4_XS>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq4_xs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_IQ4_XS>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ3_S ── (256-element super-blocks, LUT-indexed 3-bit quants with signs)

void ggml_cuda_tmac_iq3_s_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_IQ3_S>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq3_s_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_IQ3_S>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ2_XXS host wrappers ──────────────────────────────────────────────────
// IQ2_XXS: 256-element super-blocks (8 sub-blocks × 32), grid-LUT dequant
// (iq2xxs_grid[256] uint64 → 2048 bytes shared memory)

void ggml_cuda_tmac_iq2_xxs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used,
    int64_t src1_expert_stride, const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256
    if (M == 0 || K == 0) return;
    tmac_launch_simple<GGML_TYPE_IQ2_XXS>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq2_xxs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_IQ2_XXS>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ3_XXS ──

void ggml_cuda_tmac_iq3_xxs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used,
    int64_t src1_expert_stride, const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256
    if (M == 0 || K == 0) return;
    tmac_launch_simple<GGML_TYPE_IQ3_XXS>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq3_xxs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;
    if (M == 0 || K == 0) return;
    tmac_launch_fused<GGML_TYPE_IQ3_XXS>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ2_XS ──

void ggml_cuda_tmac_iq2_xs_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used,
    int64_t src1_expert_stride, const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256
    if (M == 0 || K == 0) return;
    tmac_launch_simple<GGML_TYPE_IQ2_XS>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq2_xs_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;
    if (M == 0 || K == 0) return;
    tmac_launch_fused<GGML_TYPE_IQ2_XS>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ2_S ──

void ggml_cuda_tmac_iq2_s_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used,
    int64_t src1_expert_stride, const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256
    if (M == 0 || K == 0) return;
    tmac_launch_simple<GGML_TYPE_IQ2_S>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq2_s_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;
    if (M == 0 || K == 0) return;
    tmac_launch_fused<GGML_TYPE_IQ2_S>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ1_M ──

void ggml_cuda_tmac_iq1_m_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used,
    int64_t src1_expert_stride, const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256
    if (M == 0 || K == 0) return;
    tmac_launch_simple<GGML_TYPE_IQ1_M>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq1_m_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;
    if (M == 0 || K == 0) return;
    tmac_launch_fused<GGML_TYPE_IQ1_M>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── MXFP4 ──

void ggml_cuda_tmac_mxfp4_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_MXFP4>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_mxfp4_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_MXFP4>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── IQ4_NL ──

void ggml_cuda_tmac_iq4_nl_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_IQ4_NL>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_iq4_nl_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 32;

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_IQ4_NL>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}

// ── Q3_K ──

void ggml_cuda_tmac_q3_K_simple(
    ggml_backend_cuda_context & ctx,
    const void * src0, const float * src1, float * dst,
    int64_t ne00, int64_t ne0,
    cudaStream_t stream,
    const int32_t * ids,
    int64_t expert_stride,
    int n_expert_used,
    int64_t src1_expert_stride,
    const float * bias
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_simple<GGML_TYPE_Q3_K>(
        dst, src0, src1, M, nb, stream,
        ids, expert_stride, n_expert_used, src1_expert_stride, bias);
}

void ggml_cuda_tmac_q3_K_fused(
    ggml_backend_cuda_context & ctx,
    const void * src0_up, const void * src0_gate,
    const float * src1, float * dst,
    int64_t ne00, int64_t ne0, ggml_glu_op glu_op, cudaStream_t stream,
    const int32_t * ids, int64_t expert_stride, int n_expert_used
) {
    GGML_UNUSED(ctx);
    const int K  = (int)ne00;
    const int M  = (int)ne0;
    const int nb = K / 256;  // QK_K = 256

    if (M == 0 || K == 0) return;

    tmac_launch_fused<GGML_TYPE_Q3_K>(
        dst, src0_up, src0_gate, src1, M, nb, glu_op, stream,
        ids, expert_stride, n_expert_used);
}
