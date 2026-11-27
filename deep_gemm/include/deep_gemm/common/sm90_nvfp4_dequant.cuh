#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/compile.cuh>

// NVFP4 weights for BF16 RS-WGMMA, in the layout of `deep_gemm/mega/nvfp4_sm90.py`
// sm_90 has no E2M1 or E4M3 to BF16 `cvt`; both are placed into BF16 bits, and their product is exact in BF16
// Decoded weights are `e2m1 * e4m3 * 2 ^ -13` (`2 ^ -126` from the code placement times `2 ^ 113` from the scale's), the per-expert
// global scales carry `2 ^ 13` (`GLOBAL_SCALE_LOG2` in `deep_gemm/mega/nvfp4_sm90.py`)
namespace deep_gemm::sm90_nvfp4 {

// Byte offset in a stage of the 16-byte scale unit of rows `row` and `row + 8` (`row % 16 < 8`)
CUTLASS_HOST_DEVICE constexpr uint32_t get_sf_smem_offset(const uint32_t& row) {
    return row / 16 * 128 + row % 16 * 16;
}

#if defined(DG_IN_CUDA_COMPILATION)

CUTLASS_DEVICE uint32_t rotr(const uint32_t& x, const uint32_t& n) {
    return __funnelshift_r(x, x, n);
}

// The 8 codes of a word, each at BF16 bits 15 and 8..6 (`e2m1 * 2 ^ -126`): columns `2q + {0, 1}` and `2q + 8 + {0, 1}`
// of k16 step `2w`, then of step `2w + 1`
CUTLASS_DEVICE void decode_codes(const uint32_t& word, uint32_t (&codes)[4]) {
    constexpr uint32_t kMask = 0x81c081c0u;
    codes[0] = word & kMask;
    codes[1] = rotr(word, 4) & kMask;
    codes[2] = (word << 6) & kMask;
    codes[3] = ((word << 2) & 0x00400040u) | (rotr(word, 6) & 0x81808180u);
}

// The BF16 scales `e4m3 * 2 ^ 113` of rows `r` (low half) and `r + 8` (high half): `(e + 16) << 3 | m` at bits 11..4
CUTLASS_DEVICE uint32_t decode_sf(const uint32_t& word) {
    // One LOP3 with the exponent bits in a register (two immediates would take two)
    uint32_t sf;
    asm("lop3.b32 %0, %1, 0x0ff00ff0, %2, 0xea;" : "=r"(sf) : "r"(word), "r"(0x70007000u));
    return sf;
}

// `codes * scale`, with the scale broadcast from half `kHalf` of `sf`
template <uint32_t kHalf>
CUTLASS_DEVICE uint32_t scale_codes(const uint32_t& codes, const uint32_t& sf) {
    uint32_t d;
    if constexpr (kHalf == 0) {
        asm("{\n\t.reg .b16 lo, hi;\n\t.reg .b32 s;\n\tmov.b32 {lo, hi}, %2;\n\tmov.b32 s, {lo, lo};\n\tmul.rn.bf16x2 %0, %1, s;\n\t}"
            : "=r"(d) : "r"(codes), "r"(sf));
    } else {
        asm("{\n\t.reg .b16 lo, hi;\n\t.reg .b32 s;\n\tmov.b32 {lo, hi}, %2;\n\tmov.b32 s, {hi, hi};\n\tmul.rn.bf16x2 %0, %1, s;\n\t}"
            : "=r"(d) : "r"(codes), "r"(sf));
    }
    return d;
}

// The BF16 A fragments of wgmma m64nNk16 for k16 steps `2w` and `2w + 1`, from word `w` of the codes of rows `r` and `r + 8`
// and of their scale unit
CUTLASS_DEVICE void decode_k32(const uint32_t& codes_r, const uint32_t& codes_r8, const uint32_t& sf, uint32_t (&a)[2][4]) {
    uint32_t codes[2][4];
    decode_codes(codes_r, codes[0]);
    decode_codes(codes_r8, codes[1]);
    const uint32_t scales[2] = {decode_sf(sf), decode_sf(rotr(sf, 8))};
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++ i) {
        a[i][0] = scale_codes<0>(codes[0][2 * i], scales[i]);
        a[i][1] = scale_codes<1>(codes[1][2 * i], scales[i]);
        a[i][2] = scale_codes<0>(codes[0][2 * i + 1], scales[i]);
        a[i][3] = scale_codes<1>(codes[1][2 * i + 1], scales[i]);
    }
}

#endif

} // namespace deep_gemm::sm90_nvfp4
