#pragma once

#include <cutlass/arch/memory_sm80.h>
#include <math_constants.h>
#include <deep_gemm/layout/sm90_sparse_mqa_logits.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

template <uint32_t PAGE_KV, uint32_t SPARSE_BLOCK_KV>
CUTLASS_GLOBAL void sm90_paged_sparse_mqa_logits_metadata(
    const int num_rows, const int num_sparse_blocks, const int table_width,
    const int64_t table_stride, const int* context_lens, const int* block_table,
    const int* sparse_indices, const int* valid_sparse_tokens,
    layout::sm90_sparse_mqa_logits::Block* metadata
) {
    using layout::sm90_sparse_mqa_logits::Block;
    const int64_t block_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block_index >= static_cast<int64_t>(num_rows) * num_sparse_blocks)
        return;
    const int row = block_index / num_sparse_blocks;
    const int sparse_block = block_index % num_sparse_blocks;
    const int64_t kv_position = static_cast<int64_t>(sparse_indices[block_index]) * SPARSE_BLOCK_KV;
    const int64_t remaining_candidates = static_cast<int64_t>(valid_sparse_tokens[row]) - sparse_block * SPARSE_BLOCK_KV;
    const int64_t remaining_context = static_cast<int64_t>(context_lens[row]) - kv_position;
    const int64_t num_tokens = min(static_cast<int64_t>(SPARSE_BLOCK_KV),
                                   min(remaining_candidates, remaining_context));
    Block block = {0, 0, 0};
    if (num_tokens > 0 and kv_position >= 0 and kv_position / PAGE_KV < table_width) {
        const int page = block_table[static_cast<int64_t>(row) * table_stride + kv_position / PAGE_KV];
        if (page >= 0)
            block = {page, static_cast<uint16_t>(kv_position % PAGE_KV), static_cast<uint16_t>(num_tokens)};
    }
    auto row_metadata = metadata + static_cast<int64_t>(row) * (num_sparse_blocks + 1);
    row_metadata[sparse_block] = block;
    if (sparse_block == 0) {
        const int valid_width = max(0, min(valid_sparse_tokens[row],
                                           num_sparse_blocks * static_cast<int>(SPARSE_BLOCK_KV)));
        row_metadata[num_sparse_blocks] = {valid_width, 0, 0};
    }
}

template <uint32_t PAGE_KV, uint32_t SPARSE_BLOCK_KV, bool kAlignedCache>
CUTLASS_GLOBAL __launch_bounds__(layout::sm90_sparse_mqa_logits::kNumThreads)
void sm90_fp8_paged_sparse_mqa_logits(
    const uint8_t* q, const uint8_t* cache, const float* weights,
    const layout::sm90_sparse_mqa_logits::Block* metadata, float* logits,
    const int num_sparse_blocks, const int num_pages,
    const int64_t page_stride, const int64_t weight_stride, const int logits_stride
) {
    using namespace layout::sm90_sparse_mqa_logits;
    constexpr int kVectorBytes = 16;
    constexpr int kChunksPerToken = kHeadDim / kVectorBytes;
    using WGMMA = typename mma::sm90::FP8MMASelector<kNumHeads>::type;
    const int logits_width = num_sparse_blocks * SPARSE_BLOCK_KV;
    const int num_tiles = (logits_width + kTileTokens - 1) / kTileTokens;
    const int row = blockIdx.x / num_tiles;
    const int tile_start = (blockIdx.x % num_tiles) * kTileTokens;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const auto row_metadata = metadata + static_cast<int64_t>(row) * (num_sparse_blocks + 1);
    const int valid_width = row_metadata[num_sparse_blocks].page;
    if (tile_start >= valid_width) {
        if (tid < kTileTokens and tile_start + tid < logits_width)
            logits[static_cast<int64_t>(row) * logits_stride + tile_start + tid] = -CUDART_INF_F;
        return;
    }

    struct Shared {
        alignas(1024) uint8_t q[kNumHeads * kHeadDim];
        alignas(1024) uint8_t k[kTileTokens * kHeadDim];
        float scales[kTileTokens];
        int valid[kTileTokens];
        int64_t k_offsets[kTileTokens];
    };
    __shared__ Shared smem;

    bool is_valid_token = false;
    if (tid < kTileTokens) {
        const int col = tile_start + tid;
        const auto block = col < logits_width ? row_metadata[col / SPARSE_BLOCK_KV] : Block{0, 0, 0};
        is_valid_token = col < logits_width and col % SPARSE_BLOCK_KV < block.num_tokens and
                         block.page >= 0 and block.page < num_pages;
        const int64_t page_offset = static_cast<int64_t>(block.page) * page_stride;
        const auto token_offset = block.token_offset + col % SPARSE_BLOCK_KV;
        smem.valid[tid] = is_valid_token;
        smem.k_offsets[tid] = is_valid_token ? page_offset + token_offset * kHeadDim : 0;
        // A page stores all FP8 K vectors first, followed by one FP32 scale per token.
        smem.scales[tid] = is_valid_token ? *reinterpret_cast<const float*>(
            cache + page_offset + PAGE_KV * kHeadDim + token_offset * sizeof(float)) : 0.0f;
    }
    if (not __syncthreads_or(is_valid_token)) {
        if (tid < kTileTokens and tile_start + tid < logits_width)
            logits[static_cast<int64_t>(row) * logits_stride + tile_start + tid] = -CUDART_INF_F;
        return;
    }

    // WGMMA's 128-byte swizzle, with each thread moving a 16-byte segment.
    #pragma unroll
    for (int i = tid; i < kNumHeads * kChunksPerToken; i += kNumThreads) {
        const int head = i / kChunksPerToken;
        const int chunk = i % kChunksPerToken;
        const auto src = q + static_cast<int64_t>(row) * kNumHeads * kHeadDim + i * kVectorBytes;
        auto dst = smem.q + head * kHeadDim + (chunk ^ (head % kChunksPerToken)) * kVectorBytes;
        ptx::cp_async_cg<64>(reinterpret_cast<const uint4*>(src), reinterpret_cast<uint4*>(dst));
    }
    #pragma unroll
    for (int i = tid; i < kTileTokens * kChunksPerToken; i += kNumThreads) {
        const int token = i / kChunksPerToken;
        const int chunk = i % kChunksPerToken;
        const bool valid = smem.valid[token];
        const auto src = cache + smem.k_offsets[token] + chunk * kVectorBytes;
        auto dst = reinterpret_cast<uint4*>(smem.k + token * kHeadDim +
                                            (chunk ^ (token % kChunksPerToken)) * kVectorBytes);
        if constexpr (kAlignedCache) {
            ptx::cp_async_cg_zfill<64>(reinterpret_cast<const uint4*>(src), dst, valid ? kVectorBytes : 0);
        } else {
            const auto words = reinterpret_cast<const uint32_t*>(src);
            *dst = valid ? make_uint4(words[0], words[1], words[2], words[3]) : make_uint4(0, 0, 0, 0);
        }
    }
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    float accum[WGMMA::kNumAccum];
    #pragma unroll
    for (int i = 0; i < WGMMA::kNumAccum; ++i)
        accum[i] = 0;
    #pragma unroll
    for (int i = 0; i < WGMMA::kNumAccum; ++i)
        ptx::warpgroup_fence_operand(accum[i]);
    ptx::warpgroup_arrive();
    #pragma unroll
    for (int k = 0; k < kHeadDim; k += WGMMA::K) {
        const auto a = mma::sm90::make_smem_desc(smem.k + k, mma::sm90::to_swizzle_cute_type<kHeadDim>());
        const auto b = mma::sm90::make_smem_desc(smem.q + k, mma::sm90::to_swizzle_cute_type<kHeadDim>());
        WGMMA::wgmma(a, b, accum, k != 0);
    }
    ptx::warpgroup_commit_batch();
    ptx::warpgroup_wait<0>();
    #pragma unroll
    for (int i = 0; i < WGMMA::kNumAccum; ++i)
        ptx::warpgroup_fence_operand(accum[i]);

    // Each group of four lanes reduces all heads for two tokens eight rows apart.
    float head_weights[kNumHeads / 4];
    #pragma unroll
    for (int j = 0; j < kNumHeads / 4; ++j)
        head_weights[j] = weights[static_cast<int64_t>(row) * weight_stride + (j / 2) * 8 + (j & 1) + (lane % 4) * 2];
    float weighted_sums[4] = {};
    #pragma unroll
    for (int j = 0; j < kNumHeads / 8; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            weighted_sums[k] += fmaxf(accum[j * 4 + k], 0.0f) * head_weights[j * 2 + (k & 1)];
    }
    const int token = (tid / 32) * 16 + lane / 4;
    float score0 = (weighted_sums[0] + weighted_sums[1]) * smem.scales[token];
    float score1 = (weighted_sums[2] + weighted_sums[3]) * smem.scales[token + 8];
    #pragma unroll
    for (int delta = 1; delta <= 2; delta *= 2) {
        score0 += __shfl_xor_sync(0xffffffff, score0, delta);
        score1 += __shfl_xor_sync(0xffffffff, score1, delta);
    }
    if (lane % 4 == 0) {
        const int64_t logit_offset = static_cast<int64_t>(row) * logits_stride + tile_start + token;
        if (tile_start + token < logits_width)
            logits[logit_offset] = smem.valid[token] ? score0 : -CUDART_INF_F;
        if (tile_start + token + 8 < logits_width)
            logits[logit_offset + 8] = smem.valid[token + 8] ? score1 : -CUDART_INF_F;
    }
}

} // namespace deep_gemm
