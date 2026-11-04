#pragma once

#include <cstdint>

namespace deep_gemm::layout::sm90_sparse_mqa_logits {

constexpr int kNumHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kTileTokens = 64;
constexpr int kNumThreads = 128;

// Each row has one record per sparse block, followed by a record whose page
// field stores the row's valid logit width. The remaining fields are zero.
struct Block {
    int32_t page;
    uint16_t token_offset;
    uint16_t num_tokens;
};
static_assert(sizeof(Block) == sizeof(int64_t));

} // namespace deep_gemm::layout::sm90_sparse_mqa_logits
