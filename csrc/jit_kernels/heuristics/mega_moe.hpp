#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"
#include "sm90.hpp"

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m_l1, store_block_m_l2;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Ring capacity and SF ring token count
    int num_ring_tokens;
    int num_sf_ring_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    // SM90 only: each math warpgroup computes whole tasks
    bool pingpong = false;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m_l1=" << config.store_block_m_l1 << ", store_block_m_l2=" << config.store_block_m_l2
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull
           << ", pingpong=" << config.pingpong << ")";
        return os;
    }
};

// We need to distribute work across locality domains, so we must use the whole GPU
static int get_num_sms_for_mega_moe() {
    const auto num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms == jit->device.get_num_sms());
    return num_sms;
}

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16" or mma_type_str == "bf16xnvfp4")
        return MmaKind::BF16;
    if (mma_type_str == "fp4xfp4")
        return MmaKind::NVFP4;
    DG_HOST_ASSERT(mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8");
    return MmaKind::MXFP8FP4;
}

static int get_num_mma_elem_bytes(const MmaKind& mma_kind) {
    DG_HOST_ASSERT(mma_kind != MmaKind::NVFP4 and "NVFP4 elements are sub-byte, use `get_element_bits`");
    return mma_kind == MmaKind::BF16 ? 2 : 1;
}

static bool is_mma_with_sf(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP8FP4 or mma_kind == MmaKind::NVFP4;
}

// SF group size along K: MX recipes use 32, NVFP4 uses 16
static int get_mma_sf_gran_k(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::NVFP4 ? 16 : 32;
}

// SM90 NVFP4 runs ping-pong up to BLOCK_M 64 at any expert count, with one 64-row tile per math warpgroup task:
// this halves the task size for the load balance at decode, if the L1 N blocks still fit the 64-bit L2 readiness mask
static int get_block_n_for_mega_moe(const bool& routed_nvfp4, const int& block_m, const int& intermediate_hidden) {
    return routed_nvfp4 and block_m <= 64 and intermediate_hidden * 2 / 64 <= 64 ? 64 : 128;
}

// SM90 NVFP4 decode at one rank runs 3 math warpgroups. With ping-pong, only when this launch cannot reuse the ring (see the kernel's
// `kNumRingBlocks` assert): its pool has at most one partial block per expert next to the full ones. Cooperative, they take BLOCK_N 192,
// whose fewer and longer tasks pay off from `kNumMinCoopL1Waves` waves of L1 tasks for the experts `num_tile_tokens` (which picks the
// tiles) can route to, and the ring must hold the live pool blocks at BLOCK_N 192
static int get_num_sm90_math_warpgroups(const int& num_ranks, const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
                                        const int& block_m, const int& num_ring_tokens, const bool& routed_nvfp4, const bool& pingpong,
                                        const int& num_tile_tokens, const int& hidden, const int& intermediate_hidden) {
    if (not routed_nvfp4 or num_ranks != 1 or num_experts_per_rank > 128)
        return 2;
    const int num_routed_tokens = num_tokens * std::min(num_topk, num_experts_per_rank);
    const int num_max_pool_blocks = ceil_div(num_routed_tokens, block_m) + std::min(num_experts_per_rank, num_routed_tokens);
    if (pingpong)
        return num_max_pool_blocks <= num_ring_tokens / block_m ? 3 : 2;
    if (block_m > 32 or hidden % 384 != 0 or intermediate_hidden % 192 != 0)
        return 2;
    constexpr int kNumMinCoopL1Waves = 2;
    const int num_sms = get_num_sms_for_mega_moe();
    const int num_expected_l1_tasks = std::min(num_experts_per_rank, num_tile_tokens * num_topk) * (intermediate_hidden * 2 / 192);
    const int num_live_pool_blocks = sched::get_num_max_live_pool_blocks(num_max_pool_blocks, num_sms, hidden, intermediate_hidden, 1, 192);
    return num_expected_l1_tasks >= kNumMinCoopL1Waves * num_sms and num_live_pool_blocks <= num_ring_tokens / block_m ? 3 : 2;
}

static std::tuple<int, int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind,
    const bool& routed_fp8, const bool& routed_nvfp4) {
    // Expected tokens per expert, plus a one-sigma routing margin for the tile choice
    const float num_expected_tokens = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);

    // SM90: at most 128 rows per block, or 64 with more than 64 local experts unless the routed weights are FP8 or NVFP4
    if (jit->device.get_arch_major() == 9) {
        DG_HOST_ASSERT(routed_nvfp4 ? mma_kind == MmaKind::BF16 : mma_kind == MmaKind::MXFP8FP4);
        // NVFP4 blocks load and multiply only their valid tokens, so they are sized for the hot experts, which hold most tokens of a
        // decode batch: a block then covers a whole expert, whose weights are decoded once per K block
        // NOTES: measured at one rank with 128 experts (Inkling-2-Mini); 96 local experts (Inkling-2-Small TP-local) lose at BLOCK_M 128
        if (routed_nvfp4 and num_ranks == 1 and num_experts >= 128) {
            const int block_m = num_expected_tokens <= 1 ? 16 : num_expected_tokens <= 12 ? 32 : num_expected_tokens <= 16 ? 64 : 128;
            return {2, block_m, block_m, std::min(block_m, 32), 128, 2 * 128};
        }
        int block_m = num_expected_tokens <= 10 ? 16 : 32;
        // BLOCK_M 128 with tokens on M pays off from about 56 tokens per expert
        constexpr int kNumFP8BlockM128Tokens = 56;
        if (num_expected_tokens > 24)
            block_m = (num_covered_tokens <= (routed_fp8 ? kNumFP8BlockM128Tokens : routed_nvfp4 ? 64 : 96) or
                       (num_experts / num_ranks > 64 and not routed_fp8 and not routed_nvfp4)) ? 64 : 128;
        return {2, block_m, block_m, std::min(block_m, 32), 128, 2 * 128};
    }

    // Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
    // per expert and the smallest tile covering them: one block up to 192 rows, then blocks of up to 240 rows.
    // A single 240-row block per expert only pays off with many local experts (about 14 or more blocks per rank).
    int block_m = num_expected_tokens <= 10 ? 16 : 32;
    if (num_expected_tokens > 24) {
        int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
        if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
            num_blocks = 2;
        for (const int& candidate: {64, 128, 192, 240}) {
            block_m = candidate;
            if (num_blocks * candidate >= num_covered_tokens)
                break;
        }
    }
    // B300 EP4 tuning for NVFP4 decode: a wider tile tolerates skewed routing
    // without splitting a hot expert into additional weight-loading tasks.
    // Keep the public tile query and kernel launch on the same selection path.
    if (mma_kind == MmaKind::NVFP4 and num_ranks == 4 and num_experts == 256 and num_topk == 8 and
        num_tokens >= 32 and num_tokens <= 64 and
        jit->device.get_arch_major() == 10 and jit->device.get_arch_minor() == 3 and
        runtime->get_num_sms() == 148)
        block_m = 64;

    // Use smaller, asymmetric store blocks at FP8xFP4 prefill to save shared memory for sufficient TMA pipeline depth
    const int store_block_m_l1 = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 :
                                 mma_kind == MmaKind::MXFP8FP4 ? 24 : 40;
    const int store_block_m_l2 = (block_m == 240 and mma_kind == MmaKind::MXFP8FP4) ? 8 : store_block_m_l1;

    // 128 bytes of K per tile, in elements -- NVFP4 packs two elements per byte
    const int block_k = 1024 / get_element_bits(mma_kind);

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    // Return configs: 2-CTA clusters and 2 epilogue warpgroups
    return {2, block_m, store_block_m_l1, store_block_m_l2, block_k, 2 * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, 
    const int& num_bytes_per_pull, const int& store_block_m_l1, const int& store_block_m_l2,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    const int num_mma_elem_bits = get_element_bits(mma_kind);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 output staging and L2 BF16 staging.
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m_l1 * ((block_n / 2) * num_mma_elem_bits / 8) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Schedule task payloads
    constexpr int kNumScheduleStages = 2;
    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    // + schedule task publish full/empty barriers.
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;

    // Amax warp-pair reduction buffer for SwiGLU's cross-warp amax exchange.
    // NOTES: NVFP4 does not need it, as one warp covers exactly one 16-element SF group
    const int smem_amax_reduction = mma_kind == MmaKind::MXFP8FP4 ?
        store_block_m_l1 * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = is_mma_with_sf(mma_kind) ? sf_block_m * (block_k / gran_k) : 0;
    const int smem_sfb_per_stage = is_mma_with_sf(mma_kind) ? sf_block_n * (block_k / gran_k) : 0;

    // Per-stage: A tile + B tile + optional SF tiles + full/empty barriers.
    // NOTES: sizes use `num_mma_elem_bits / 8` (not bytes) to support sub-byte NVFP4 elements
    const int smem_a_size_per_stage = load_block_m * block_k * num_mma_elem_bits / 8;
    const int smem_b_size_per_stage = block_n * block_k * num_mma_elem_bits / 8;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers +
        smem_task_info + smem_tmem_ptr;

    // Select maximum number of stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static std::pair<int, int> get_sm90_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_bytes_per_pull, const int& store_block_m_l1, const int& store_block_m_l2,
    const int& sf_block_m, const int& sf_block_n,
    const int& num_dispatch_warps, const int& num_math_warps,
    const bool& with_shared, const bool& routed_fp8, const bool& routed_nvfp4, const bool& pingpong) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumAmaxSlots = 2;

    // With ping-pong, every math warpgroup has its own output staging, amax exchange slots and schedule stage
    const int num_pipelines = pingpong ? num_math_warps / 4 : 1;
    const int num_schedule_stages = pingpong ? num_pipelines : 2;

    // Fixed regions, in the kernel's `SharedStorage` order
    const int smem_expert_count_size = align(num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_cd_l1 = store_block_m_l1 * (block_n / 2) * (routed_nvfp4 ? static_cast<int>(sizeof(nv_bfloat16)) : 1);
    const int smem_cd_l2 = store_block_m_l2 * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = num_pipelines * align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);
    const int smem_amax = (num_math_warps + num_pipelines * kNumAmaxSlots) * block_m * static_cast<int>(sizeof(float));
    const int smem_task_info = num_schedule_stages * static_cast<int>(sizeof(sched::TaskInfo<true>));
    const int smem_barriers = (num_dispatch_warps + num_pipelines * kNumAmaxSlots + num_math_warps * 2 + num_schedule_stages * 2) * 8;

    // Per stage: tokens (FP8, or BF16 without SF), weights (packed FP4, or FP8 with shared experts or FP8 routed weights), SFA, SFB,
    // and the full/empty barriers; NVFP4 weights have 2 SF words (8 E4M3 scales) per row and K block
    const int smem_a_size_per_stage = block_m * block_k * (routed_nvfp4 ? static_cast<int>(sizeof(nv_bfloat16)) : 1);
    const int smem_b_size_per_stage = block_n * block_k / (with_shared or routed_fp8 ? 1 : 2);
    const int smem_sfb_size_per_stage = routed_fp8 ? static_cast<int>(layout::kNumBlockSFStageWords) : routed_nvfp4 ? 2 * block_n : sf_block_n;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage +
        (sf_block_m + smem_sfb_size_per_stage) * static_cast<int>(sizeof(uint32_t)) + 2 * 8;
    const int smem_fixed = smem_expert_count_size + smem_send_buffers_size + smem_cd + smem_amax + smem_task_info + smem_barriers;

    // Select maximum number of stages, with 3 pipelines a multiple of 3 so that their rings are equal
    // The kernel's `SharedStorage` is padded to 1024 bytes
    int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    if (num_pipelines == 3)
        num_stages -= num_stages % 3;
    DG_HOST_ASSERT(num_stages >= 2);
    return {num_stages, align(smem_fixed + num_stages * smem_size_per_stage, kSmemAlignment)};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind,
    const int& num_shared_experts = 0,
    const bool& routed_fp8 = false,
    const bool& routed_nvfp4 = false,
    const int& num_launch_tokens = 0) {
    const bool is_sm90 = jit->device.get_arch_major() == 9;

    // Block config
    // `num_tokens` picks the tiles, `num_launch_tokens` (SM90, the tokens of this launch) bounds its pool blocks for the math warpgroups
    const auto [cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_block_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind, routed_fp8, routed_nvfp4);
    DG_HOST_ASSERT(num_ring_tokens % block_m == 0);

    // SM90 ping-pong: each math warpgroup computes whole tasks, so its epilogue overlaps the other's mainloop
    // The 128-row accumulators fit only with at most 64 tokens per block and the registers of at most 64 local experts; NVFP4 tasks are
    // one 64-row tile
    const int block_n_2wg = get_block_n_for_mega_moe(routed_nvfp4, block_m, intermediate_hidden);
    const bool pingpong = is_sm90 and block_m <= 64 and (routed_nvfp4 ? block_n_2wg == 64 : num_experts_per_rank <= 64);
    const int num_math_warpgroups = is_sm90 ? get_num_sm90_math_warpgroups(
        num_ranks, num_experts_per_rank, num_launch_tokens, num_topk, block_m, num_ring_tokens, routed_nvfp4, pingpong,
        num_tokens, hidden, intermediate_hidden) : 0;
    const int block_n = num_math_warpgroups == 3 and not pingpong ? 192 : block_n_2wg;
    const int num_epilogue_threads = is_sm90 ? 128 * num_math_warpgroups : num_block_epilogue_threads;
    // SM90 NVFP4 loads only the row groups of a block that hold valid tokens
    const int load_block_m = is_sm90 ? (routed_nvfp4 ? std::min(block_m, static_cast<int>(layout::kNumSM90NVFP4TokenGroupRows)) : block_m) : block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = is_mma_with_sf(mma_kind) ?
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, mma_kind) : std::pair(0, 0);
    // NOTES: FP8 activations, FP8 weights, and FP4 weights (unpacked to 8-bit in smem,
    // or kept packed for NVFP4) all use 128B swizzle
    // SM90 keeps FP4 weights packed in smem, 64 bytes per 128-element row
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = is_sm90 and not routed_fp8 ? 64 : 128;
    const int gran_k = get_mma_sf_gran_k(mma_kind);

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pull: divide token bytes by 2 until <= num_max_pull_bytes
    const int num_max_pull_bytes = is_mma_with_sf(mma_kind) ? 8192 : 4096;
    int num_bytes_per_pull = hidden * get_element_bits(mma_kind) / 8;
    while (num_bytes_per_pull > num_max_pull_bytes) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }

    // Pipeline
    const auto [num_stages, smem_size] = is_sm90 ?
        get_sm90_pipeline_config_for_mega_moe(
            SM90ArchSpec::smem_capacity,
            num_experts,
            block_m, block_n, block_k, num_bytes_per_pull, store_block_m_l1, store_block_m_l2,
            sf_block_m, sf_block_n,
            num_dispatch_threads / 32, num_epilogue_threads / 32,
            num_shared_experts > 0, routed_fp8, routed_nvfp4, pingpong) :
        get_pipeline_config_for_mega_moe(
            SM100ArchSpec::smem_capacity,
            num_experts, hidden,
            block_m, block_n, block_k, num_bytes_per_pull, store_block_m_l1, store_block_m_l2,
            sf_block_m, sf_block_n, gran_k,
            num_dispatch_threads / 32, num_epilogue_threads / 32,
            mma_kind);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull, pingpong
    };

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
