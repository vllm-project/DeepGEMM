#pragma once

#include <algorithm>
#include <format>
#include <iostream>
#include <unordered_set>

#include <deep_gemm/layout/sm90_mega_gate.cuh>
#include <deep_jit/utils/env.hpp>

#include "sm90.hpp"

namespace deep_gemm {

struct SM90MegaGateConfig {
    bool use_gemv = false, use_cublas = false;
    int num_expert_groups = 1, tile_experts = 0;
    int block_tokens = 0, block_k = 0, mma_tokens = 0, swizzle = 0;
    int num_math_groups = 0, num_threads = 128, num_stages = 0, smem_size = 0;
    int num_launch_ctas = 0, num_route_ctas = 0;

    friend std::ostream& operator << (std::ostream& os, const SM90MegaGateConfig& config) {
        return os << "SM90MegaGateConfig(use_gemv=" << config.use_gemv << ", use_cublas=" << config.use_cublas
                  << ", num_expert_groups=" << config.num_expert_groups << ", tile_experts=" << config.tile_experts
                  << ", block_tokens=" << config.block_tokens << ", block_k=" << config.block_k
                  << ", num_math_groups=" << config.num_math_groups << ", num_threads=" << config.num_threads
                  << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
                  << ", num_launch_ctas=" << config.num_launch_ctas << ", num_route_ctas=" << config.num_route_ctas << ")";
    }
};

template <uint32_t kBlockTokens, uint32_t kNumExperts, bool kSplitExperts>
static void set_sm90_mega_gate_tile(SM90MegaGateConfig& config) {
    using Tile = layout::sm90_mega_gate::Config<kBlockTokens, kNumExperts, 0>;
    constexpr auto kMaxStages = (SM90ArchSpec::smem_capacity - layout::mega_gate::kSharedMemoryAlignment) / Tile::kStageBytes;
    constexpr auto kNumStages = std::min(kMaxStages, kSplitExperts ? 3u : 4u);
    using Config = layout::sm90_mega_gate::Config<kBlockTokens, kNumExperts, kNumStages>;
    static_assert(kNumStages >= 2 and Config::kSmemBytes <= SM90ArchSpec::smem_capacity);
    config.tile_experts = kNumExperts;
    config.block_tokens = kBlockTokens;
    config.block_k = Config::kBlockK;
    config.mma_tokens = Config::kMmaTokens;
    config.swizzle = Config::kSwizzle;
    config.num_math_groups = Config::kNumMathGroups;
    config.num_threads = Config::kNumThreads;
    config.num_stages = kNumStages;
    config.smem_size = Config::kSmemBytes;
}

template <uint32_t kBlockTokens>
static void set_sm90_mega_gate_tile(SM90MegaGateConfig& config, const int num_aligned_experts) {
    switch (num_aligned_experts) {
        case 128: return set_sm90_mega_gate_tile<kBlockTokens, 128, false>(config);
        case 256: return set_sm90_mega_gate_tile<kBlockTokens, 256, false>(config);
        case 384: return set_sm90_mega_gate_tile<kBlockTokens, 384, false>(config);
        case 512: return set_sm90_mega_gate_tile<kBlockTokens, 512, false>(config);
        default: DG_HOST_UNREACHABLE("Invalid number of gate experts");
    }
}

static SM90MegaGateConfig get_sm90_mega_gate_config(const int num_tokens, const int hidden,
                                                   const int num_routed_experts, const int num_sms,
                                                   const bool cublas_available, const bool has_fixed_routing) {
    constexpr int kMaxGemvTokens = 16;
    constexpr int kMinSplitTokens = 32768;
    const auto num_aligned_experts = align(num_routed_experts, static_cast<int>(layout::mega_gate::kExpertAlignment));
    const auto deterministic = heuristics_runtime->get_deterministic_algorithms();
    SM90MegaGateConfig config;
    config.use_gemv = num_tokens <= kMaxGemvTokens and not deterministic;
    // Retain the measured H20 (78 SMs) crossover. Other SM90 configurations use
    // cuBLASLt at every prefill size, avoiding the large-batch regression on H200.
    const auto prefer_split_n = num_sms == 78 and num_tokens >= kMinSplitTokens and
                                num_aligned_experts > 128 and not has_fixed_routing;
    config.use_cublas = num_tokens > kMaxGemvTokens and not deterministic and cublas_available and not prefer_split_n;
    config.num_route_ctas = ceil_div(num_tokens, 4);
    if (not cublas_available)
        config.num_route_ctas = std::min(config.num_route_ctas, num_sms);
    if (config.use_gemv) {
        config.num_launch_ctas = num_tokens * (num_routed_experts / 4);
        if (not cublas_available)
            config.num_launch_ctas = std::min(config.num_launch_ctas, num_sms);
    } else if (not config.use_cublas) {
        // Splitting experts preserves each logit's full-K accumulation order and
        // also exposes multiple CTAs for small deterministic batches.
        const auto split_experts = num_aligned_experts > 128 and not has_fixed_routing and
                                   (deterministic or num_tokens >= kMinSplitTokens);
        config.num_expert_groups = split_experts ? num_aligned_experts / 128 : 1;
        const auto small_tile = num_tokens <= num_sms * 16;
        if (split_experts) {
            if (small_tile)
                set_sm90_mega_gate_tile<16, 128, true>(config);
            else
                set_sm90_mega_gate_tile<64, 128, true>(config);
        } else if (small_tile) {
            set_sm90_mega_gate_tile<16>(config, num_aligned_experts);
        } else {
            set_sm90_mega_gate_tile<64>(config, num_aligned_experts);
        }
        config.num_launch_ctas = ceil_div(num_tokens, config.block_tokens) * config.num_expert_groups;
        if (not cublas_available)
            config.num_launch_ctas = std::min(config.num_launch_ctas, num_sms);
    }

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format("SM90MegaGateConfig(num_tokens={}, hidden={}, num_routed_experts={}, num_sms={}, deterministic={}, fixed_routing={})",
                                     num_tokens, hidden, num_routed_experts, num_sms, deterministic, has_fixed_routing);
        static std::unordered_set<std::string> printed;
        if (printed.insert(key).second)
            std::cout << key << ": " << config << std::endl;
    }
    return config;
}

} // namespace deep_gemm
