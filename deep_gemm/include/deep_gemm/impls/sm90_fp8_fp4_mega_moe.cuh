#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// `ptx` helpers with a `memory` clobber, so that the compiler keeps plain memory accesses on their side
namespace ordered_ptx {

CUTLASS_DEVICE void sync_aligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_idx), "r"(num_threads) : "memory");
}

CUTLASS_DEVICE void sync_unaligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("barrier.sync %0, %1;" :: "r"(barrier_idx), "r"(num_threads) : "memory");
}

CUTLASS_DEVICE void red_add_rel(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}

CUTLASS_DEVICE uint64_t atomic_add_sys(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value) : "memory");
    return ret;
}

// Generic-to-async proxy shared memory fence, with the `memory` clobber the CuTe/CUTLASS fences lack
CUTLASS_DEVICE void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

} // namespace ordered_ptx

// An 8-entry E4M3 lookup table of the FP4 (E2M1) magnitudes times `2 ^ (shift - 5)`, for `shift` in [0, 11]
CUTLASS_DEVICE uint2 make_fp4_to_fp8_lut(const uint32_t& shift) {
    return {0x14100800u + shift * 0x08080800u, 0x24201c18u + shift * 0x08080808u};
}

// Dequantize the 8 FP4 codes of a word laid out by `relayout_fp4_weights_for_sm90_mega_moe` into 2 x 4 E4M3 values:
// magnitudes in bits `4i` (first) and `16 + 4i` (second), signs in bits `8i + 7` (first) and `8i + 3` (second)
CUTLASS_DEVICE void dequant_fp4x8_to_fp8x8(const uint32_t& codes, const uint2& lut, uint32_t& first, uint32_t& second) {
    const auto prmt = [&](const uint32_t& selectors) {
        uint32_t magnitudes;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(magnitudes) : "r"(lut.x), "r"(lut.y), "r"(selectors));
        return magnitudes;
    };
    const auto selectors = codes & 0x77777777u;
    first = prmt(selectors) | (codes & 0x80808080u);
    second = prmt(selectors >> 16) | ((codes << 4) & 0x80808080u);
}

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumSharedExperts,
    uint32_t kNumTopk,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M_L2,
    uint32_t SF_BLOCK_M,
    uint32_t kNumRingTokens,
    uint32_t kNumSFRingTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonMathThreads,
    uint32_t kNumMathThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kClampAfterSilu,
    float kActivationAlpha,
    float kActivationBeta,
    bool kFastMath,
    bool kPingPong,
    bool kHasShared = (kNumSharedExperts > 0),
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMathWarps = kNumMathThreads / 32,
    uint32_t kNumMathWarpgroups = kNumMathWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonMathThreads + kNumMathThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
    uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M,
    uint32_t kNumSharedSFTokens = layout::get_num_max_shared_sf_tokens(kNumMaxTokensPerRank),
    typename task_info_t = sched::TaskInfo<kHasShared>
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_fp4_mega_moe_impl(void* y,
                           int* cumulative_local_expert_recv_stats,
                           const uint32_t num_tokens,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads == 128, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonMathThreads == 128, "Invalid number of non-math threads");
    DG_STATIC_ASSERT(kNumMathThreads == 256, "Invalid number of math threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    const bool is_leader_cta = cta_rank_in_cluster == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights_sf);
    }

    // Workspaces and Buffer
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(),
        kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens,
        /*with_sf=*/ true,
        kNumSharedExperts
    );
    const auto workspace = buffer.workspace;

    using L2KBlockDependency = sched::L2KBlockDependency<L1_SHAPE_N, BLOCK_N, BLOCK_K>;

    // SF configs
    // NOTES: activations have one FP32 scale `amax / 448` per 128 K, routed weights one `transform_fp4_sf_for_sm90_mega_moe` word per 128 K, both in the SM100 MN-major layouts
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only one activation SF per K block is supported");
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, 128u), "Invalid SF_BLOCK_M");
    const auto transform_sf_token_idx = [](const uint32_t& idx) {
        return (idx % 32) * 4 + idx / 32;
    };

    // MMA configs
    // NOTES: always swap A/B, routed weights are dequantized from FP4 into registers
    using WGMMA = typename mma::sm90::FP8MMARSSelector<BLOCK_M>::type;
    using SharedWGMMA = typename mma::sm90::FP8MMASelector<BLOCK_M>::type;
    using a_dtype_t = cutlass::float_e4m3_t;
    using shared_b_dtype_t = cutlass::float_e4m3_t;
    DG_STATIC_ASSERT(BLOCK_N == WGMMA::M * kNumMathWarpgroups, "Each math warpgroup computes 64 weight rows");
    DG_STATIC_ASSERT(SharedWGMMA::M == WGMMA::M and SharedWGMMA::K == WGMMA::K, "Shared and routed WGMMAs must share the accumulator layout");
    DG_STATIC_ASSERT(BLOCK_M <= 128, "BLOCK_M > 128 does not fit into the math registers");

    // Math warpgroup roles
    // NOTES: cooperatively, every task is computed by both math warpgroups (64 weight rows each); with ping-pong, every task by one
    // warpgroup (all `BLOCK_N` rows as `kNumRowTiles` WGMMA tiles), which owns a TMA producer warp, a schedule stage, a part of the
    // pipeline stages and its epilogue buffers, so that its epilogue overlaps the other warpgroup's mainloop
    constexpr uint32_t kNumPipelines = kPingPong ? kNumMathWarpgroups : 1;
    constexpr uint32_t kNumTaskMathWarpgroups = kNumMathWarpgroups / kNumPipelines;
    constexpr uint32_t kNumRowTiles = BLOCK_N / WGMMA::M / kNumTaskMathWarpgroups;
    constexpr uint32_t kNumEpilogueThreads = kNumTaskMathWarpgroups * 128;
    constexpr uint32_t kNumWGMMAsPerTile = BLOCK_K / WGMMA::K;
    DG_STATIC_ASSERT(not kPingPong or (BLOCK_M <= 64 and kNumExperts / kNumRanks <= 64),
                     "Ping-pong needs the accumulators of 2 tiles in the 208 math registers");
    DG_STATIC_ASSERT(kNumPipelines <= kNumStages and kNumRowTiles <= 2, "Invalid ping-pong config");

    // Swizzle configs
    // NOTES: tokens and FP8 shared weights are 128 bytes per row per K block, packed FP4 weights 64 bytes
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 64;
    constexpr uint32_t kSwizzleSharedBMode = 128;
    constexpr uint32_t kNumWeightBytesPerRow = BLOCK_K / 2;
    constexpr uint32_t kNumSharedWeightBytesPerRow = BLOCK_K;
    DG_STATIC_ASSERT(kNumWeightBytesPerRow == kSwizzleBMode and kNumSharedWeightBytesPerRow == kSwizzleSharedBMode, "Invalid weight swizzling");

    // Epilogue configs
    // NOTES: L1 stores the whole `[BLOCK_M, 64]` FP8 tile with one TMA store, L2 stages `STORE_BLOCK_M_L2` tokens per 64-row tile
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t kNumL2StoreRowBytes = WGMMA::M * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M_L2 == 0 and STORE_BLOCK_M_L2 % 16 == 0, "Invalid L2 store block M");
    constexpr uint32_t kNumBankGroupBytes = 16;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Scheduler configs
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kNumScheduleConsumerThreads = 2 * kNumMathThreads / kNumPipelines;

    // Amax exchange slots between the 2 CTAs of a cluster
    constexpr uint32_t kNumAmaxSlots = 2;

    struct SharedStorage {
        alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
        alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];
        union {
            alignas(kSharedMemoryAlignment) cutlass::float_e4m3_t l1[BLOCK_M * L1_OUT_BLOCK_N];
            alignas(kSharedMemoryAlignment) nv_bfloat16 l2[kNumMathWarpgroups][STORE_BLOCK_M_L2 * WGMMA::M];
        } smem_d[kNumPipelines];
        alignas(kSharedMemoryAlignment) a_dtype_t smem_a[kNumStages][BLOCK_M * BLOCK_K];
        alignas(kSharedMemoryAlignment) uint8_t smem_b[kNumStages][BLOCK_N * (kHasShared ? kNumSharedWeightBytesPerRow : kNumWeightBytesPerRow)];
        uint32_t smem_sfa[kNumStages][SF_BLOCK_M];
        uint32_t smem_sfb[kNumStages][BLOCK_N];
        float4 amax_reduction[kNumMathWarps][BLOCK_M / 4];
        float4 peer_amax[kNumPipelines][kNumAmaxSlots][BLOCK_M / 4];
        task_info_t task_infos[kNumScheduleStages];
        Barrier dispatch_barriers[kNumDispatchWarps];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier peer_amax_full_barriers[kNumPipelines][kNumAmaxSlots];
        Barrier combine_barriers[kNumMathWarps * 2];
        Barrier task_info_full_barriers[kNumScheduleStages];
        Barrier task_info_empty_barriers[kNumScheduleStages];
    };
    constexpr uint32_t kNumReusableSmemBytes = offsetof(SharedStorage, peer_amax);
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Send buffers
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        static_cast<void*>(shared_storage.dispatch_send_buffer));

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        constexpr uint32_t kNumExpertCountUint4 = math::constexpr_ceil_div<uint32_t>(kNumExperts * sizeof(uint32_t), sizeof(uint4));
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExpertCountUint4; i += 32)
            ptx::st_shared(reinterpret_cast<uint4*>(shared_storage.expert_token_count) + i, 0u, 0u, 0u, 0u);
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            shared_storage.dispatch_barriers[i].init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at A + B (one producer for both with ping-pong), released by lane 0 of every math warp of the task
                shared_storage.full_barriers[i].init(kPingPong ? 1 : 2);
                shared_storage.empty_barriers[i].init(kNumTaskMathWarpgroups * 4);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumPipelines * kNumAmaxSlots; ++ i)
                shared_storage.peer_amax_full_barriers[i / kNumAmaxSlots][i % kNumAmaxSlots].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarps * 2; ++ i)
                shared_storage.combine_barriers[i].init(1);
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                shared_storage.task_info_full_barriers[i].init(1);
                shared_storage.task_info_empty_barriers[i].init(kNumScheduleConsumerThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    cute::cluster_sync();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumSMs, kNumRanks,
        kNumRingBlocks, /*kNumLocalityDomains=*/ 1,
        kNumSharedExperts, false, kPingPong>(
            workspace,
            shared_storage.task_info_full_barriers,
            shared_storage.task_info_empty_barriers,
            shared_storage.task_infos
    );

    // GEMM pipeline and TMA phases
    // NOTES: with ping-pong, pipeline `i` (a TMA producer warp and its math warpgroup) owns the stages `[stage_begin, stage_end)`
    uint32_t stage_begin = 0, stage_end = kNumStages;
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == stage_end - 1 ? stage_begin : stage_idx + 1;
        phase ^= stage_idx == stage_begin;
    };
    const auto bind_to_pipeline = [&](const uint32_t& pipeline_idx) {
        stage_begin = pipeline_idx * kNumStages / kNumPipelines;
        stage_end = (pipeline_idx + 1) * kNumStages / kNumPipelines;
        stage_idx = stage_begin;
        scheduler.bind_to_sched_stage(pipeline_idx);
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithMathBarrierIdx = 1;
    constexpr uint32_t kMathFullBarrierIdx = 2;
    constexpr uint32_t kMathWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 2;

    // Adjust registers
    constexpr bool kUseMoreMathRegisters = kNumExpertsPerRank <= 64;
    constexpr uint32_t kNumDispatchRegisters = kUseMoreMathRegisters ? 48 : 96;
    constexpr uint32_t kNumNonMathRegisters = kUseMoreMathRegisters ? 40 : 72;
    constexpr uint32_t kNumMathRegisters = kUseMoreMathRegisters ? 208 : 168;
    DG_STATIC_ASSERT(kUseMoreMathRegisters or BLOCK_M <= 64, "BLOCK_M > 64 does not fit into the reduced math registers");
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonMathRegisters * kNumNonMathThreads +
                     kNumMathRegisters * kNumMathThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and math use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kMathGridSyncIndex = 1;

    // TMA copies of the weights with SF of a K block into the current stage, returns their bytes
    const auto load_weights = [&](const task_info_t& task_info, const uint32_t& k_block_idx) {
        const auto tensor_map_b_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights :
                                      task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights :
                                      task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights :
                                    /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights;
        const auto tensor_map_sfb_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_weights_sf :
                                        task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf :
                                        task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_weights_sf :
                                      /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_weights_sf;
        const auto num_k_blocks = task_info.shape_k / BLOCK_K;
        const auto n_block_idx = task_info.n_cluster_idx * 2 + cta_rank_in_cluster;

        // NOTES: the weight descriptor splits N into 2 halves, and its K is in bytes
        const uint32_t half_n = task_info.shape_n / 2, n_idx = n_block_idx * BLOCK_N, half_idx = n_idx >= half_n;
        const uint32_t half_n_idx = n_idx - half_idx * half_n, expert_idx = task_info.is_shared() ? 0 : task_info.local_expert_idx;
        if (task_info.is_shared()) {
            tma::copy_nd<BLOCK_K, BLOCK_N, kSwizzleSharedBMode, shared_b_dtype_t, 4>(
                tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], reinterpret_cast<shared_b_dtype_t*>(shared_storage.smem_b[stage_idx]), 1,
                k_block_idx * BLOCK_K, half_n_idx, half_idx, expert_idx);
        } else {
            tma::copy_nd<kNumWeightBytesPerRow, BLOCK_N, kSwizzleBMode, uint8_t, 4>(
                tensor_map_b_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx], 1,
                k_block_idx * kNumWeightBytesPerRow, half_n_idx, half_idx, expert_idx);
        }
        tma::copy<BLOCK_N, 1, 0>(
            tensor_map_sfb_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfb[stage_idx],
            n_idx, expert_idx * num_k_blocks + k_block_idx);
        const uint32_t num_weight_bytes = BLOCK_N * (task_info.is_shared() ? kNumSharedWeightBytesPerRow : kNumWeightBytesPerRow);
        return num_weight_bytes + static_cast<uint32_t>(sizeof(SharedStorage::smem_sfb[0]));
    };

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[i * kNumTopk + lane_idx]);
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
        });
        ordered_ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
            shared_storage.expert_token_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ordered_ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ordered_ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            // Clear the peers' combine readiness, and push this launch's grid index to every peer for tagging it; +1 differs from zero
            if (thread_idx < kNumRanks) {
                *workspace.get_combine_ready_grid_idx_ptr(thread_idx) = 0;
                *sym_buffer.map(workspace.get_peer_grid_idx_ptr(sym_buffer.rank_idx), thread_idx) = ptx::get_grid_idx() + 1;
            }
            __syncwarp();

            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ordered_ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ordered_ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ordered_ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );

        ptx::fence_proxy_async_global();

        // Wait token data arrival
        scheduler.fetch_expert_recv_count();

        // Ensure the math barrier cannot run with the pull barrier
        ordered_ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithMathBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = &shared_storage.dispatch_barriers[warp_idx];

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks
            constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;
            DG_STATIC_ASSERT(kNumChunks * kNumBytesPerPull == kHidden, "kNumBytesPerPull must divide hidden");

            // TMA load token from remote rank and store into local
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;

            // Wait for ring buffer slot to be available (previous consumer must have finished all N blocks)
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (l1_empty_count_target > 0) {
                const auto empty_ptr = workspace.get_l1_empty_count_ptr(ring_block_idx);
                while (ptx::ld_acq(empty_ptr) < l1_empty_count_target);
                ptx::fence_proxy_async_global();
            }

            const auto src_base_ptr = sym_buffer.map(
                buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), current_rank_in_expert_idx);
            const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                asm volatile("" ::: "memory");
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull
                );
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++ i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull
                    );
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Load the weight first
            const auto weight = *sym_buffer.map(
                buffer.input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                current_rank_in_expert_idx);

            // Load and store SF
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                buffer.input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = buffer.l1_sf_buffer.get_base_ptr<uint32_t>();
            const auto sf_ring_token_idx = ring_block_idx * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_block);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumSFRingTokens + sf_ring_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and metadata
            if (cute::elect_one_sync()) {
                *buffer.l1_topk_weights_buffer.get_data_buffer(pool_token_idx % kNumRingTokens).template get_base_ptr<float>() = weight;

                // Write source metadata for combine write-back (logical pool token)
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Complete last chunk's store
                issue_and_wait_pull_store(kNumChunks - 1);
                const bool is_last_token = (token_idx == expert_end_idx - 1);
                ordered_ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(ring_block_idx),
                    is_last_token ? BLOCK_M - token_idx_in_block : 1u
                );
            }
            __syncwarp();
        }

        // Clean workspace for the next usage, and also do cumulative stats
        ordered_ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithMathBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count and schedule task counters
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if (warp_idx == 0 and cute::elect_one_sync()) {
                *workspace.get_l1_task_count_ptr() = *workspace.get_l2_task_count_ptr() = 0;
                *workspace.get_shared_l1_task_count_ptr() = *workspace.get_shared_l2_task_count_ptr() = 0;
            }
            __syncwarp();
            for (uint32_t i = thread_idx; i < workspace.num_shared_l2_pool_blocks; i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            __syncwarp();
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ordered_ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 full stuffs and ring buffer counts
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_mask_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((expert_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ordered_ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps or (kPingPong and warp_idx == kNumDispatchWarps + 1)) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonMathRegisters>();

        // GEMM TMA load warp for tokens with SFA; with ping-pong, also for weights with SFB, and only for its math warpgroup's tasks
        if constexpr (kPingPong)
            bind_to_pipeline(warp_idx - kNumDispatchWarps);
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const auto tensor_map_a_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts :
                                          task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts :
                                          task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts :
                                        /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts;
            const auto tensor_map_sfa_ptr = task_info.block_phase == sched::BlockPhase::Linear1 ? &tensor_map_l1_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_acts_sf :
                                            task_info.block_phase == sched::BlockPhase::SharedLinear1 ? &tensor_map_shared_l1_acts_sf :
                                          /*task_info.block_phase == sched::BlockPhase::SharedLinear2*/ &tensor_map_shared_l2_acts_sf;
            const auto num_k_blocks = task_info.shape_k / BLOCK_K;
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = task_info.is_shared() ? pool_block_idx : ring_block_idx;

            // Wait the entire token arrival
            if (task_info.block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_full_count_ptr(block_idx);
                const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                while (ptx::ld_acq(ptr) != num_expected_tokens);
                ptx::fence_proxy_async_global();
            } else if (task_info.block_phase == sched::BlockPhase::SharedLinear2) {
                const auto ptr = workspace.get_shared_l2_full_count_ptr(block_idx);
                const auto num_expected_blocks = (SHARED_L2_SHAPE_K / BLOCK_N) * 2;
                while (ptx::ld_acq(ptr) != num_expected_blocks);
                ptx::fence_proxy_async_global();
            }

            L2KBlockDependency l2_k_block_dependency(workspace.get_l2_full_mask_ptr(ring_block_idx), pool_block_idx / kNumRingBlocks);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if (task_info.block_phase == sched::BlockPhase::Linear2) {
                    l2_k_block_dependency.wait(k_block_idx);
                    ptx::fence_proxy_async_global();
                }

                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // TMA copy tokens and SFA (with ping-pong also weights with SFB), then arrive at full barrier
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx],
                        k_block_idx * BLOCK_K, block_idx * BLOCK_M);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, &shared_storage.full_barriers[stage_idx], shared_storage.smem_sfa[stage_idx],
                        block_idx * SF_BLOCK_M, k_block_idx);
                    shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(
                        sizeof(SharedStorage::smem_a[0]) + sizeof(SharedStorage::smem_sfa[0]) + (kPingPong ? load_weights(task_info, k_block_idx) : 0));
                }
                __syncwarp();
            }
        }
    } else if (not kPingPong and warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonMathRegisters>();

        // GEMM TMA load warp for weights with SF
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            for (uint32_t k_block_idx = 0; k_block_idx < task_info.shape_k / BLOCK_K; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);

                // TMA copy weights with SF
                if (cute::elect_one_sync())
                    shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(load_weights(task_info, k_block_idx));
                __syncwarp();
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Only joins the register deallocation of its warpgroup
        cutlass::arch::warpgroup_reg_dealloc<kNumNonMathRegisters>();
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonMathRegisters>();

        // Do mainloop by the leader CTA
        if (is_leader_cta)
            scheduler.mainloop(num_tokens, nullptr);
    } else {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Math warps
        const auto math_thread_idx = thread_idx - (kNumDispatchThreads + kNumNonMathThreads);
        const auto math_warp_idx = __shfl_sync(0xffffffff, math_thread_idx / 32, 0);
        const auto math_wg_idx = math_warp_idx / 4;
        const auto warp_idx_in_wg = math_warp_idx % 4;

        // The pipeline, first 64-row tile and epilogue threads of this warpgroup's tasks
        const uint32_t pipeline_idx = kPingPong ? math_wg_idx : 0;
        const uint32_t first_row_tile_idx = kPingPong ? 0 : math_wg_idx;
        const uint32_t first_epilogue_warp_idx = kPingPong ? math_wg_idx * 4 : 0;
        const uint32_t epilogue_thread_idx = kPingPong ? math_thread_idx % kNumEpilogueThreads : math_thread_idx;
        const auto sync_epilogue = [&]() {
            ordered_ptx::sync_aligned(kNumEpilogueThreads, kPingPong ? kMathWGBarrierStartIdx + math_wg_idx : kMathFullBarrierIdx);
        };
        if constexpr (kPingPong)
            bind_to_pipeline(pipeline_idx);

        // Each thread holds weight rows `row_idx` and `row_idx + 8` of every 64-row tile (the tile `t` adds `t * WGMMA::M` rows),
        // and tokens `8 * i + 2 * quad_idx + {0, 1}`
        // NOTES: with the 8-row gate/up weight interleave, the two rows are the gate and up of one L1 output channel
        const uint32_t quad_idx = lane_idx % 4;
        const uint32_t row_idx = first_row_tile_idx * WGMMA::M + warp_idx_in_wg * 16 + lane_idx / 4;

        // Shared memory addresses of the weight fragment and SF word of row `row_idx` in stage 0 (row `row_idx + 8` is 8 weight rows
        // and, as `row_idx % 32 + 8 < 32`, 32 SF words further)
        // NOTES: within a K block, the 16-byte chunk `quad_idx` of a 64-byte FP4 row holds the thread's codes of the 4 32-element
        // groups (one word per WGMMA), and chunks are XOR-swizzled with `(row / 2) % 4 = lane_idx / 8`, which the next tiles keep;
        // the addresses are pinned into registers, so that the compiler does not recompute them per K block
        const auto pin = [](uint32_t x) { asm volatile("mov.b32 %0, %0;" : "+r"(x)); return x; };
        const uint32_t weight_smem_addr = pin(static_cast<uint32_t>(__cvta_generic_to_shared(shared_storage.smem_b[0])) +
                                              row_idx * kNumWeightBytesPerRow + (quad_idx ^ (lane_idx / 8)) * kNumBankGroupBytes);
        const uint32_t sfb_smem_addr = pin(static_cast<uint32_t>(__cvta_generic_to_shared(shared_storage.smem_sfb[0] + transform_sf_token_idx(row_idx))));
        constexpr uint32_t kNumTileWeightBytes = WGMMA::M * kNumWeightBytesPerRow;
        constexpr uint32_t kNumTileSFBWords = WGMMA::M / 32;

        // Token and shared expert weight descriptors
        auto b_desc = mma::sm90::make_smem_desc(shared_storage.smem_a[0],
                                                static_cast<int>(mma::sm90::to_gmma_layout_type<cute::UMMA::Major::K, kSwizzleAMode, a_dtype_t>()));
        auto shared_a_desc = mma::sm90::make_smem_desc(shared_storage.smem_b[0] + first_row_tile_idx * WGMMA::M * kNumSharedWeightBytesPerRow,
                                                       static_cast<int>(mma::sm90::to_gmma_layout_type<cute::UMMA::Major::K, kSwizzleSharedBMode, shared_b_dtype_t>()));
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
        const uint32_t shared_a_desc_lo = __shfl_sync(0xffffffff, shared_a_desc.reg32_[0], 0);
        constexpr uint32_t kNumTileSharedADescUnits = WGMMA::M * kNumSharedWeightBytesPerRow / 16;

        // Ensure the math barrier cannot run with the pull barrier
        ordered_ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithMathBarrierIdx);

        // Persistently schedule over blocks
        uint32_t num_linear1_tasks = 0;
        task_info_t task_info;
        while (scheduler.get_next_task(task_info)) {
            const bool is_linear1 = task_info.block_phase == sched::BlockPhase::Linear1 or task_info.block_phase == sched::BlockPhase::SharedLinear1;
            const bool is_shared = task_info.is_shared();
            const uint32_t num_k_blocks = task_info.shape_k / BLOCK_K;

            // Mainloop: one WGMMA per 32-element group and tile (for routed experts, dequantized from FP4 first), then promote per K block
            // NOTES: tokens are the WGMMA N dimension, so a task issues the smallest N of `BLOCK_M`, `BLOCK_M / 2` and `BLOCK_M / 4` (at least 8)
            // that covers `valid_m`, with its own accumulators; `final_accum` stays zero for the tokens from N on
            float final_accum[kNumRowTiles][WGMMA::kNumAccum] = {};
            uint32_t sfb[kNumRowTiles][2];
            uint32_t next_full_ready = 1;
            const auto mainloop = [&](const auto& n) {
                using NWGMMA = typename mma::sm90::FP8MMARSSelector<std::decay_t<decltype(n)>::value>::type;
                using SharedNWGMMA = typename mma::sm90::FP8MMASelector<std::decay_t<decltype(n)>::value>::type;

                // Test the next stage's full barrier without blocking (so that its latency hides behind the WGMMA wait), wait for the
                // WGMMAs tile by tile (the first tile while the second tile's WGMMAs run), then promote with the token scales times the
                // weight scales `get_scale(tile, row, token_scale)`
                float accum[kNumRowTiles][NWGMMA::kNumAccum];
                const auto wait_and_promote = [&](const auto& get_scale) {
                    const uint32_t next_stage_idx = stage_idx == stage_end - 1 ? stage_begin : stage_idx + 1;
                    next_full_ready = shared_storage.full_barriers[next_stage_idx].test_wait(phase ^ (next_stage_idx == stage_begin));

                    float2 token_scales[NWGMMA::kNumAccum / 4];
                    #pragma unroll
                    for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                        #pragma unroll
                        for (uint32_t i = 0; i < NWGMMA::kNumAccum; ++ i)
                            ptx::warpgroup_fence_operand(accum[t][i]);
                        if (t + 1 < kNumRowTiles) {
                            ptx::warpgroup_wait<kNumWGMMAsPerTile>();
                        } else {
                            ptx::warpgroup_wait<0>();
                        }

                        #pragma unroll
                        for (uint32_t i = 0; i < NWGMMA::kNumAccum / 4; ++ i) {
                            if (t == 0) {
                                const auto sfa_ptr = shared_storage.smem_sfa[stage_idx] + transform_sf_token_idx(i * 8 + quad_idx * 2);
                                token_scales[i] = {__uint_as_float(ptx::ld_shared(sfa_ptr)), __uint_as_float(ptx::ld_shared(sfa_ptr + 4))};
                            }
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; ++ j) {
                                const auto token_scale = j % 2 == 0 ? token_scales[i].x : token_scales[i].y;
                                final_accum[t][i * 4 + j] = __fmaf_rn(accum[t][i * 4 + j], get_scale(t, j / 2, token_scale), final_accum[t][i * 4 + j]);
                            }
                        }
                    }
                };

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA arrivals unless the previous K block has seen them
                    if (not next_full_ready)
                        shared_storage.full_barriers[stage_idx].wait(phase);

                    // Weight SF words of both rows of every tile
                    const uint32_t sfb_stage_addr = sfb_smem_addr + stage_idx * static_cast<uint32_t>(sizeof(SharedStorage::smem_sfb[0]));
                    #pragma unroll
                    for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                        #pragma unroll
                        for (uint32_t i = 0; i < 2; ++ i)
                            sfb[t][i] = ptx::ld_shared_u32(sfb_stage_addr + static_cast<uint32_t>((t * kNumTileSFBWords + i * 32) * sizeof(uint32_t)));
                    }
                    const auto b_desc_base_lo = b_desc_lo + stage_idx * (sizeof(SharedStorage::smem_a[0]) / 16);
                    if (is_shared) {
                        // FP8 weights with 4 equal exponent bytes per SF word (see `requantize_fp8_weights_for_sm90_mega_moe`)
                        float weight_scales[kNumRowTiles][2];
                        #pragma unroll
                        for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                            weight_scales[t][0] = __uint_as_float((sfb[t][0] & 0xffu) << 23);
                            weight_scales[t][1] = __uint_as_float((sfb[t][1] & 0xffu) << 23);
                        }

                        // Issue WGMMAs
                        const auto a_desc_base_lo = shared_a_desc_lo + stage_idx * (sizeof(SharedStorage::smem_b[0]) / 16);
                        #pragma unroll
                        for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumWGMMAsPerTile; ++ k) {
                                #pragma unroll
                                for (uint32_t i = 0; i < NWGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum[t][i]);
                                ptx::warpgroup_arrive();
                                shared_a_desc.reg32_[0] = a_desc_base_lo + t * kNumTileSharedADescUnits + k * WGMMA::K / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                SharedNWGMMA::wgmma(shared_a_desc, b_desc, accum[t], k > 0);
                                ptx::warpgroup_commit_batch();
                            }
                        }
                        wait_and_promote([&](const uint32_t& t, const uint32_t& i, const float& token_scale) { return weight_scales[t][i] * token_scale; });
                    } else {
                        // FP4 weights with per-32 table shifts in the SF words (see `transform_fp4_sf_for_sm90_mega_moe`)
                        const auto get_lut = [&](const uint32_t& t, const uint32_t& i, const uint32_t& k) {
                            return make_fp4_to_fp8_lut((sfb[t][i] << (12 - 4 * k)) >> 28);
                        };

                        // Issue WGMMAs
                        // NOTES: the fragments of every tile stay in registers until the WGMMAs reading them complete
                        const uint32_t weight_stage_addr = weight_smem_addr + stage_idx * static_cast<uint32_t>(sizeof(SharedStorage::smem_b[0]));
                        uint32_t a[kNumRowTiles][kNumWGMMAsPerTile][4];
                        const auto issue = [&](const uint32_t& t, const uint32_t& k, const bool& do_fence) {
                            if (do_fence) {
                                #pragma unroll
                                for (uint32_t i = 0; i < NWGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum[t][i]);
                                ptx::warpgroup_arrive();
                            }
                            b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                            NWGMMA::wgmma(a[t][k], b_desc, accum[t], k > 0);
                            ptx::warpgroup_commit_batch();
                        };
                        #pragma unroll
                        for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                            const uint32_t tile_weight_addr = weight_stage_addr + t * kNumTileWeightBytes;
                            const uint4 codes[2] = {ptx::ld_shared_u32x4(tile_weight_addr), ptx::ld_shared_u32x4(tile_weight_addr + 8 * kNumWeightBytesPerRow)};
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumWGMMAsPerTile; ++ k) {
                                dequant_fp4x8_to_fp8x8(reinterpret_cast<const uint32_t*>(&codes[0])[k], get_lut(t, 0, k), a[t][k][0], a[t][k][2]);
                                dequant_fp4x8_to_fp8x8(reinterpret_cast<const uint32_t*>(&codes[1])[k], get_lut(t, 1, k), a[t][k][1], a[t][k][3]);
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++ i)
                                    ptx::warpgroup_fence_operand(a[t][k][i]);
                                if constexpr (kNumRowTiles == 1)
                                    issue(t, k, true);
                            }

                            // With ping-pong (two tiles), dequantize the whole tile first (its chains issue back to back, the second
                            // tile's under the first tile's WGMMAs), then one WGMMA fence for the tile
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumWGMMAsPerTile and kNumRowTiles > 1; ++ k)
                                issue(t, k, k == 0);
                        }
                        wait_and_promote([](const uint32_t&, const uint32_t&, const float& token_scale) { return token_scale; });
                    }

                    // Release the stage after all shared memory reads
                    __syncwarp();
                    if (lane_idx == 0)
                        shared_storage.empty_barriers[stage_idx].arrive();
                }
            };

            // TMA warps have read the task information once the first stage arrives
            shared_storage.full_barriers[stage_idx].wait(phase);
            scheduler.release_task_info();
            // NOTES: at BLOCK_M 16, the quarter is also the half
            constexpr uint32_t kQuarterN = cute::max(BLOCK_M / 4, 8u);
            if (task_info.valid_m <= kQuarterN) {
                mainloop(cute::Int<kQuarterN>{});
            } else if (kQuarterN < BLOCK_M / 2 and task_info.valid_m <= BLOCK_M / 2) {
                mainloop(cute::Int<BLOCK_M / 2>{});
            } else {
                mainloop(cute::Int<BLOCK_M>{});
            }

            // Apply the per-row weight scales of routed experts
            if (not is_shared) {
                #pragma unroll
                for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                    const float weight_scales[2] = {__uint_as_float((sfb[t][0] & 0xffu) << 23), __uint_as_float((sfb[t][1] & 0xffu) << 23)};
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                        final_accum[t][i] *= weight_scales[(i / 2) % 2];
                }
            }

            // Compute offsets
            const uint32_t valid_m = ptx::exchange(task_info.valid_m, 0);
            const uint32_t pool_block_idx = task_info.pool_block_idx;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = is_shared ? pool_block_idx : ring_block_idx;
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;  // Ring-buffer offset for reusable data buffers
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t n_block_idx = task_info.n_cluster_idx * 2 + cta_rank_in_cluster;

            if (is_linear1) {
                // Wait L2 block empty
                if (not is_shared) {
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = (L2_SHAPE_N / BLOCK_N) * (pool_block_idx / kNumRingBlocks);
                    while (ptx::ld_acq(l2_empty_ptr) != num_expected_blocks);
                    ptx::fence_proxy_async_global();
                }

                // Apply SwiGLU and top-k weights (1 for shared experts): gate * sigmoid(alpha * gate) * (up + beta) * weight
                float2 activations[kNumRowTiles][WGMMA::kNumAccum / 4];
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    float2 weights = {1.0f, 1.0f};
                    if (not is_shared)
                        weights = *reinterpret_cast<const float2*>(buffer.l1_topk_weights_buffer.get_base_ptr<float>() + ring_m_idx + i * 8 + quad_idx * 2);

                    #pragma unroll
                    for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                        auto bf16_gate = __float22bfloat162_rn({final_accum[t][i * 4 + 0], final_accum[t][i * 4 + 1]});
                        auto bf16_up = __float22bfloat162_rn({final_accum[t][i * 4 + 2], final_accum[t][i * 4 + 3]});

                        // Clamp
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                            if constexpr (not kClampAfterSilu)
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                            bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                            bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                        }

                        // SwiGLU
                        const auto gate = __bfloat1622float2(bf16_gate);
                        auto sigmoid_input = gate;
                        if constexpr (kActivationAlpha != 1.0f)
                            sigmoid_input = {sigmoid_input.x * kActivationAlpha, sigmoid_input.y * kActivationAlpha};
                        const float2 neg_gate_exp = {
                            kFastMath ? __expf(-sigmoid_input.x) : expf(-sigmoid_input.x),
                            kFastMath ? __expf(-sigmoid_input.y) : expf(-sigmoid_input.y)};
                        const float2 denom = {1.0f + neg_gate_exp.x, 1.0f + neg_gate_exp.y};
                        float2 gated;
                        if constexpr (kFastMath) {
                            gated = {gate.x * math::fast_rcp(denom.x), gate.y * math::fast_rcp(denom.y)};
                        } else {
                            gated = {gate.x / denom.x, gate.y / denom.y};
                        }
                        if constexpr (kClampAfterSilu and kActivationClamp != cute::numeric_limits<float>::infinity())
                            gated = {fminf(gated.x, kActivationClamp), fminf(gated.y, kActivationClamp)};
                        auto up = __bfloat1622float2(bf16_up);
                        if constexpr (kActivationBeta != 0.0f)
                            up = {up.x + kActivationBeta, up.y + kActivationBeta};
                        activations[t][i] = {gated.x * up.x * weights.x, gated.y * up.y * weights.y};
                    }
                }

                // Reduce the per-token amax over the channels of the warp, then over the 64 channels of the CTA
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const auto reduce_amax = [&](const auto& get) {
                        float amax = fabsf(get(activations[0][i]));
                        #pragma unroll
                        for (uint32_t t = 1; t < kNumRowTiles; ++ t)
                            amax = fmaxf(amax, fabsf(get(activations[t][i])));
                        return math::warp_reduce<4, true>(amax, math::ReduceMax<float>());
                    };
                    const float2 amax = {reduce_amax([](const float2& a) { return a.x; }), reduce_amax([](const float2& a) { return a.y; })};
                    if (lane_idx < 4)
                        ptx::st_shared(reinterpret_cast<float2*>(shared_storage.amax_reduction[math_warp_idx]) + i * 4 + lane_idx, amax);
                }
                sync_epilogue();

                // Thread `t < BLOCK_M / 4` reduces tokens `4t` to `4t + 3` into the first warp's row, and sends them to the same
                // pipeline of the peer CTA, which computes the same tasks
                const auto amax_slot_idx = num_linear1_tasks % kNumAmaxSlots;
                const auto amax_slot_phase = (num_linear1_tasks / kNumAmaxSlots) & 1;
                auto& peer_amax_full_barrier = shared_storage.peer_amax_full_barriers[pipeline_idx][amax_slot_idx];
                const auto peer_cta_idx = cta_rank_in_cluster ^ 1;
                if (epilogue_thread_idx < BLOCK_M / 4) {
                    float4 amax = ptx::ld_shared(&shared_storage.amax_reduction[first_epilogue_warp_idx][epilogue_thread_idx]);
                    #pragma unroll
                    for (uint32_t i = 1; i < kNumEpilogueThreads / 32; ++ i) {
                        const auto other = ptx::ld_shared(&shared_storage.amax_reduction[first_epilogue_warp_idx + i][epilogue_thread_idx]);
                        amax = {fmaxf(amax.x, other.x), fmaxf(amax.y, other.y), fmaxf(amax.z, other.z), fmaxf(amax.w, other.w)};
                    }
                    shared_storage.amax_reduction[first_epilogue_warp_idx][epilogue_thread_idx] = amax;
                    if (epilogue_thread_idx == 0)
                        peer_amax_full_barrier.arrive_and_expect_tx(BLOCK_M * sizeof(float), peer_cta_idx);
                    ptx::fence_acq_rel_cluster();
                    ptx::st_async_cluster(&shared_storage.peer_amax[pipeline_idx][amax_slot_idx][epilogue_thread_idx], amax, peer_cta_idx, peer_amax_full_barrier);
                }
                sync_epilogue();
                peer_amax_full_barrier.wait(amax_slot_phase);
                ++ num_linear1_tasks;

                // The 128-channel amax of token `i`
                const auto get_amax = [&](const uint32_t& i) {
                    const auto local_amax = reinterpret_cast<const float*>(shared_storage.amax_reduction[first_epilogue_warp_idx]);
                    const auto peer_amax = reinterpret_cast<const float*>(shared_storage.peer_amax[pipeline_idx][amax_slot_idx]);
                    return fmaxf(ptx::ld_shared(local_amax + i), ptx::ld_shared(peer_amax + i));
                };

                // FP32 scale `amax * (1 / 448)` and its correctly rounded inverse, like `per_token_cast_to_fp8`
                // NOTES: one Newton step on `rcp.approx` equals `__frcp_rn` for these scales, without its slow path
                const auto get_sf = [](const float& amax) { return fmaxf(amax, 1e-4f) * (1.0f / 448.0f); };
                const auto get_sf_inv = [](const float& sf) {
                    const float r = math::fast_rcp(sf);
                    return __fmaf_rn(__fmaf_rn(-sf, r, 1.0f), r, r);
                };

                // Cast into FP8 E4M3 and store into shared memory
                const uint32_t channel_idx = first_row_tile_idx * (WGMMA::M / 2) + warp_idx_in_wg * 8 + lane_idx / 4;
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const uint32_t token_idx = i * 8 + quad_idx * 2;
                    const float2 sf_inv = {get_sf_inv(get_sf(get_amax(token_idx))), get_sf_inv(get_sf(get_amax(token_idx + 1)))};
                    #pragma unroll
                    for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                        const auto fp8x2 = __nv_fp8x2_e4m3(make_float2(activations[t][i].x * sf_inv.x, activations[t][i].y * sf_inv.y));

                        // 64B swizzling: 16-byte chunk `c` of token row `t` is at `c ^ ((t / 2) % 4)`; each further 64-row tile adds 2 chunks
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(shared_storage.smem_d[pipeline_idx].l1) + token_idx * L1_OUT_BLOCK_N +
                            (((channel_idx / kNumBankGroupBytes + t * 2) ^ ((token_idx / 2) % 4)) * kNumBankGroupBytes) + channel_idx % kNumBankGroupBytes;
                        smem_ptr[0] = static_cast<uint8_t>(fp8x2.__x);
                        smem_ptr[L1_OUT_BLOCK_N] = static_cast<uint8_t>(fp8x2.__x >> 8);
                    }
                }

                // Store SF into `l2_sf_buffer` or `shared_l2_sf_buffer` as FP32 (MN-major layout, one scale per int32 word),
                // only by the leader CTA
                if (is_leader_cta and epilogue_thread_idx < BLOCK_M) {
                    const auto sf_ptr = is_shared ?
                        buffer.shared_l2_sf_buffer.get_base_ptr<uint32_t>() + task_info.n_cluster_idx * kNumSharedSFTokens :
                        buffer.l2_sf_buffer.get_base_ptr<uint32_t>() + task_info.n_cluster_idx * kNumSFRingTokens;
                    sf_ptr[block_idx * SF_BLOCK_M + transform_sf_token_idx(epilogue_thread_idx)] = __float_as_uint(get_sf(get_amax(epilogue_thread_idx)));
                }

                // Issue TMA store
                ordered_ptx::fence_view_async_shared();
                sync_epilogue();
                if (epilogue_thread_idx == 0) {
                    const auto tensor_map_l1_output_ptr = is_shared ? &tensor_map_shared_l1_output : &tensor_map_l1_output;
                    cute::SM90_TMA_STORE_2D::copy(tensor_map_l1_output_ptr, shared_storage.smem_d[pipeline_idx].l1, n_block_idx * L1_OUT_BLOCK_N, m_idx);
                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                }

                // Notify L2 and increment L1 empty count
                sync_epilogue();
                if (epilogue_thread_idx == 0) {
                    if (is_shared) {
                        ordered_ptx::red_add_rel(workspace.get_shared_l2_full_count_ptr(pool_block_idx), 1u);
                    } else {
                        L2KBlockDependency::arrive(workspace.get_l2_full_mask_ptr(ring_block_idx), n_block_idx);
                        ordered_ptx::red_add_rel(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
            } else {
                // Increment L2 empty count for this physical slot (one per N block)
                if (not is_shared and epilogue_thread_idx == 0)
                    ptx::red_add(workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                // NOTES: each warpgroup transposes `STORE_BLOCK_M_L2` tokens by the 64 channels of each tile (128 bytes, 128B swizzling)
                // through shared memory, then 8 lanes write one token row of a tile
                const auto smem_d = reinterpret_cast<uint8_t*>(shared_storage.smem_d[pipeline_idx].l2[first_row_tile_idx]);
                constexpr uint32_t kNumL2StoreTileBytes = STORE_BLOCK_M_L2 * kNumL2StoreRowBytes;
                const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;
                #pragma unroll
                for (uint32_t s = 0; s < BLOCK_M / STORE_BLOCK_M_L2; ++ s) {
                    if (s * STORE_BLOCK_M_L2 >= valid_m)
                        break;

                    // Wait shared memory release from the previous NVLink store
                    if (s > 0)
                        ordered_ptx::sync_aligned(128, kMathWGBarrierStartIdx + math_wg_idx);

                    // Store into shared memory: 4 transposed 8x8 matrices per `stmatrix`, i.e. 2 token atoms by 2 row atoms
                    #pragma unroll
                    for (uint32_t t = 0; t < kNumRowTiles; ++ t) {
                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_M_L2 / 8; i += 2) {
                            const uint32_t accum_idx = s * (STORE_BLOCK_M_L2 / 8) + i;
                            const uint32_t row = (i + lane_idx / 16) * 8 + lane_idx % 8;
                            const uint32_t col = warp_idx_in_wg * 2 + (lane_idx / 8) % 2;
                            ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                                math::cast_into_bf16_and_pack(final_accum[t][accum_idx * 4 + 0], final_accum[t][accum_idx * 4 + 1]),
                                math::cast_into_bf16_and_pack(final_accum[t][accum_idx * 4 + 2], final_accum[t][accum_idx * 4 + 3]),
                                math::cast_into_bf16_and_pack(final_accum[t][accum_idx * 4 + 4], final_accum[t][accum_idx * 4 + 5]),
                                math::cast_into_bf16_and_pack(final_accum[t][accum_idx * 4 + 6], final_accum[t][accum_idx * 4 + 7]),
                                smem_d + t * kNumL2StoreTileBytes + row * kNumL2StoreRowBytes + (col ^ (row % 8)) * kNumBankGroupBytes
                            );
                        }
                    }
                    ordered_ptx::sync_aligned(128, kMathWGBarrierStartIdx + math_wg_idx);

                    // Write into remote buffers
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M_L2 / 16; ++ i) {
                        const uint32_t row = i * 16 + warp_idx_in_wg * 4 + lane_idx / 8;
                        const uint32_t m_idx_in_block = s * STORE_BLOCK_M_L2 + row;
                        if (m_idx_in_block >= valid_m)
                            break;

                        // NOTES: shared experts write the extra combine slot `kNumTopk` of their own tokens
                        const auto [dst_rank_idx, dst_token_idx, dst_topk_idx] = is_shared ?
                            layout::TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk) :
                            *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
                        const auto packed = ptx::ld_shared(reinterpret_cast<const float4*>(
                            smem_d + row * kNumL2StoreRowBytes + ((lane_idx % 8) ^ (row % 8)) * kNumBankGroupBytes));
                        const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            (n_block_idx * BLOCK_N + first_row_tile_idx * WGMMA::M) * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (lane_idx % 8) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;

                        // The second tile holds the next 64 channels
                        if constexpr (kNumRowTiles > 1) {
                            *sym_buffer.map(dst_ptr + kNumL2StoreRowBytes / sizeof(float4), dst_rank_idx) = ptx::ld_shared(reinterpret_cast<const float4*>(
                                smem_d + kNumL2StoreTileBytes + row * kNumL2StoreRowBytes + ((lane_idx % 8) ^ (row % 8)) * kNumBankGroupBytes));
                        }
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                sync_epilogue();
            }
        }

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from the start up to `peer_amax`
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks, limited by both smem and registers
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumMathWarps * kNumHiddenBytes <= kNumReusableSmemBytes and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kNumChunkSlots * kNumMathWarps * kNumHiddenBytes / kNumChunks <= kNumReusableSmemBytes, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk + (kHasShared ? 1u : 0u) <= 32u, "Top-k + shared must fit in a single warp");

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (math_warp_idx + i * kNumMathWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (math_warp_idx + kNumMathWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return &shared_storage.combine_barriers[i + math_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        // Peers' grid indices for tagging their combine readiness
        uint64_t peer_grid_idx = 0;
        if (sm_idx == 0 and math_thread_idx < kNumRanks)
            peer_grid_idx = *workspace.get_peer_grid_idx_ptr(math_thread_idx);

        // All local L2 writes are done after this grid sync
        comm::grid_sync<kNumSMs, kMathGridSyncIndex>(
            workspace, sm_idx, math_thread_idx,
            [&]() { ordered_ptx::sync_aligned(kNumMathThreads, kMathFullBarrierIdx); }
        );

        // Notify remote ranks; ordered before the cleanup barrier by the dispatch/math sync below
        if (sm_idx == 0 and math_thread_idx < kNumRanks)
            ptx::st_rel_sys(sym_buffer.map(workspace.get_combine_ready_grid_idx_ptr(sym_buffer.rank_idx), math_thread_idx), peer_grid_idx);

        // Barrier with dispatch warps, so that they can do clean workspace
        ordered_ptx::sync_unaligned(kNumDispatchThreads + kNumMathThreads, kDispatchWithMathBarrierIdx);
        ordered_ptx::fence_view_async_shared();

        // Iterate over all token chunks
        const auto grid_idx = ptx::get_grid_idx() + 1;
        for (uint32_t token_chunk_idx = math_warp_idx * kNumSMs + sm_idx; token_chunk_idx < num_tokens * kNumChunks; token_chunk_idx += kNumSMs * kNumMathWarps) {
            const uint32_t token_idx = token_chunk_idx / kNumChunks;
            const uint32_t chunk_idx = token_chunk_idx % kNumChunks;

            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>()[token_idx * kNumTopk + lane_idx]) :
                (kHasShared and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Wait for the ranks of the selected experts to finish their L2 writes
            const bool is_routed = lane_idx < kNumTopk and stored_topk_slot_idx >= 0;
            const auto peer_ready_ptr = workspace.get_combine_ready_grid_idx_ptr(
                is_routed ? static_cast<uint32_t>(stored_topk_slot_idx) / kNumExpertsPerRank : 0);
            comm::wait_until([&]() { return __all_sync(0xffffffff, not is_routed or ptx::ld_acq_sys(peer_ready_ptr) == grid_idx); },
                             [&]() { printf("DeepGEMM combine peers timeout: rank=%u, token=%u\n", sym_buffer.rank_idx, token_idx); });
            ptx::fence_proxy_async_global();

            const uint32_t chunk_byte_offset = chunk_idx * kNumChunkBytes;

            // Move mask and load
            uint32_t mask = total_mask;
            const auto move_mask_and_load = [&](const uint32_t& i) {
                if (mask) {
                    // Shared slot goes first, then routed slots in top-k order
                    const uint32_t slot_idx = (mask >> kNumTopk) ? kNumTopk : __ffs(mask) - 1;
                    mask ^= 1 << slot_idx;

                    // Load
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            buffer.combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            chunk_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                    }
                    __syncwarp();
                    return true;
                }
                return false;
            };

            // Load the first selection
            bool do_reduce = move_mask_and_load(load_stage_idx);

            // Accumulate all top-k contributions for this chunk in float registers
            float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
            while (do_reduce) {
                // Prefetch next top-k into the buffer while current is being accumulated
                do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                // Accumulate
                combine_load_barriers[load_stage_idx]->wait(combine_phase);
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                    const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                }
                ordered_ptx::fence_view_async_shared();
                __syncwarp();
                combine_phase ^= load_stage_idx;
                load_stage_idx ^= 1;
            }

            // Cast
            #pragma unroll
            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                uint4 casted;
                auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                #pragma unroll
                for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                    casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                // Wait share memory release and write
                if (j == 0) {
                    ptx::tma_store_wait<0>();
                    __syncwarp();
                }
                ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                               casted.x, casted.y, casted.z, casted.w);
            }
            __syncwarp();

            // TMA store the token chunk
            if (cute::elect_one_sync()) {
                ordered_ptx::fence_view_async_shared();
                ptx::tma_store_1d(
                    math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                    combine_store_buffer, kNumChunkBytes);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
        cute::tma_store_wait<0>();
    }

    // The peer CTA may still access the shared memory
    cute::cluster_sync();
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
