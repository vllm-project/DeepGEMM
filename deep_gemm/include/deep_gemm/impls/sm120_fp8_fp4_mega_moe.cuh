#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm120_utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm120.cuh>

namespace deep_gemm {

template <uint32_t kBlockM>
struct SM120MegaMoESharedStorage {
    static constexpr uint32_t kBlockN = kBlockM == 64 ? 64 : 32;
    static constexpr uint32_t kNumStages = 2;
    static constexpr uint32_t kPackedWeightStride = 80;
    static constexpr uint32_t kPackedWeightBytes = kBlockN * 2 * kPackedWeightStride;
    union {
        struct {
            alignas(16) char activations[kNumStages][kBlockM * 128];
            alignas(16) char packed_weights[kNumStages][kPackedWeightBytes];
            uint32_t activation_scales[kNumStages][kBlockM];
            uint32_t weight_scales[kNumStages][kBlockN * 2];
        } gemm;
        struct {
            float values[kBlockM * kBlockN];
        } activation;
    };
    layout::TokenSrcMetadata sources[kBlockM];
    float routing_weights[kBlockM];
    uint32_t task_idx, tile_idx, task_offset, ring_idx, expert_idx, num_tokens, pool_start;
};

template <uint32_t kNumCTAs>
CUTLASS_DEVICE void sm120_mega_moe_grid_sync(const layout::Workspace& workspace) {
    __threadfence();
    comm::grid_sync<kNumCTAs>(workspace, blockIdx.x, threadIdx.x, []() { __syncthreads(); });
}

template <uint32_t kNumRanks, uint32_t kNumCTAs, bool kCachePeerSchedule = false>
CUTLASS_DEVICE void sm120_mega_moe_peer_barrier(
    const layout::Workspace& workspace, const layout::SymBuffer<kNumRanks>& sym_buffer
) {
    __threadfence_system();
    sm120_mega_moe_grid_sync<kNumCTAs>(workspace);
    if constexpr (kNumRanks > 1) {
        if (blockIdx.x == 0) {
            __shared__ uint64_t epoch;
            if (threadIdx.x == 0)
                epoch = ++*workspace.get_nvl_barrier_counter_ptr();
            __syncthreads();
            auto* ready = (epoch & 1) ? workspace.get_combine_ready_grid_idx_ptr()
                                      : workspace.get_peer_grid_idx_ptr();
            if (threadIdx.x < kNumRanks)
                ptx::st_rel_sys(sym_buffer.map(ready + sym_buffer.rank_idx, threadIdx.x), epoch);
            __syncthreads();
            if (threadIdx.x < kNumRanks) {
                comm::wait_until([&]() { return ptx::ld_acq_sys(ready + threadIdx.x) == epoch; }, [&]() {
                    printf("SM120 MegaMoE peer barrier timeout: rank=%u peer=%u epoch=%llu\n",
                           sym_buffer.rank_idx, threadIdx.x, static_cast<unsigned long long>(epoch));
                });
            }
            __syncthreads();
            if constexpr (kCachePeerSchedule) {
                __shared__ uint32_t peer_num_batches[kNumRanks];
                auto* const cached_capacity = workspace.get_l1_full_count_ptr(workspace.num_experts_per_rank + 1);
                if (threadIdx.x < kNumRanks) {
                    peer_num_batches[threadIdx.x] = ptx::ld_acq_sys(
                        sym_buffer.map(workspace.get_l1_task_count_ptr(), threadIdx.x));
                    cached_capacity[threadIdx.x] = ptx::ld_acq_sys(
                        sym_buffer.map(workspace.get_shared_l2_task_count_ptr(), threadIdx.x));
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    uint32_t max_batches = 0;
                    for (uint32_t rank_idx = 0; rank_idx < kNumRanks; ++rank_idx)
                        max_batches = cute::max(max_batches, peer_num_batches[rank_idx]);
                    cached_capacity[kNumRanks] = max_batches;
                }
                __syncthreads();
            }
        }
    }
    sm120_mega_moe_grid_sync<kNumCTAs>(workspace);
}

CUTLASS_DEVICE uint32_t sm120_mega_moe_sf_row(const uint32_t row) {
    return (row / 128) * 128 + (row % 32) * 4 + (row % 128) / 32;
}

CUTLASS_DEVICE uint32_t sm120_mega_moe_smem_offset(const uint32_t row, const uint32_t col) {
    return row * 128 + (col ^ ((row % 8) * 16));
}

CUTLASS_DEVICE uint32_t sm120_mega_moe_activation_offset(
    const uint32_t row, const uint32_t col, const uint32_t block_n
) {
    return row * block_n + (col ^ ((row % 4) * 8));
}

CUTLASS_DEVICE uint32_t sm120_mega_moe_expand_fp4(const uint32_t packed) {
    return ((packed & 0x000fu) << 2) | ((packed & 0x00f0u) << 6) |
           ((packed & 0x0f00u) << 10) | ((packed & 0xf000u) << 14);
}

template <uint32_t kBytes>
CUTLASS_DEVICE void sm120_mega_moe_copy_async(void* destination, const uint8_t* source) {
    const auto address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    if constexpr (kBytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(source) : "memory");
    else {
        DG_STATIC_ASSERT(kBytes == 4, "Invalid SM120 MegaMoE copy size");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(address), "l"(source) : "memory");
    }
}

template <uint32_t kInputWidth, uint32_t kOutputWidth, uint32_t kBlockM, bool kIsL1>
CUTLASS_DEVICE void sm120_mega_moe_load_stage(
    const layout::MegaMoEBuffer& buffers,
    const uint8_t* weights, const uint32_t* weight_scales,
    const int64_t sf_expert_stride, const int64_t sf_k_stride,
    SM120MegaMoESharedStorage<kBlockM>& shared,
    const uint32_t channel_start, const uint32_t k_start, const uint32_t stage_idx
) {
    constexpr uint32_t kBlockN = SM120MegaMoESharedStorage<kBlockM>::kBlockN;
    constexpr uint32_t kNumThreads = kBlockM == 16 ? 128 : 256;
    constexpr uint32_t kNumWeightRows = kIsL1 ? kBlockN * 2 : kBlockN;
    constexpr uint32_t kWeightRowsPerExpert = kIsL1 ? kOutputWidth * 2 : kOutputWidth;
    const auto& activation_buffer = kIsL1 ? buffers.l1_token_buffer : buffers.l2_token_buffer;
    const auto& scale_buffer = kIsL1 ? buffers.l1_sf_buffer : buffers.l2_sf_buffer;
    const auto* activations = activation_buffer.get_base_ptr<uint8_t>();
    const auto* activation_scales = scale_buffer.get_base_ptr<uint32_t>();
    const uint32_t ring_start = shared.ring_idx * kBlockM;
    for (uint32_t copy_idx = threadIdx.x; copy_idx < kBlockM * 8; copy_idx += kNumThreads) {
        const uint32_t row = copy_idx / 8;
        const uint32_t col = (copy_idx % 8) * 16;
        sm120_mega_moe_copy_async<16>(shared.gemm.activations[stage_idx] + sm120_mega_moe_smem_offset(row, col),
            activations + static_cast<int64_t>(ring_start + row) * kInputWidth + k_start + col);
    }
    for (uint32_t copy_idx = threadIdx.x; copy_idx < kNumWeightRows * 4; copy_idx += kNumThreads) {
        const uint32_t row = copy_idx / 4;
        const uint32_t col = (copy_idx % 4) * 32;
        const uint32_t channel = channel_start + row % kBlockN;
        const uint32_t weight_row = kIsL1 ? (channel / 8) * 16 + (row / kBlockN) * 8 + channel % 8 : channel;
        sm120_mega_moe_copy_async<16>(shared.gemm.packed_weights[stage_idx] +
            row * SM120MegaMoESharedStorage<kBlockM>::kPackedWeightStride + col / 2,
            weights + (static_cast<int64_t>(shared.expert_idx) * kWeightRowsPerExpert + weight_row) *
                (kInputWidth / 2) + (k_start + col) / 2);
    }
    if (threadIdx.x < kBlockM) {
        const uint32_t sf_row = shared.ring_idx * 128 + sm120_mega_moe_sf_row(threadIdx.x);
        sm120_mega_moe_copy_async<4>(&shared.gemm.activation_scales[stage_idx][threadIdx.x],
            reinterpret_cast<const uint8_t*>(activation_scales +
                static_cast<int64_t>(k_start / 128) * scale_buffer.num_max_tokens_per_rank + sf_row));
    }
    if (threadIdx.x < kNumWeightRows) {
        const uint32_t channel = channel_start + threadIdx.x % kBlockN;
        const uint32_t weight_row = kIsL1 ? (channel / 8) * 16 + (threadIdx.x / kBlockN) * 8 + channel % 8 : channel;
        sm120_mega_moe_copy_async<4>(&shared.gemm.weight_scales[stage_idx][threadIdx.x],
            reinterpret_cast<const uint8_t*>(weight_scales + static_cast<int64_t>(shared.expert_idx) * sf_expert_stride +
                static_cast<int64_t>(k_start / 128) * sf_k_stride + sm120_mega_moe_sf_row(weight_row)));
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool kSystemFence>
CUTLASS_DEVICE void sm120_mega_moe_complete_tile(uint32_t* counter) {
    if constexpr (kSystemFence)
        __threadfence_system();
    else
        __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t previous_count;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(previous_count) : "l"(counter) : "memory");
    }
    __syncthreads();
}

template <uint32_t kInputWidth, uint32_t kOutputWidth, uint32_t kBlockM, uint32_t kNumMChunks, bool kIsL1>
CUTLASS_DEVICE void sm120_mega_moe_gemm(
    const layout::MegaMoEBuffer& buffers,
    const uint8_t* weights, const uint32_t* weight_scales,
    const int64_t sf_expert_stride, const int64_t sf_k_stride,
    SM120MegaMoESharedStorage<kBlockM>& shared,
    const uint32_t channel_start, float (&gate_accum)[kNumMChunks][4], float (&up_accum)[kNumMChunks][4]
) {
    constexpr uint32_t kBlockN = SM120MegaMoESharedStorage<kBlockM>::kBlockN;
    constexpr uint32_t kNumThreads = kBlockM == 16 ? 128 : 256;
    constexpr uint32_t kNumMGroups = kNumThreads / (kBlockN * 4);
    constexpr uint32_t kNumStages = SM120MegaMoESharedStorage<kBlockM>::kNumStages;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t warp_n = warp_idx % (kBlockN / 8);
    const uint32_t warp_m = warp_idx / (kBlockN / 8);
    const uint32_t group_idx = lane_idx / 4;
    const uint32_t thread_in_group = lane_idx % 4;

    #pragma unroll
    for (uint32_t m_chunk = 0; m_chunk < kNumMChunks; ++m_chunk) {
        #pragma unroll
        for (uint32_t accum_idx = 0; accum_idx < 4; ++accum_idx) {
            gate_accum[m_chunk][accum_idx] = 0;
            up_accum[m_chunk][accum_idx] = 0;
        }
    }
    #pragma unroll
    for (uint32_t stage_idx = 0; stage_idx < kNumStages && stage_idx * 128 < kInputWidth; ++stage_idx)
        sm120_mega_moe_load_stage<kInputWidth, kOutputWidth, kBlockM, kIsL1>(
            buffers, weights, weight_scales, sf_expert_stride, sf_k_stride,
            shared, channel_start, stage_idx * 128, stage_idx);
    for (uint32_t k_start = 0; k_start < kInputWidth; k_start += 128) {
        const uint32_t stage_idx = (k_start / 128) % kNumStages;
        if (k_start + 128 < kInputWidth)
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        else
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        #pragma unroll
        for (uint32_t k_offset = 0; k_offset < 128; k_offset += 32) {
            uint32_t gate_fragment[2], up_fragment[2];
            const char* weight_fragment_ptr = shared.gemm.packed_weights[stage_idx] +
                (warp_n * 8 + group_idx) * SM120MegaMoESharedStorage<kBlockM>::kPackedWeightStride +
                k_offset / 2 + thread_in_group * 2;
            gate_fragment[0] = sm120_mega_moe_expand_fp4(*reinterpret_cast<const uint16_t*>(weight_fragment_ptr));
            gate_fragment[1] = sm120_mega_moe_expand_fp4(*reinterpret_cast<const uint16_t*>(weight_fragment_ptr + 8));
            if constexpr (kIsL1) {
                weight_fragment_ptr += kBlockN * SM120MegaMoESharedStorage<kBlockM>::kPackedWeightStride;
                up_fragment[0] = sm120_mega_moe_expand_fp4(*reinterpret_cast<const uint16_t*>(weight_fragment_ptr));
                up_fragment[1] = sm120_mega_moe_expand_fp4(*reinterpret_cast<const uint16_t*>(weight_fragment_ptr + 8));
            }
            #pragma unroll
            for (uint32_t m_chunk = 0; m_chunk < kNumMChunks; ++m_chunk) {
                const uint32_t row_start = (warp_m + m_chunk * kNumMGroups) * 16;
                uint32_t activation_fragment[4];
                sm120::ldmatrix_x4(activation_fragment[0], activation_fragment[1], activation_fragment[2], activation_fragment[3],
                    shared.gemm.activations[stage_idx] + sm120_mega_moe_smem_offset(
                        row_start + (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8, k_offset + (lane_idx >> 4) * 16));
                const uint8_t activation_scale = static_cast<uint8_t>(
                    shared.gemm.activation_scales[stage_idx][row_start + group_idx + (thread_in_group & 1) * 8] >> (k_offset / 32 * 8));
                mma::sm120::fp8_fp4_mixed_mma_block_scaled(gate_accum[m_chunk], activation_fragment, gate_fragment,
                    activation_scale, static_cast<uint8_t>(shared.gemm.weight_scales[stage_idx][warp_n * 8 + group_idx] >> (k_offset / 32 * 8)));
                if constexpr (kIsL1)
                    mma::sm120::fp8_fp4_mixed_mma_block_scaled(up_accum[m_chunk], activation_fragment, up_fragment,
                        activation_scale, static_cast<uint8_t>(shared.gemm.weight_scales[stage_idx][kBlockN + warp_n * 8 + group_idx] >> (k_offset / 32 * 8)));
            }
        }
        __syncthreads();
        if (k_start + kNumStages * 128 < kInputWidth)
            sm120_mega_moe_load_stage<kInputWidth, kOutputWidth, kBlockM, kIsL1>(
                buffers, weights, weight_scales, sf_expert_stride, sf_k_stride,
                shared, channel_start, k_start + kNumStages * 128, stage_idx);
    }
}

template <uint32_t kHidden, uint32_t kIntermediate, uint32_t kNumExperts,
          uint32_t kTopK, uint32_t kNumCTAs, uint32_t kNumRingSlots, uint32_t kNumRanks,
          uint32_t kBlockM, uint32_t kCTAsPerSM, bool kFastMath>
__global__ __launch_bounds__((kBlockM == 16 ? 128 : 256), kCTAsPerSM)
void sm120_fp8_fp4_mega_moe_impl(
    nv_bfloat16* output,
    const uint8_t* l1_weights, const uint8_t* l2_weights,
    const uint32_t* l1_weight_scales, const uint32_t* l2_weight_scales,
    const int64_t l1_sf_expert_stride, const int64_t l1_sf_k_stride,
    const int64_t l2_sf_expert_stride, const int64_t l2_sf_k_stride,
    int* cumulative_recv_stats, const uint32_t num_tokens,
    const float activation_clamp, const float activation_alpha, const float activation_beta,
    const __grid_constant__ layout::MegaMoEBuffer buffers,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer
) {
    constexpr uint32_t kBlockN = SM120MegaMoESharedStorage<kBlockM>::kBlockN;
    constexpr uint32_t kNumLocalExperts = kNumExperts / kNumRanks;
    constexpr uint32_t kNumL1Tiles = kIntermediate / kBlockN;
    constexpr uint32_t kNumL2Tiles = kHidden / kBlockN;
    constexpr uint32_t kTasksPerTile = 1 + kNumL1Tiles + kNumL2Tiles;
    constexpr bool kStagePeerOutput = kNumRanks > 1;
    constexpr bool kUsePeerPull = kNumRanks > 2;
    constexpr bool kUseSourceLocalMap = kUsePeerPull and kNumRanks <= 4;
    constexpr bool kUsePhasedScheduler = kBlockM != 16 or kStagePeerOutput;
    constexpr uint32_t kNumPhases = kStagePeerOutput ? 4 : 3;
    constexpr uint32_t kNumThreads = kBlockM == 16 ? 128 : 256;
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    constexpr uint32_t kNumMGroups = kNumThreads / (kBlockN * 4);
    constexpr uint32_t kNumMChunks = kBlockM / (16 * kNumMGroups);
    DG_STATIC_ASSERT(kBlockM == 16 or kBlockM == 32 or kBlockM == 64, "Invalid SM120 MegaMoE tile");
    DG_STATIC_ASSERT(kTopK > 0 and kTopK <= kNumExperts, "Invalid SM120 MegaMoE top-k");
    DG_STATIC_ASSERT(kNumRingSlots > 0 and kNumRingSlots <= kNumCTAs, "Invalid SM120 MegaMoE ring");
    if constexpr (kStagePeerOutput)
        DG_DEVICE_ASSERT(kNumRingSlots * kBlockM * 2 <= buffers.l1_token_buffer.num_max_tokens_per_rank);
    __shared__ SM120MegaMoESharedStorage<kBlockM> shared;
    __shared__ uint32_t peer_output_capacity[kNumRanks];
    __shared__ uint32_t num_batches;
    const auto& workspace = buffers.workspace;
    auto* const source_pull_rows = workspace.get_src_token_topk_idx_ptr();
    auto* const cached_peer_capacity = workspace.get_l1_full_count_ptr(kNumLocalExperts + 1);
    auto* const cached_num_batches = cached_peer_capacity + kNumRanks;
    const uint32_t global_thread = blockIdx.x * kNumThreads + threadIdx.x;
    const uint32_t thread_stride = kNumCTAs * kNumThreads;
    const uint32_t first_expert = sym_buffer.rank_idx * kNumLocalExperts;
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t warp_idx = threadIdx.x / 32;
    const uint32_t warp_n = warp_idx % (kBlockN / 8);
    const uint32_t warp_m = warp_idx / (kBlockN / 8);
    const uint32_t group_idx = lane_idx / 4;
    const uint32_t thread_in_group = lane_idx % 4;

    for (uint32_t expert_idx = global_thread; expert_idx < kNumLocalExperts; expert_idx += thread_stride)
        *workspace.get_expert_recv_count_sum_ptr(expert_idx) = 0;
    if (threadIdx.x == 0) {
        *workspace.get_l1_empty_count_ptr(blockIdx.x) = 0;
        *workspace.get_l2_empty_count_ptr(blockIdx.x) = 0;
        *workspace.get_l2_full_mask_ptr(blockIdx.x) = 0;
    }
    if (global_thread == 0) {
        *workspace.get_shared_l1_task_count_ptr() = num_tokens;
        if constexpr (kUsePeerPull)
            *workspace.get_shared_l2_task_count_ptr() = kNumRingSlots * kBlockM;
    }
    sm120_mega_moe_peer_barrier<kNumRanks, kNumCTAs>(workspace, sym_buffer);

    for (uint32_t rank_idx = 0; rank_idx < kNumRanks; ++rank_idx) {
        const uint32_t peer_tokens = *sym_buffer.map(workspace.get_shared_l1_task_count_ptr(), rank_idx);
        const auto* routes = sym_buffer.map(buffers.input_topk_idx_buffer.get_base_ptr<int64_t>(), rank_idx);
        for (uint32_t route_idx = global_thread; route_idx < peer_tokens * kTopK; route_idx += thread_stride) {
            const int64_t expert_idx = routes[route_idx];
            if (expert_idx >= first_expert && expert_idx < first_expert + kNumLocalExperts)
                atomicAdd(reinterpret_cast<unsigned long long*>(workspace.get_expert_recv_count_sum_ptr(expert_idx - first_expert)), 1ull);
        }
    }
    sm120_mega_moe_grid_sync<kNumCTAs>(workspace);
    if (global_thread < 32) {
        uint32_t total_tiles = 0;
        for (uint32_t expert_base = 0; expert_base < kNumLocalExperts; expert_base += 32) {
            const uint32_t expert_idx = expert_base + lane_idx;
            const uint32_t count = expert_idx < kNumLocalExperts ?
                *workspace.get_expert_recv_count_sum_ptr(expert_idx) : 0;
            const uint32_t num_expert_tiles = math::ceil_div(count, kBlockM);
            uint32_t tile_prefix = num_expert_tiles;
            #pragma unroll
            for (uint32_t offset = 1; offset < 32; offset <<= 1) {
                const uint32_t previous = __shfl_up_sync(0xffffffff, tile_prefix, offset);
                if (lane_idx >= offset)
                    tile_prefix += previous;
            }
            if (expert_idx < kNumLocalExperts) {
                const uint32_t tile_start = total_tiles + tile_prefix - num_expert_tiles;
                *workspace.get_expert_send_count_ptr(expert_idx) = tile_start * kBlockM;
                *workspace.get_expert_recv_count_ptr(0, expert_idx) = 0;
                *workspace.get_l1_full_count_ptr(expert_idx) = tile_start;
                if (cumulative_recv_stats != nullptr)
                    cumulative_recv_stats[expert_idx] += count;
            }
            total_tiles += __shfl_sync(0xffffffff, tile_prefix, 31);
        }
        if (lane_idx == 0) {
            *workspace.get_l1_full_count_ptr(kNumLocalExperts) = total_tiles;
            *workspace.get_l1_task_count_ptr() = kUsePeerPull ? math::ceil_div(total_tiles, kNumRingSlots) : 0;
            DG_DEVICE_ASSERT(total_tiles <= 0xffffffffu / kTasksPerTile);
            *workspace.get_l2_task_count_ptr() = total_tiles * kTasksPerTile;
            DG_DEVICE_ASSERT(static_cast<uint64_t>(total_tiles) * kBlockM <= workspace.num_max_pool_tokens);
        }
    }
    sm120_mega_moe_grid_sync<kNumCTAs>(workspace);
    for (uint32_t rank_idx = 0; rank_idx < kNumRanks; ++rank_idx) {
        const uint32_t peer_tokens = *sym_buffer.map(workspace.get_shared_l1_task_count_ptr(), rank_idx);
        const auto* routes = sym_buffer.map(buffers.input_topk_idx_buffer.get_base_ptr<int64_t>(), rank_idx);
        for (uint32_t route_idx = global_thread; route_idx < peer_tokens * kTopK; route_idx += thread_stride) {
            const int64_t expert_idx = routes[route_idx];
            if (expert_idx >= first_expert && expert_idx < first_expert + kNumLocalExperts) {
                const uint32_t local_expert = expert_idx - first_expert;
                const uint32_t offset = atomicAdd(reinterpret_cast<unsigned long long*>(workspace.get_expert_recv_count_ptr(0, local_expert)), 1ull);
                const uint32_t pool_start = *workspace.get_expert_send_count_ptr(local_expert);
                *workspace.get_token_src_metadata_ptr(pool_start + offset) =
                    layout::TokenSrcMetadata(rank_idx, route_idx / kTopK, route_idx % kTopK);
                if constexpr (kUsePeerPull) {
                    if constexpr (kUseSourceLocalMap)
                        *sym_buffer.map(source_pull_rows + route_idx, rank_idx) = pool_start + offset;
                    else
                        *workspace.get_src_token_topk_idx_ptr(local_expert, rank_idx, route_idx / kTopK) = pool_start + offset;
                }
            }
        }
    }
    if constexpr (kUsePeerPull)
        sm120_mega_moe_peer_barrier<kNumRanks, kNumCTAs, true>(workspace, sym_buffer);
    else
        sm120_mega_moe_grid_sync<kNumCTAs>(workspace);

    const uint32_t num_tiles = *workspace.get_l1_full_count_ptr(kNumLocalExperts);
    if constexpr (kUsePeerPull) {
        if (threadIdx.x == 0)
            num_batches = *cached_num_batches;
        if (threadIdx.x < kNumRanks)
            peer_output_capacity[threadIdx.x] = cached_peer_capacity[threadIdx.x];
    } else if (threadIdx.x == 0)
        num_batches = math::ceil_div(num_tiles, kNumRingSlots);
    __syncthreads();
    uint32_t batch_idx = 0, batch_start = 0, phase_idx = 0, task_cursor = blockIdx.x;
    while (not kUsePhasedScheduler or batch_idx < num_batches) {
        if constexpr (kUsePeerPull) {
            if (phase_idx == 3) {
                const auto* routes = buffers.input_topk_idx_buffer.get_base_ptr<int64_t>();
                for (uint32_t route_idx = blockIdx.x * kNumWarps + warp_idx;
                     route_idx < num_tokens * kTopK; route_idx += kNumCTAs * kNumWarps) {
                    uint32_t peer_rank = 0, output_row = 0, active = 0;
                    if (lane_idx == 0) {
                        const int64_t expert_idx = routes[route_idx];
                        if (expert_idx >= 0 and expert_idx < kNumExperts) {
                            peer_rank = expert_idx / kNumLocalExperts;
                            uint32_t logical_row;
                            if constexpr (kUseSourceLocalMap)
                                logical_row = ptx::ld_acq_sys(source_pull_rows + route_idx);
                            else
                                logical_row = ptx::ld_acq_sys(sym_buffer.map(
                                    workspace.get_src_token_topk_idx_ptr(expert_idx % kNumLocalExperts,
                                        sym_buffer.rank_idx, route_idx / kTopK), peer_rank));
                            const uint32_t capacity = peer_output_capacity[peer_rank];
                            active = logical_row / capacity == batch_idx;
                            output_row = logical_row % capacity;
                        }
                    }
                    active = __shfl_sync(0xffffffff, active, 0);
                    if (active) {
                        peer_rank = __shfl_sync(0xffffffff, peer_rank, 0);
                        output_row = __shfl_sync(0xffffffff, output_row, 0);
                        const auto* staged_output = sym_buffer.map(
                            buffers.l1_token_buffer.get_base_ptr<uint4>(), peer_rank);
                        auto* destination = buffers.combine_token_buffer.get_rank_buffer(route_idx % kTopK)
                            .get_data_buffer(route_idx / kTopK).template get_base_ptr<uint4>();
                        for (uint32_t col = lane_idx; col < kHidden / 8; col += 32)
                            destination[col] = staged_output[static_cast<int64_t>(output_row) * (kHidden / 8) + col];
                    }
                }
                sm120_mega_moe_peer_barrier<kNumRanks, kNumCTAs>(workspace, sym_buffer);
                phase_idx = 0;
                task_cursor = blockIdx.x;
                batch_start += kNumRingSlots;
                ++batch_idx;
                continue;
            }
        }
        const uint32_t num_batch_tiles = batch_start < num_tiles ?
            math::constexpr_min(kNumRingSlots, num_tiles - batch_start) : 0;
        const uint32_t tasks_per_phase_tile = phase_idx == 1 ? kNumL1Tiles : phase_idx == 2 ? kNumL2Tiles : 1;
        const uint32_t num_phase_tasks = kUsePhasedScheduler ? num_batch_tiles * tasks_per_phase_tile
                                                          : *workspace.get_l2_task_count_ptr();
        if (threadIdx.x == 0) {
            if constexpr (not kUsePhasedScheduler)
                shared.task_idx = atomicAdd(workspace.get_l1_task_count_ptr(), 1u);
            else
                shared.task_idx = task_cursor;
        }
        task_cursor += kNumCTAs;
        __syncthreads();
        if (shared.task_idx >= num_phase_tasks) {
            if constexpr (not kUsePhasedScheduler)
                break;
            else {
                if constexpr (kUsePeerPull) {
                    if (phase_idx == 2)
                        sm120_mega_moe_peer_barrier<kNumRanks, kNumCTAs>(workspace, sym_buffer);
                    else
                        sm120_mega_moe_grid_sync<kNumCTAs>(workspace);
                } else
                    sm120_mega_moe_grid_sync<kNumCTAs>(workspace);
                task_cursor = blockIdx.x;
                if (++phase_idx == kNumPhases) {
                    phase_idx = 0;
                    batch_start += kNumRingSlots;
                    ++batch_idx;
                }
                continue;
            }
        }
        if (threadIdx.x == 0) {
            if constexpr (not kUsePhasedScheduler) {
                shared.tile_idx = shared.task_idx / kTasksPerTile;
                shared.task_offset = shared.task_idx % kTasksPerTile;
            } else {
                shared.tile_idx = batch_start + shared.task_idx / tasks_per_phase_tile;
                shared.task_offset = phase_idx == 0 ? 0 : phase_idx == 3 ? kTasksPerTile :
                    (phase_idx == 1 ? 1 : kNumL1Tiles + 1) + shared.task_idx % tasks_per_phase_tile;
            }
            shared.ring_idx = shared.tile_idx % kNumRingSlots;
            uint32_t lower = 0, upper = kNumLocalExperts;
            while (lower + 1 < upper) {
                const uint32_t middle = (lower + upper) / 2;
                if (*workspace.get_l1_full_count_ptr(middle) <= shared.tile_idx)
                    lower = middle;
                else
                    upper = middle;
            }
            shared.expert_idx = lower;
            const uint32_t token_start = (shared.tile_idx - *workspace.get_l1_full_count_ptr(lower)) * kBlockM;
            shared.num_tokens = math::constexpr_min(kBlockM, static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(lower)) - token_start);
            shared.pool_start = *workspace.get_expert_send_count_ptr(lower) + token_start;
        }
        __syncthreads();
        if (shared.task_offset == 0 or
            (not kStagePeerOutput and shared.task_offset > kNumL1Tiles) or
            (kStagePeerOutput and not kUsePeerPull and shared.task_offset == kTasksPerTile)) {
            if (threadIdx.x < kBlockM) {
                const bool valid = threadIdx.x < shared.num_tokens;
                shared.sources[threadIdx.x] = valid ? *workspace.get_token_src_metadata_ptr(shared.pool_start + threadIdx.x)
                                                   : layout::TokenSrcMetadata(0, 0, 0);
            }
            __syncthreads();
        }
        const uint64_t epoch = shared.tile_idx / kNumRingSlots + 1;
        if (shared.task_offset == 0) {
            if (threadIdx.x == 0) {
                if (epoch > 1) {
                    comm::wait_until([&]() {
                        return ptx::ld_acq_sys(workspace.get_l2_full_mask_ptr(shared.ring_idx)) == epoch - 1 &&
                               ptx::ld_acq(workspace.get_l2_empty_count_ptr(shared.ring_idx)) == kNumL2Tiles;
                    }, [&]() {
                        printf("SM120 MegaMoE ring reuse timeout: slot=%u epoch=%llu\n",
                               shared.ring_idx, static_cast<unsigned long long>(epoch));
                    });
                }
                *workspace.get_l1_empty_count_ptr(shared.ring_idx) = 0;
                *workspace.get_l2_empty_count_ptr(shared.ring_idx) = 0;
            }
            __syncthreads();
            if (threadIdx.x < kBlockM) {
                const auto source = shared.sources[threadIdx.x];
                const auto* routing = sym_buffer.map(buffers.input_topk_weights_buffer.get_base_ptr<float>(), source.rank_idx);
                buffers.l1_topk_weights_buffer.get_base_ptr<float>()[shared.ring_idx * kBlockM + threadIdx.x] =
                    threadIdx.x < shared.num_tokens ? routing[source.token_idx * kTopK + source.topk_idx] : 0;
            }
            __syncthreads();
            for (uint32_t copy_idx = threadIdx.x; copy_idx < kBlockM * kHidden / 16; copy_idx += kNumThreads) {
                const uint32_t row = copy_idx / (kHidden / 16);
                const uint32_t col = copy_idx % (kHidden / 16);
                const auto source = shared.sources[row];
                const auto* input = sym_buffer.map(buffers.input_token_buffer.get_base_ptr<uint4>(), source.rank_idx);
                auto* destination = buffers.l1_token_buffer.get_base_ptr<uint4>() +
                    (shared.ring_idx * kBlockM + row) * (kHidden / 16) + col;
                *destination = row < shared.num_tokens ? input[source.token_idx * (kHidden / 16) + col] : make_uint4(0, 0, 0, 0);
            }
            for (uint32_t copy_idx = threadIdx.x; copy_idx < kBlockM * (kHidden / 128); copy_idx += kNumThreads) {
                const uint32_t row = copy_idx / (kHidden / 128);
                const uint32_t col = copy_idx % (kHidden / 128);
                const auto source = shared.sources[row];
                const auto* input = sym_buffer.map(buffers.input_sf_buffer.get_base_ptr<uint32_t>(), source.rank_idx);
                const uint32_t sf_row = shared.ring_idx * 128 + sm120_mega_moe_sf_row(row);
                buffers.l1_sf_buffer.get_base_ptr<uint32_t>()[static_cast<int64_t>(col) * buffers.l1_sf_buffer.num_max_tokens_per_rank + sf_row] =
                    row < shared.num_tokens ? input[source.token_idx * (kHidden / 128) + col] : 0x7f7f7f7f;
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0)
                ptx::st_rel(workspace.get_l2_full_mask_ptr(shared.ring_idx), epoch);
            __syncthreads();
            continue;
        }
        if (threadIdx.x == 0) {
            comm::wait_until([&]() {
                return ptx::ld_acq_sys(workspace.get_l2_full_mask_ptr(shared.ring_idx)) == epoch &&
                       (shared.task_offset <= kNumL1Tiles ||
                        ptx::ld_acq(workspace.get_l1_empty_count_ptr(shared.ring_idx)) == kNumL1Tiles);
            }, [&]() {
                printf("SM120 MegaMoE tile readiness timeout: slot=%u epoch=%llu task=%u\n",
                       shared.ring_idx, static_cast<unsigned long long>(epoch), shared.task_offset);
            });
        }
        __syncthreads();
        if constexpr (kStagePeerOutput and not kUsePeerPull) {
            if (shared.task_offset == kTasksPerTile) {
                const auto* staged_output = buffers.l1_token_buffer.get_base_ptr<uint4>();
                for (uint32_t copy_idx = threadIdx.x; copy_idx < shared.num_tokens * (kHidden / 8); copy_idx += kNumThreads) {
                    const uint32_t row = copy_idx / (kHidden / 8);
                    const uint32_t col = copy_idx % (kHidden / 8);
                    const auto source = shared.sources[row];
                    auto* destination = sym_buffer.map(buffers.combine_token_buffer.get_rank_buffer(source.topk_idx)
                        .get_data_buffer(source.token_idx).template get_base_ptr<uint4>(), source.rank_idx);
                    destination[col] = staged_output[static_cast<int64_t>(shared.ring_idx * kBlockM + row) * (kHidden / 8) + col];
                }
                __threadfence_system();
                __syncthreads();
                continue;
            }
        }
        float gate_accum[kNumMChunks][4], up_accum[kNumMChunks][4];
        if (shared.task_offset <= kNumL1Tiles) {
            const uint32_t channel_start = (shared.task_offset - 1) * kBlockN;
            if (threadIdx.x < kBlockM)
                shared.routing_weights[threadIdx.x] = buffers.l1_topk_weights_buffer.get_base_ptr<float>()[
                    shared.ring_idx * kBlockM + threadIdx.x];
            __syncthreads();
            sm120_mega_moe_gemm<kHidden, kIntermediate, kBlockM, kNumMChunks, true>(buffers, l1_weights, l1_weight_scales,
                l1_sf_expert_stride, l1_sf_k_stride, shared, channel_start, gate_accum, up_accum);
            #pragma unroll
            for (uint32_t m_chunk = 0; m_chunk < kNumMChunks; ++m_chunk) {
                #pragma unroll
                for (uint32_t accum_idx = 0; accum_idx < 4; ++accum_idx) {
                    const uint32_t row = (warp_m + m_chunk * kNumMGroups) * 16 + group_idx + (accum_idx / 2) * 8;
                    const uint32_t col = warp_n * 8 + thread_in_group * 2 + accum_idx % 2;
                    const float gate = fminf(__bfloat162float(__float2bfloat16_rn(gate_accum[m_chunk][accum_idx])), activation_clamp);
                    const float up = fminf(fmaxf(__bfloat162float(__float2bfloat16_rn(up_accum[m_chunk][accum_idx])),
                                               -activation_clamp), activation_clamp);
                    const float exponential = kFastMath ? __expf(-activation_alpha * gate) : expf(-activation_alpha * gate);
                    shared.activation.values[sm120_mega_moe_activation_offset(row, col, kBlockN)] =
                        gate / (1.0f + exponential) * (up + activation_beta) * shared.routing_weights[row];
                }
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t row_start = 0; row_start < kBlockM; row_start += kNumWarps) {
                const uint32_t row = row_start + warp_idx;
                #pragma unroll
                for (uint32_t channel_offset = 0; channel_offset < kBlockN; channel_offset += 32) {
                    const uint32_t channel = channel_start + channel_offset;
                    const float value = shared.activation.values[
                        sm120_mega_moe_activation_offset(row, channel_offset + lane_idx, kBlockN)];
                    float amax = fabsf(value);
                    #pragma unroll
                    for (uint32_t offset = 16; offset > 0; offset >>= 1)
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
                    const uint32_t sf_exp = math::get_ue8m0_sf_exp(fmaxf(amax, 1e-4f));
                    const float sf_inv = math::get_ue8m0_sf_inv<float>(sf_exp);
                    buffers.l2_token_buffer.get_base_ptr<cutlass::float_e4m3_t>()[
                        static_cast<int64_t>(shared.ring_idx * kBlockM + row) * kIntermediate + channel + lane_idx] = cutlass::float_e4m3_t(value * sf_inv);
                    if (lane_idx == 0) {
                        const uint32_t sf_row = shared.ring_idx * 128 + sm120_mega_moe_sf_row(row);
                        const int64_t sf_offset = (static_cast<int64_t>(channel / 128) * buffers.l2_sf_buffer.num_max_tokens_per_rank + sf_row) * 4 + channel / 32 % 4;
                        buffers.l2_sf_buffer.get_base_ptr<uint8_t>()[sf_offset] = static_cast<uint8_t>(sf_exp);
                    }
                }
            }
            sm120_mega_moe_complete_tile<false>(workspace.get_l1_empty_count_ptr(shared.ring_idx));
        } else {
            const uint32_t channel_start = (shared.task_offset - 1 - kNumL1Tiles) * kBlockN;
            sm120_mega_moe_gemm<kIntermediate, kHidden, kBlockM, kNumMChunks, false>(buffers, l2_weights, l2_weight_scales,
                l2_sf_expert_stride, l2_sf_k_stride, shared, channel_start, gate_accum, up_accum);
            #pragma unroll
            for (uint32_t m_chunk = 0; m_chunk < kNumMChunks; ++m_chunk) {
                #pragma unroll
                for (uint32_t row_half = 0; row_half < 2; ++row_half) {
                    const uint32_t row = (warp_m + m_chunk * kNumMGroups) * 16 + group_idx + row_half * 8;
                    if (row < shared.num_tokens) {
                        nv_bfloat16* destination;
                        if constexpr (kStagePeerOutput)
                            destination = buffers.l1_token_buffer.get_base_ptr<nv_bfloat16>() +
                                static_cast<int64_t>(shared.ring_idx * kBlockM + row) * kHidden;
                        else {
                            const auto source = shared.sources[row];
                            destination = sym_buffer.map(buffers.combine_token_buffer.get_rank_buffer(source.topk_idx)
                                .get_data_buffer(source.token_idx).template get_base_ptr<nv_bfloat16>(), source.rank_idx);
                        }
                        const uint32_t col = channel_start + warp_n * 8 + thread_in_group * 2;
                        *reinterpret_cast<nv_bfloat162*>(destination + col) =
                            __floats2bfloat162_rn(gate_accum[m_chunk][row_half * 2], gate_accum[m_chunk][row_half * 2 + 1]);
                    }
                }
            }
            sm120_mega_moe_complete_tile<not kStagePeerOutput>(workspace.get_l2_empty_count_ptr(shared.ring_idx));
        }
    }
    if constexpr (not kUsePeerPull)
        sm120_mega_moe_peer_barrier<kNumRanks, kNumCTAs>(workspace, sym_buffer);
    const auto* routes = buffers.input_topk_idx_buffer.get_base_ptr<int64_t>();
    for (int64_t output_idx = global_thread;
         output_idx < static_cast<int64_t>(num_tokens) * kHidden; output_idx += thread_stride) {
        const uint32_t token_idx = output_idx / kHidden;
        const uint32_t channel_idx = output_idx % kHidden;
        float value = 0;
        #pragma unroll
        for (uint32_t topk_idx = 0; topk_idx < kTopK; ++topk_idx) {
            const int64_t expert_idx = routes[token_idx * kTopK + topk_idx];
            if (expert_idx >= 0 && expert_idx < kNumExperts)
                value += __bfloat162float(buffers.combine_token_buffer.get_rank_buffer(topk_idx)
                    .get_data_buffer(token_idx).get_base_ptr<nv_bfloat16>()[channel_idx]);
        }
        output[output_idx] = __float2bfloat16_rn(value);
    }
}

}

#endif
