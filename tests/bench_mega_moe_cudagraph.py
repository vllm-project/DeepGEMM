import argparse
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_nvfp4
from deep_gemm.utils.dist import dist_print, init_dist


def _make_weights(mma_type: str, num_experts_per_rank: int, hidden: int, intermediate_hidden: int,
                  transform: bool = True):
    def cast_mx(num_rows, num_columns):
        weights = torch.empty((num_experts_per_rank, num_rows, num_columns // 2), device='cuda', dtype=torch.int8)
        weight_scales = torch.empty((num_experts_per_rank, num_rows, num_columns // 32), device='cuda', dtype=torch.float)
        for expert_idx in range(num_experts_per_rank):
            bf16_weights = 0.1 * torch.randn((num_rows, num_columns), dtype=torch.bfloat16, device='cuda')
            weights[expert_idx], weight_scales[expert_idx] = per_token_cast_to_fp4(bf16_weights, use_ue8m0=True, gran_k=32)
        weight_scales = deep_gemm.transform_sf_into_required_layout(
            weight_scales, num_rows, num_columns, (1, 32), num_experts_per_rank)
        return weights, weight_scales

    def cast_nv(num_rows, num_columns):
        weights = torch.empty((num_experts_per_rank, num_rows, num_columns // 2), device='cuda', dtype=torch.int8)
        weight_scales = torch.empty((num_experts_per_rank, num_rows, num_columns // 64), device='cuda', dtype=torch.int32)
        for expert_idx in range(num_experts_per_rank):
            bf16_weights = 0.1 * torch.randn((num_rows, num_columns), dtype=torch.bfloat16, device='cuda')
            weights[expert_idx], weight_scales[expert_idx] = per_token_cast_to_nvfp4(bf16_weights, gran_k=16, use_packed_e4m3=True)
        return weights, weight_scales.transpose(-1, -2).contiguous().transpose(-1, -2)

    cast = cast_mx if mma_type == 'fp8xfp4' else cast_nv
    l1_weights, l2_weights = cast(intermediate_hidden * 2, hidden), cast(hidden, intermediate_hidden)
    torch.cuda.empty_cache()
    return deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights) if transform else (l1_weights, l2_weights)


def _make_unfused_runner(x_data, x_sf, topk_idx, topk_weights, l1_weights, l2_weights, activation_clamp,
                         implementation='triton'):
    if implementation == 'triton':
        from bench_mega_moe_triton import combine_routed, dispatch_quantized, swiglu_quantized
    num_tokens, hidden = x_data.shape
    num_topk = topk_idx.size(1)
    num_experts, doubled_intermediate, _ = l1_weights[0].shape
    intermediate = doubled_intermediate // 2
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    order = torch.argsort(topk_idx.flatten())
    counts = torch.bincount(topk_idx.flatten(), minlength=num_experts).tolist()
    padded_counts = [(count + alignment - 1) // alignment * alignment for count in counts]
    num_routed = sum(padded_counts)
    token_indices = torch.zeros(num_routed, dtype=torch.long, device='cuda')
    route_rows = torch.empty(num_tokens * num_topk, dtype=torch.long, device='cuda')
    routing_weights = torch.zeros(num_routed, dtype=torch.float, device='cuda')
    labels = torch.empty(num_routed, dtype=torch.int32, device='cuda')
    route_start, row_start = 0, 0
    for expert_idx, (count, padded) in enumerate(zip(counts, padded_counts)):
        route_order = order[route_start:route_start + count]
        token_indices[row_start:row_start + count] = route_order // num_topk
        routing_weights[row_start:row_start + count] = topk_weights.flatten()[route_order]
        route_rows[route_order] = torch.arange(row_start, row_start + count, device='cuda')
        labels[row_start:row_start + padded] = expert_idx
        route_start += count
        row_start += padded
    routed_inputs = torch.empty((num_routed, hidden), dtype=x_data.dtype, device='cuda')
    routed_input_scales = torch.empty((hidden // 128, num_routed), dtype=torch.int32, device='cuda').t()
    gate_up = torch.empty((num_routed, intermediate * 2), dtype=torch.bfloat16, device='cuda')
    intermediate_inputs = torch.empty((num_routed, intermediate), dtype=x_data.dtype, device='cuda')
    intermediate_scales = torch.empty((intermediate // 128, num_routed), dtype=torch.int32, device='cuda').t()
    expert_outputs = torch.empty((num_routed, hidden), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def dispatch():
        if implementation == 'triton':
            dispatch_quantized(x_data, x_sf, token_indices, routed_inputs, routed_input_scales)
        else:
            routed_inputs.view(torch.uint8).copy_(x_data.view(torch.uint8)[token_indices])
            routed_input_scales.copy_(x_sf[token_indices])

    def gemm1():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (routed_inputs, routed_input_scales), l1_weights, gate_up, labels, recipe=(1, 1, 32))

    def activation_quantization():
        if implementation == 'triton':
            swiglu_quantized(gate_up, routing_weights, intermediate_inputs, intermediate_scales, activation_clamp)
            return
        gate, up = gate_up.float().chunk(2, dim=-1)
        if activation_clamp is not None:
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(-activation_clamp, activation_clamp)
        values = torch.nn.functional.silu(gate) * up * routing_weights[:, None]
        quantized, scales = per_token_cast_to_fp8(values, use_ue8m0=True, gran_k=32)
        packed_scales = (scales.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)
        intermediate_inputs.copy_(quantized)
        intermediate_scales.copy_(packed_scales)

    def gemm2():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (intermediate_inputs, intermediate_scales), l2_weights, expert_outputs, labels, recipe=(1, 1, 32))

    def combine():
        if implementation == 'triton':
            combine_routed(expert_outputs, route_rows, output, num_topk)
        else:
            output.copy_(expert_outputs[route_rows].view(num_tokens, num_topk, hidden).float().sum(dim=1))

    phases = dict(dispatch=dispatch, gemm1=gemm1, activation_quantization=activation_quantization,
                  gemm2=gemm2, combine=combine)

    def run():
        for phase in phases.values():
            phase()
        return output

    return run, phases, output


def _time_single_rank_graph(launch, num_reps):
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(5):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_reps):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / num_reps


# noinspection PyShadowingNames
def bench_mma_type(mma_type: str, args, group, rank_idx: int, num_ranks: int, token_counts):
    torch.manual_seed(args.seed + rank_idx)
    num_experts, num_topk = args.num_experts, args.num_topk
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = max(token_counts)

    # BF16 shared experts are only supported by the NVFP4 kernel
    num_shared_experts = args.num_shared_experts if mma_type == 'fp4xfp4' else 0
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_shared_experts=num_shared_experts, mma_type=mma_type)
    raw_l1_weights, raw_l2_weights = _make_weights(mma_type, num_experts_per_rank, hidden, intermediate_hidden, transform=False)
    l1_weights, l2_weights = deep_gemm.transform_weights_for_mega_moe(raw_l1_weights, raw_l2_weights)

    # BF16 shared expert weights (folded on N for L1, K for L2)
    shared_l1_weights, shared_l2_weights = None, None
    if num_shared_experts > 0:
        sih = intermediate_hidden * num_shared_experts
        shared_l1_weights, shared_l2_weights = deep_gemm.transform_weights_for_mega_moe(
            0.1 * torch.randn((sih * 2, hidden), dtype=torch.bfloat16, device='cuda'),
            0.1 * torch.randn((hidden, sih), dtype=torch.bfloat16, device='cuda'))

    # Inputs at max size (per-rank seed for routing diversity)
    torch.manual_seed(args.seed + rank_idx)
    x = torch.randn((num_max_tokens_per_rank, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_max_tokens_per_rank, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_weights = topk_weights.softmax(dim=-1)
    if mma_type == 'fp8xfp4':
        x_data, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    else:
        x_data, x_sf = per_token_cast_to_nvfp4(x, gran_k=16, use_packed_e4m3=True)
        x_data = x_data.view(torch.uint8)
    kernel_fn = deep_gemm.fp8_fp4_mega_moe if mma_type == 'fp8xfp4' else deep_gemm.fp4_fp4_mega_moe

    # Count per-size routed tokens on this rank (for FLOP stats)
    gathered_topk = [torch.empty_like(topk_idx) for _ in range(num_ranks)]
    dist.all_gather(gathered_topk, topk_idx, group=group)

    results = {}
    for num_tokens in token_counts:
        buffer.x[:num_tokens].copy_(x_data[:num_tokens])
        buffer.x_sf[:num_tokens].copy_(x_sf[:num_tokens])
        buffer.topk_idx[:num_tokens].copy_(topk_idx[:num_tokens])
        buffer.topk_weights[:num_tokens].copy_(topk_weights[:num_tokens])
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

        x_bf16 = x[:num_tokens] if num_shared_experts > 0 else None
        kernel_kwargs = dict(activation_clamp=args.activation_clamp, fast_math=True)
        if args.num_tokens_for_block_m is not None:
            kernel_kwargs['num_tokens_for_block_m'] = args.num_tokens_for_block_m
        if torch.cuda.get_device_capability()[0] == 12:
            selected_block_m = deep_gemm.get_block_m_for_mega_moe(
                num_ranks, num_experts, num_max_tokens_per_rank,
                args.num_tokens_for_block_m if args.num_tokens_for_block_m is not None else num_tokens,
                num_topk, mma_type)
            selected_block_n = 64 if selected_block_m == 64 else 32
            dist_print(f' > SM120 tile: BLOCK_M={selected_block_m}, BLOCK_N={selected_block_n}, BLOCK_K=128',
                       once_in_node=True)

        def launch():
            if num_shared_experts > 0:
                kernel_fn(y=y, l1_weights=l1_weights, l2_weights=l2_weights,
                          sym_buffer=buffer, shared_l1_weights=shared_l1_weights,
                          shared_l2_weights=shared_l2_weights, x_bf16=x_bf16, fast_math=True)
            else:
                kernel_fn(y=y, l1_weights=l1_weights, l2_weights=l2_weights,
                          sym_buffer=buffer, **kernel_kwargs)

        # Warmup (includes JIT compilation for this token count's config)
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        dist.barrier()

        # Capture a single kernel launch into a CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()

        # Warm replays, then timed replays (CPU only enqueues graph launches)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        num_reps = args.num_reps if num_tokens >= 1024 else args.num_reps * 4
        start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(num_reps):
            graph.replay()
        end_ev.record()
        torch.cuda.synchronize()
        t_us = start_ev.elapsed_time(end_ev) * 1e3 / num_reps

        # Aggregate across ranks: wall time is the slowest rank
        t_tensor = torch.tensor([t_us], dtype=torch.float64, device='cuda')
        dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX, group=group)
        t_max_us = t_tensor.item()
        if args.compare_unfused:
            from test_mega_moe import _calc_token_diff
            unfused, phases, unfused_output = _make_unfused_runner(
                x_data[:num_tokens], x_sf[:num_tokens], topk_idx[:num_tokens], topk_weights[:num_tokens],
                raw_l1_weights, raw_l2_weights, args.activation_clamp, args.unfused_impl)
            unfused()
            diff = calc_diff(y, unfused_output)
            max_token_diff = _calc_token_diff(y, unfused_output).max().item()
            assert diff < 1e-5 and max_token_diff < 1e-4, \
                f'Non-fused reference differs: calc_diff={diff}, max_token_diff={max_token_diff}'
            unfused_us = _time_single_rank_graph(unfused, num_reps)
            phase_times = {name: _time_single_rank_graph(phase, num_reps) for name, phase in phases.items()}
            dist_print(f' > Non-fused composition: precomputed routing, {args.unfused_impl} dispatch/activation/combine, single rank; '
                       f'{unfused_us:.1f} us vs fused {t_max_us:.1f} us; calc_diff={diff:.3e}, '
                       f'max_token_diff={max_token_diff:.3e}', once_in_node=True)
            dist_print(' > Phase us: ' + ', '.join(f'{name}={elapsed:.1f}' for name, elapsed in phase_times.items()), once_in_node=True)

        lo, hi = rank_idx * num_experts_per_rank, (rank_idx + 1) * num_experts_per_rank
        local_mask = [(g[:num_tokens] >= lo) & (g[:num_tokens] < hi) for g in gathered_topk]
        num_recv_tokens = sum(m.sum().item() for m in local_mask)
        # Unique local experts hit on this rank (dominant weight-stream term at decode sizes)
        touched = torch.cat([g[:num_tokens][m] for g, m in zip(gathered_topk, local_mask)]) \
            if num_recv_tokens > 0 else torch.empty(0, dtype=topk_idx.dtype, device='cuda')
        num_touched_experts = torch.unique(touched).numel()
        tflops = 2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12 / (t_max_us / 1e6)
        if torch.cuda.get_device_capability()[0] == 12:
            results[num_tokens] = (t_max_us, tflops)
            dist_print(f' > {mma_type} | {num_tokens:5} tokens/rank | {t_max_us:8.1f} us | {tflops:6.1f} TFLOPS',
                       once_in_node=True)
            del graph
            continue

        # Roofline theoretical times (per-rank), NVFP4: act=1 B/elem, weight=0.5 B/elem
        flops = 2 * num_recv_tokens * (hidden * intermediate_hidden * 3)
        weight_bytes = num_touched_experts * (3 * hidden * intermediate_hidden) * 0.5
        act_bytes = num_recv_tokens * (hidden * 1 + intermediate_hidden * 1 +
                                       intermediate_hidden * 1 + hidden * 2)
        hbm_bytes = weight_bytes + act_bytes
        nvlink_bytes = num_recv_tokens * hidden * 3
        t_compute = flops / (args.peak_fp4_tflops * 1e12)
        t_hbm = hbm_bytes / (args.hbm_bw_gbs * 1e9)
        t_hbm_weight = weight_bytes / (args.hbm_bw_gbs * 1e9)
        t_nvlink = nvlink_bytes / (args.nvlink_bw_gbs * 1e9)
        # Wall time is the slowest rank, so bind roofline to the busiest rank too (MAX)
        roof = torch.tensor([t_compute, t_hbm, t_hbm_weight, t_nvlink],
                            dtype=torch.float64, device='cuda')
        dist.all_reduce(roof, op=dist.ReduceOp.MAX, group=group)
        t_compute, t_hbm, t_hbm_weight, t_nvlink = (x * 1e6 for x in roof.tolist())  # -> us

        t_actual = t_max_us
        t_ideal = max(t_compute, t_hbm, t_nvlink)       # perfect-overlap lower bound
        t_serial = t_compute + t_hbm + t_nvlink         # zero-overlap upper bound
        overlap_pct = (t_serial - t_actual) / (t_serial - t_ideal) * 100 if t_serial > t_ideal else float('nan')
        achieved_vs_ideal = t_actual / t_ideal if t_ideal > 0 else float('nan')
        weight_stream_pct = t_hbm_weight / t_actual * 100 if t_actual > 0 else float('nan')
        results[num_tokens] = (t_actual, tflops, t_ideal, overlap_pct, weight_stream_pct)
        dist_print(f' > {mma_type} | {num_tokens:5} tokens/rank | {t_actual:8.1f} us | {tflops:6.0f} TFLOPS | '
                   f'ideal {t_ideal:7.1f} us (cmp {t_compute:6.1f} / hbm {t_hbm:6.1f} / nvl {t_nvlink:6.1f}) | '
                   f'overlap {overlap_pct:5.1f}% | {achieved_vs_ideal:4.2f}x ideal | wgt-bound {weight_stream_pct:5.1f}%',
                   once_in_node=True)

        del graph
    dist.barrier()
    buffer.destroy()
    torch.cuda.empty_cache()
    return results


# noinspection PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    token_counts = [int(x) for x in args.token_counts.split(',')] if args.token_counts \
        else [2 ** i for i in range(0, 14)]  # 1 .. 8192

    dist_print('Config:', once_in_node=True)
    dist_print(f' > Shapes: hidden={args.hidden}, intermediate={args.intermediate_hidden}, '
               f'experts={args.num_topk}/{args.num_experts}, EP={num_ranks}', once_in_node=True)
    dist_print(f' > Timing: CUDA graph replays (CPU overhead excluded)', once_in_node=True)
    dist_print(f' > Weights and routing seed: {args.seed} + rank', once_in_node=True)
    dist_print(once_in_node=True)

    mma_types = args.mma_types.split(',')
    if args.num_tokens_for_block_m is not None:
        assert torch.cuda.get_device_capability()[0] == 12 and mma_types == ['fp8xfp4']
        assert args.num_tokens_for_block_m >= 0
    if args.compare_unfused:
        assert num_ranks == 1 and mma_types == ['fp8xfp4']
        assert torch.cuda.get_device_capability()[0] == 12 and all(count > 0 for count in token_counts)
    all_results = {}
    for mma_type in mma_types:
        all_results[mma_type] = bench_mma_type(mma_type, args, group, rank_idx, num_ranks, token_counts)
        dist_print(once_in_node=True)

    # Summary table
    dist_print('Summary (time = slowest rank, per-iteration):', once_in_node=True)
    is_sm120 = torch.cuda.get_device_capability()[0] == 12
    if is_sm120:
        dist_print(' > SM120: measured timing/TFLOPS only; B200 roofline assumptions disabled', once_in_node=True)
    else:
        dist_print(f' > Roofline peaks: FP4 {args.peak_fp4_tflops:.0f} TFLOPS, '
                   f'HBM {args.hbm_bw_gbs:.0f} GB/s, NVLink {args.nvlink_bw_gbs:.0f} GB/s', once_in_node=True)
    cols = ['us', 'TFLOPS'] if is_sm120 else ['us', 'TFLOPS', 'ideal us', 'overlap%', 'wgt-bound%']
    header = f'{"tokens/rank":>12}' + ''.join(
        f' | {t + " " + c:>14}' for t in mma_types for c in cols)
    dist_print(header, once_in_node=True)
    for n in token_counts:
        row = f'{n:>12}'
        for t in mma_types:
            if is_sm120:
                elapsed_us, measured_tflops = all_results[t][n]
                row += f' | {elapsed_us:>14.1f} | {measured_tflops:>14.1f}'
            else:
                us, fl, ideal, ov, wgt = all_results[t][n]
                row += f' | {us:>14.1f} | {fl:>14.0f} | {ideal:>14.1f} | {ov:>14.1f} | {wgt:>14.1f}'
        dist_print(row, once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CUDA-graph benchmark: NVFP4 vs MXFP8xMXFP4 Mega MoE')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=6144)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0, help='Weights and routing seed (each rank adds its index)')
    parser.add_argument('--mma-types', type=str, default='fp8xfp4,fp4xfp4')
    parser.add_argument('--token-counts', type=str, default='',
                        help='Comma-separated tokens/rank list (default: powers of 2 up to 8192)')
    # Roofline peak constants (theoretical-time denominators); defaults follow B200/GB200
    parser.add_argument('--peak-fp4-tflops', type=float, default=9000.0,
                        help='FP4 tensor-core peak (TFLOPS); B200/GB200 ~9000 (dense)')
    parser.add_argument('--hbm-bw-gbs', type=float, default=8000.0,
                        help='HBM bandwidth (GB/s); B200/GB200 HBM3e ~8000')
    parser.add_argument('--nvlink-bw-gbs', type=float, default=900.0,
                        help='NVLink unidirectional bandwidth (GB/s); NVLink5 ~900 (1.8 TB/s bidir)')
    parser.add_argument('--num-shared-experts', type=int, default=0,
                        help='Fused BF16 shared experts (fp4xfp4 only; 0 disables)')
    parser.add_argument('--activation-clamp', type=float, default=None)
    parser.add_argument('--compare-unfused', action='store_true',
                        help='SM120 single-rank grouped-GEMM comparison with precomputed routing')
    parser.add_argument('--unfused-impl', choices=('triton', 'torch'), default='triton',
                        help='Dispatch, activation/quantization and combine implementation for --compare-unfused')
    parser.add_argument('--num-tokens-for-block-m', type=int, default=None,
                        help='SM120 FP8 x FP4 tile-selection override for ablation; timing still uses --token-counts')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
