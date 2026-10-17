import argparse
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_nvfp4
from deep_gemm.utils.dist import dist_print, init_dist


def _make_weights(mma_type: str, num_experts_per_rank: int, hidden: int, intermediate_hidden: int):
    l1_bf16 = 0.1 * torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_bf16 = 0.1 * torch.randn((num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')

    def cast_mx(bf16_weights):
        g, n, k = bf16_weights.shape
        w = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((g, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(g):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), g)
        return w, w_sf

    def cast_nv(bf16_weights):
        g, n, k = bf16_weights.shape
        w = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((g, n, k // 64), device='cuda', dtype=torch.int32)
        for i in range(g):
            w[i], w_sf[i] = per_token_cast_to_nvfp4(bf16_weights[i], gran_k=16, use_packed_e4m3=True)
        return w, w_sf.transpose(-1, -2).contiguous().transpose(-1, -2)

    cast = cast_mx if mma_type == 'fp8xfp4' else cast_nv
    l1_weights, l2_weights = cast(l1_bf16), cast(l2_bf16)
    del l1_bf16, l2_bf16
    torch.cuda.empty_cache()
    return deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)


# noinspection PyShadowingNames
def bench_mma_type(mma_type: str, args, group, rank_idx: int, num_ranks: int, token_counts):
    num_experts, num_topk = args.num_experts, args.num_topk
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = max(token_counts)

    # BF16 shared experts are only supported by the NVFP4 kernel
    num_shared_experts = args.num_shared_experts if mma_type == 'fp4xfp4' else 0
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, num_shared_experts=num_shared_experts, mma_type=mma_type)
    l1_weights, l2_weights = _make_weights(mma_type, num_experts_per_rank, hidden, intermediate_hidden)

    # BF16 shared expert weights (folded on N for L1, K for L2)
    shared_l1_weights, shared_l2_weights = None, None
    if num_shared_experts > 0:
        sih = intermediate_hidden * num_shared_experts
        shared_l1_weights, shared_l2_weights = deep_gemm.transform_weights_for_mega_moe(
            0.1 * torch.randn((sih * 2, hidden), dtype=torch.bfloat16, device='cuda'),
            0.1 * torch.randn((hidden, sih), dtype=torch.bfloat16, device='cuda'))

    # Inputs at max size (per-rank seed for routing diversity)
    torch.manual_seed(rank_idx)
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

        def launch():
            if num_shared_experts > 0:
                kernel_fn(y=y, l1_weights=l1_weights, l2_weights=l2_weights,
                          sym_buffer=buffer, shared_l1_weights=shared_l1_weights,
                          shared_l2_weights=shared_l2_weights, x_bf16=x_bf16, fast_math=True)
            else:
                kernel_fn(y=y, l1_weights=l1_weights, l2_weights=l2_weights,
                          sym_buffer=buffer, fast_math=True)

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

        lo, hi = rank_idx * num_experts_per_rank, (rank_idx + 1) * num_experts_per_rank
        local_mask = [(g[:num_tokens] >= lo) & (g[:num_tokens] < hi) for g in gathered_topk]
        num_recv_tokens = sum(m.sum().item() for m in local_mask)
        # Unique local experts hit on this rank (dominant weight-stream term at decode sizes)
        touched = torch.cat([g[:num_tokens][m] for g, m in zip(gathered_topk, local_mask)]) \
            if num_recv_tokens > 0 else torch.empty(0, dtype=topk_idx.dtype, device='cuda')
        num_touched_experts = torch.unique(touched).numel()
        tflops = 2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12 / (t_max_us / 1e6)

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
    dist_print(once_in_node=True)

    mma_types = args.mma_types.split(',')
    all_results = {}
    for mma_type in mma_types:
        all_results[mma_type] = bench_mma_type(mma_type, args, group, rank_idx, num_ranks, token_counts)
        dist_print(once_in_node=True)

    # Summary table
    dist_print('Summary (time = slowest rank, per-iteration):', once_in_node=True)
    dist_print(f' > Roofline peaks: FP4 {args.peak_fp4_tflops:.0f} TFLOPS, '
               f'HBM {args.hbm_bw_gbs:.0f} GB/s, NVLink {args.nvlink_bw_gbs:.0f} GB/s', once_in_node=True)
    cols = ['us', 'TFLOPS', 'ideal us', 'overlap%', 'wgt-bound%']
    header = f'{"tokens/rank":>12}' + ''.join(
        f' | {t + " " + c:>14}' for t in mma_types for c in cols)
    dist_print(header, once_in_node=True)
    for n in token_counts:
        row = f'{n:>12}'
        for t in mma_types:
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
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
