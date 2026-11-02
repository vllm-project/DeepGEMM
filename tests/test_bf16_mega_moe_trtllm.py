import argparse
import random
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils.dist import dist_print, init_dist


# TensorRT-LLM row shuffle for an epilogue tile of 128 rows (32-row blocks)
_SHUFFLE_BLOCK_32_ROW_MAP = [0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
                             4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31]


def _shuffle_row_indices(num_rows: int) -> torch.Tensor:
    # `indices[new_row] = old_row`
    old_rows = torch.arange(num_rows)
    new_rows = old_rows // 32 * 32 + torch.tensor(_SHUFFLE_BLOCK_32_ROW_MAP)[old_rows % 32]
    indices = torch.empty(num_rows, dtype=torch.long)
    indices[new_rows] = old_rows
    return indices


def _to_trtllm_block_layout(weights: torch.Tensor, is_l1: bool) -> torch.Tensor:
    # Reproduce the TensorRT-LLM BF16 MoE weight preparation:
    # `[E, N, K]` with rows permuted, viewed as BlockMajorK `[E, K / 64, N, 64]`
    num_experts, n, k = weights.shape
    indices = _shuffle_row_indices(n)
    if is_l1:
        # Interleave the two halves for the gated activation, then put the
        # second half (up) first: `[gate; up]` -> `[up; gate]`
        gated = torch.empty(n, dtype=torch.long)
        gated[0::2] = torch.arange(n // 2)
        gated[1::2] = torch.arange(n // 2, n)
        indices = (gated[indices] + n // 2) % n
    permuted = weights[:, indices.to(weights.device)]
    return permuted.view(num_experts, n, k // 64, 64).permute(0, 2, 1, 3).contiguous()


# noinspection PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type='bf16xbf16')

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: bf16xbf16, native vs TRT-LLM weight layout', once_in_node=True)
    dist_print(f' > Hidden: {hidden}, intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}, max tokens per rank: {num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Post-expert norm: {bool(args.post_expert_norm)}', once_in_node=True)
    dist_print(once_in_node=True)

    l1_weights = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden),
                             dtype=torch.bfloat16, device='cuda') / hidden ** 0.5
    l2_weights = torch.randn((num_experts_per_rank, hidden, intermediate_hidden),
                             dtype=torch.bfloat16, device='cuda') / intermediate_hidden ** 0.5
    native_l1, native_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights.clone(), l2_weights.clone())
    trtllm_l1 = _to_trtllm_block_layout(l1_weights, is_l1=True)
    trtllm_l2 = _to_trtllm_block_layout(l2_weights, is_l1=False)
    norm_kwargs = dict(
        post_expert_norm_gamma=1.0 + 0.1 * torch.randn((hidden, ), dtype=torch.float, device='cuda'),
        post_expert_norm_eps=1e-8) if args.post_expert_norm else {}

    # Each case uses new tokens and routing, so the previous launch's ring
    # contents cannot hide rows that the kernel failed to write
    token_counts = [int(t) for t in args.num_tokens.split(',')] if args.num_tokens else \
        [random.randint(1, num_max_tokens_per_rank) for _ in range(args.num_random_cases)]
    for num_tokens in token_counts:
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_weights = topk_weights.softmax(dim=-1)
        if args.masked_ratio > 0:
            topk_idx.masked_fill_(torch.rand_like(topk_weights) < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        outputs = {}
        for layout, (l1, l2) in (('native', (native_l1, native_l2)), ('trtllm', (trtllm_l1, trtllm_l2))):
            buffer.x[:num_tokens].copy_(x)
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.bf16_mega_moe(y, l1, l2, buffer, weight_layout=layout, **norm_kwargs)
            outputs[layout] = y

        equal = torch.tensor([int(torch.equal(outputs['native'], outputs['trtllm']))], device='cuda')
        dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=group)
        assert equal.item() == 1, f'Rank {rank_idx}: TRT-LLM layout output differs at {num_tokens} tokens'
        dist_print(f' > {num_tokens} tokens: bitwise equal on all ranks')

    dist_print(' > All correctness tests passed', once_in_node=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BF16 Mega MoE with TensorRT-LLM weight layouts')
    parser.add_argument('--num-processes', type=int, default=4, help='Number of processes to spawn (default: 4)')
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=str, default='',
                        help='Comma-separated tokens per rank to test (random cases if empty)')
    parser.add_argument('--num-random-cases', type=int, default=8, help='Number of random token counts')
    parser.add_argument('--hidden', type=int, default=4096, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=4096, help='Intermediate hidden size')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--post-expert-norm', type=int, default=1, help='Fuse the post-expert RMSNorm (0 or 1)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
