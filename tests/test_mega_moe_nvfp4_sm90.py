import argparse
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils.dist import dist_print, init_dist

_E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _make_weights(e: int, n: int, k: int):
    # Checkpoint layout: codes `[E, N, K / 2]` (low nibble first), positive E4M3 scales `[E, N, K / 16]` in [2^-3, 2^3), FP32 `[E]`
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8)
    sf = torch.randint(0x20, 0x50, (e, n, k // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    return w, sf, torch.rand(e) * 2e-3 + 1e-3


def _dequant(w: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    codes = torch.stack((w & 0xf, w >> 4), dim=-1).flatten(-2).long()
    return _E2M1_VALUES.to(w.device)[codes] * sf.float().repeat_interleave(16, dim=-1) * gs[:, None, None]


def _activation(gate: torch.Tensor, up: torch.Tensor, activation: str, alpha: float, beta: float) -> torch.Tensor:
    if activation == 'situ':
        softcap = lambda x, cap: cap * torch.tanh(x / cap)
        return softcap(gate, alpha) * torch.sigmoid(gate) * softcap(up, beta)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def _reference(x, topk_idx, topk_weights, checkpoint, activation, activation_alpha, activation_beta):
    # FP32 weights of all ranks' experts, BF16 rounding where the kernel rounds: gate/up, weighted activation, expert output and the sum
    y = torch.zeros(x.shape, dtype=torch.float)
    for e in topk_idx.unique().tolist():
        if e < 0:
            continue
        l1, l2 = (_dequant(*(t[e:e + 1] for t in c))[0] for c in checkpoint)
        token_idx, slot_idx = (topk_idx == e).nonzero(as_tuple=True)
        gate_up = (x[token_idx].float() @ l1.T).bfloat16().float()
        act = _activation(gate_up[:, 0::2], gate_up[:, 1::2], activation, activation_alpha, activation_beta)
        act = act * topk_weights[token_idx, slot_idx][:, None]
        y.index_add_(0, token_idx, (act.bfloat16().float() @ l2.T).bfloat16().float())
    return y.bfloat16()


def _make_routing(kind: str, num_tokens: int, num_experts: int, num_topk: int) -> torch.Tensor:
    if kind == 'blocks':
        # The most pool blocks: 16a + 1 tokens per expert, dealt round-robin so that a token's experts differ
        num_extra_blocks = (num_tokens * num_topk - num_experts) // 16
        counts = 16 * (num_extra_blocks // num_experts + (torch.arange(num_experts) < num_extra_blocks % num_experts)) + 1
        counts[-1] += num_tokens * num_topk - counts.sum()
        slots = torch.repeat_interleave(torch.randperm(num_experts), counts[torch.randperm(num_experts)])
        return slots[torch.arange(num_tokens * num_topk).view(num_topk, num_tokens).T]
    if kind == 'local':
        # One rank's experts of a 4-rank split, other ranks' ids as -1: about a quarter of the top-k is local, some tokens have none
        topk_idx = torch.rand((num_tokens, 4 * num_experts)).topk(num_topk, dim=-1).indices
        return torch.where(topk_idx < num_experts, topk_idx, -1)
    # Random, all tokens on the same 8 experts, or random with padding (token 0 has none, a quarter of the others are -1)
    pool = torch.randperm(num_experts)[:num_topk if kind == 'same' else num_experts]
    topk_idx = pool[torch.rand((num_tokens, pool.numel())).topk(num_topk, dim=-1).indices]
    if kind == 'padded':
        topk_idx[torch.rand(topk_idx.shape) < 0.25] = -1
        topk_idx[0] = -1
    return topk_idx


# noinspection PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(0)

    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk, num_max_tokens = args.num_experts, args.num_topk, args.num_max_tokens_per_rank
    token_counts = sorted(int(t) for t in args.num_tokens.split(','))
    max_tokens = token_counts[-1]
    assert max_tokens <= num_max_tokens

    # `num_tokens_for_block_m` overrides covering every BLOCK_M
    block_m_overrides = {}
    for t in (1 << i for i in range(num_max_tokens.bit_length())):
        block_m_overrides.setdefault(deep_gemm.get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens, t, num_topk, 'bf16xnvfp4'), t)
    block_ms = sorted(block_m_overrides)
    assert block_ms[:2] == [16, 32] and (num_max_tokens < 8192 or block_ms == [16, 32, 64, 128]), block_m_overrides

    # Every rank draws all experts' weights alike and keeps its own; tokens and routing differ per rank
    checkpoint = (_make_weights(num_experts, intermediate_hidden * 2, hidden), _make_weights(num_experts, hidden, intermediate_hidden))
    num_experts_per_rank = num_experts // num_ranks
    local_checkpoint = (tuple(t[rank_idx * num_experts_per_rank:(rank_idx + 1) * num_experts_per_rank] for t in c) for c in checkpoint)
    (l1_w, l1_sf, l1_gs), (l2_w, l2_sf, l2_gs) = deep_gemm.transform_nvfp4_weights_for_sm90_mega_moe(*local_checkpoint)
    torch.manual_seed(rank_idx + 1)
    buffer = deep_gemm.SymmBuffer(group, num_experts, num_max_tokens, num_topk, hidden, intermediate_hidden, mma_type='bf16xnvfp4')

    # At one rank and <= 128 experts, BLOCK_M 16 runs 3 math warpgroups while a launch cannot reuse the ring (at most one partial
    # pool block per expert)
    num_max_pool_blocks = lambda t: -(-t * num_topk // 16) + min(num_experts, t * num_topk)
    last_no_reuse = max(t for t in range(1, num_max_tokens + 1) if num_max_pool_blocks(t) <= buffer.l1_acts.shape[0] // 16)

    dist_print('Config:')
    dist_print(f' > Ranks: {num_ranks}, hidden: {hidden}, intermediate: {intermediate_hidden}, experts: {num_topk}/{num_experts}, '
               f'tokens per rank: {token_counts}')
    dist_print()

    x = (torch.randn((max_tokens, hidden)) * 0.5).bfloat16()
    topk_weights = torch.rand((max_tokens, num_topk)) + 0.5

    def run(num_tokens, topk_idx, kwargs, y=None, x=x, **extra):
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16) if y is None else y
        buffer.x[:num_tokens].copy_(x[:num_tokens])
        buffer.topk_idx[:num_tokens].copy_(topk_idx[:num_tokens])
        buffer.topk_weights[:num_tokens].copy_(topk_weights[:num_tokens])
        deep_gemm.bf16_nvfp4_mega_moe(y, (l1_w, l1_sf), (l2_w, l2_sf), buffer, l1_gs, l2_gs, **kwargs, **extra)
        return y

    def check_reference(y, num_tokens, topk_idx, kwargs, name):
        ref = _reference(x[:num_tokens], topk_idx[:num_tokens], topk_weights[:num_tokens], checkpoint, **kwargs)
        diff = (y.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-30)
        assert y.isfinite().all() and diff < 0.01, f'{name}: rel-L2 {diff:.5f}'
        return diff

    situ = dict(activation='situ', activation_alpha=4.0, activation_beta=25.0)
    routings = {kind: _make_routing(kind, max_tokens, num_experts, num_topk) for kind in ('random', 'same', 'padded', 'blocks', 'local')}
    for routing, topk_idx in routings.items():
        for kwargs in (situ, dict(activation='swiglu', activation_alpha=1.0, activation_beta=0.0)):
            name, outputs = f'{routing}, {kwargs["activation"]}', {}
            for num_tokens in token_counts:
                y = run(num_tokens, topk_idx, kwargs)
                diff = check_reference(y, num_tokens, topk_idx, kwargs, f'{name}, {num_tokens} tokens')
                for block_m, t in block_m_overrides.items():
                    assert torch.equal(run(num_tokens, topk_idx, kwargs, num_tokens_for_block_m=t), y), \
                        f'{name}, {num_tokens} tokens: BLOCK_M {block_m} differs'
                outputs[num_tokens] = y
                dist_print(f' > {name}, {num_tokens} tokens: rel-L2 {diff:.5f}, bitwise equal over BLOCK_M {block_ms}')
            for num_tokens in token_counts[:-1]:
                assert torch.equal(outputs[num_tokens], outputs[max_tokens][:num_tokens]), \
                    f'{name}: {num_tokens} tokens differ from the prefix of {max_tokens}'
            mid_tokens = token_counts[len(token_counts) // 2]
            for num_tokens in (mid_tokens, max_tokens, mid_tokens, token_counts[0], max_tokens, token_counts[0]):
                assert torch.equal(run(num_tokens, topk_idx, kwargs), outputs[num_tokens]), f'{name}: {num_tokens} tokens differ after a switch'
            dist_print(f' > {name}: every batch equals the prefix of the largest, also after switching instances')

    # The last forced BLOCK_M 16 launch with 3 math warpgroups and the first without, with the most pool blocks
    for num_tokens in (t for t in (last_no_reuse, last_no_reuse + 1) if t <= max_tokens and num_ranks == 1 and num_experts <= 128):
        topk_idx = _make_routing('blocks', num_tokens, num_experts, num_topk)
        y = run(num_tokens, topk_idx, situ)
        check_reference(y, num_tokens, topk_idx, situ, f'blocks, {num_tokens} tokens')
        assert torch.equal(run(num_tokens, topk_idx, situ, num_tokens_for_block_m=1), y), f'{num_tokens} tokens: BLOCK_M 16 differs'
        dist_print(f' > blocks, {num_tokens} tokens: forced BLOCK_M 16 bitwise equal')

    # Repeated BLOCK_M 16 launches (3 math warpgroups at one rank) against one BLOCK_M 32 result
    for num_tokens in (t for t in (128, 160) if t <= max_tokens):
        expected = run(num_tokens, routings['random'], situ, num_tokens_for_block_m=block_m_overrides[32]).clone()
        num_mismatches = sum(int(not torch.equal(run(num_tokens, routings['random'], situ), expected)) for _ in range(args.num_repeats))
        assert num_mismatches == 0, f'{num_tokens} tokens: {num_mismatches}/{args.num_repeats} launches differ'
        dist_print(f' > {num_tokens} tokens: {args.num_repeats} launches bitwise equal to BLOCK_M 32')

    variants = [(routings[kind], x.roll(i, 0)) for i, kind in enumerate(('random', 'same', 'padded', 'blocks'))]
    for num_tokens in (int(t) for t in args.graph_tokens.split(',')):
        expected = [run(num_tokens, topk_idx, situ, x=xv).clone() for topk_idx, xv in variants]
        staged_idx, staged_x, y = variants[0][0].clone(), variants[0][1].clone(), torch.empty_like(expected[0])
        run(num_tokens, staged_idx, situ, y=y, x=staged_x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(num_tokens, staged_idx, situ, y=y, x=staged_x)
        num_mismatches = 0
        for i in range(args.num_graph_replays):
            staged_idx.copy_(variants[i % len(variants)][0])
            staged_x.copy_(variants[i % len(variants)][1])
            y.zero_()
            graph.replay()
            num_mismatches += int(not torch.equal(y, expected[i % len(variants)]))
        assert num_mismatches == 0, f'{num_tokens} tokens: {num_mismatches}/{args.num_graph_replays} graph replays differ'
        dist_print(f' > {num_tokens} tokens: {args.num_graph_replays} graph replays with changing inputs bitwise equal')

    dist_print(' > All correctness tests passed')
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the SM90 NVFP4 Mega MoE (BF16 tokens)')
    parser.add_argument('--num-processes', type=int, default=1, help='Number of ranks (one process and GPU each)')
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=str, default='1,8,128,160,1000,8192', help='Comma-separated token counts')
    parser.add_argument('--graph-tokens', type=str, default='1,128,8192', help='Comma-separated token counts for graph replays')
    parser.add_argument('--hidden', type=int, default=2304, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=1152, help='Intermediate hidden size')
    parser.add_argument('--num-experts', type=int, default=128, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of expert selections')
    parser.add_argument('--num-graph-replays', type=int, default=100, help='CUDA graph replays per token count')
    parser.add_argument('--num-repeats', type=int, default=300, help='Repeated launches compared bitwise')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for sanitizers)')
    args = parser.parse_args()
    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
