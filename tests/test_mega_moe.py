import argparse
import functools
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Optional, Tuple

import deep_gemm
from deep_gemm.utils import (
    align, ceil_div,
    per_token_cast_to_fp4, per_token_cast_to_fp8,
    cast_back_from_fp4, cast_back_from_fp8
)
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


def _to_shared_mega_moe_sf_layout(sf: torch.Tensor, block_m: int, num_max_sf_tokens: int) -> torch.Tensor:
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = align(block_m, 128)
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype, device=sf.device)
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, num_tokens - block_idx * block_m)
        for m_idx in range(num_block_tokens):
            transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            result[block_idx * aligned_block_m + transposed_m_idx].copy_(sf[block_idx * block_m + m_idx])
    return result


def _to_mn_major_sf(sf: torch.Tensor) -> torch.Tensor:
    # MN-major and TMA-aligned, group-contiguous for 3D
    *num_groups, mn, packed_sf_k = sf.shape
    sf_tma = torch.empty((*num_groups, packed_sf_k, align(mn, 4)), dtype=sf.dtype, device=sf.device)[..., :mn]
    sf_tma.copy_(sf.transpose(-1, -2))
    return sf_tma.transpose(-1, -2)


def _exp2(exps: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    # Exact `2 ** (e - 127)` from the exponent bits (FP32 needs e in [1, 254])
    if dtype == torch.float:
        return (exps.int() << 23).view(torch.float)
    return ((exps.long() + 1023 - 127) << 52).view(torch.double)


def _emu_weights(w: torch.Tensor, exps: torch.Tensor, gran_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # MMA operand and per-(row, `gran_k`) promotion scale of weights with per-32 UE8M0 exponents
    n, num_sf = exps.shape
    exps = exps.int().view(n, -1, gran_k // 32)
    if w.dtype == torch.int8:
        # FP4 fold: base `max(e) - 6` per row over nonzero groups (zero scale below 7), one RN cast to E4M3
        is_nonzero = ((w.view(torch.uint8) & 0x77).view(n, -1, gran_k // 32, 16).amax(dim=-1) != 0)
        exps = torch.where(is_nonzero, exps, 0)
        max_exps = exps.amax(dim=(1, 2), keepdim=True)
        exps = torch.where(is_nonzero, exps, max_exps).clamp(min=6)
        base = torch.where(max_exps < 7, 0, max_exps - 6).repeat(1, exps.size(1), 1)
        w = cast_back_from_fp4(w, _exp2((exps - base).view(n, num_sf) + 127), gran_k=32).to(torch.float8_e4m3fn)
    else:
        # FP8: the stored values, with the per-32 exponents of a block all equal
        base = exps[:, :, :1]
    return w.float(), _exp2(base).view(n, -1)


def _emu_gemm(a: torch.Tensor, sa: torch.Tensor, b: torch.Tensor, sb: torch.Tensor, gran_k: int) -> torch.Tensor:
    # Each `gran_k` block accumulates unscaled, then promotes as `out += accum * (sa * sb)` in block order
    out = torch.zeros((a.size(0), b.size(0)), dtype=torch.float, device=a.device)
    for i in range(sa.size(1)):
        k = slice(i * gran_k, (i + 1) * gran_k)
        out += (a[:, k] @ b[:, k].T) * (sa[:, i:i + 1] * sb[:, i])
    return out


def _apply_activation(l1_y: torch.Tensor, weights: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    # SwiGLU clamps the gate before SiLU, SwiGLU-step the SiLU output (with alpha 1 and beta 0)
    gate, up = l1_y.chunk(2, dim=-1)
    is_step = args.activation == 'swiglustep'
    if args.activation_clamp is not None:
        gate, up = gate if is_step else gate.clamp(max=args.activation_clamp), up.clamp(-args.activation_clamp, args.activation_clamp)
    gated = gate / (1 + torch.exp(-args.activation_alpha * gate))
    if is_step and args.activation_clamp is not None:
        gated = gated.clamp(max=args.activation_clamp)
    return gated * (up + args.activation_beta) * weights[:, None]


def _emu_expert(x: torch.Tensor, sx: torch.Tensor, l1_weights: Tuple, l2_weights: Tuple, weights: torch.Tensor,
                gran_k: int, args: argparse.Namespace) -> torch.Tensor:
    # The kernel's numerics, with BF16 gate/up and BF16 output rounding and per-`gran_k` L2 requantization (FP32 scales on SM90)
    l1_y = _emu_gemm(x.float(), sx, *_emu_weights(*l1_weights, gran_k), gran_k)
    l2_x, l2_sx = per_token_cast_to_fp8(
        _apply_activation(l1_y.bfloat16().float(), weights, args), use_ue8m0=gran_k != 128, gran_k=gran_k)
    return _emu_gemm(l2_x.float(), l2_sx, *_emu_weights(*l2_weights, gran_k), gran_k).bfloat16()


def _hi_weights(w: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    cast_back = cast_back_from_fp4 if w.dtype == torch.int8 else cast_back_from_fp8
    return cast_back(w, _exp2(exps, torch.double), gran_k=32)


def _hi_expert(x: torch.Tensor, l1_weights: Tuple, l2_weights: Tuple, weights: torch.Tensor,
               args: argparse.Namespace) -> torch.Tensor:
    # FP64 with exact weights and no rounding in between
    return _apply_activation(x @ _hi_weights(*l1_weights).T, weights, args) @ _hi_weights(*l2_weights).T


def _moe_reference(expert, xs: Tuple, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                   l1_weights: Tuple, l2_weights: Tuple, shared_weights: Optional[Tuple],
                   dtype: torch.dtype) -> torch.Tensor:
    # Expert outputs per (token, slot) over all global experts, summed shared expert first, then in top-k order
    num_tokens, num_topk = topk_idx.shape
    slot_y = torch.zeros((num_tokens, num_topk, xs[0].size(1)), dtype=dtype, device='cuda')
    for i in range(l1_weights[0].size(0)):
        tokens, slots = torch.where(topk_idx == i)
        if tokens.numel() > 0:
            slot_y[tokens, slots] = expert(
                *(x[tokens] for x in xs), tuple(w[i] for w in l1_weights), tuple(w[i] for w in l2_weights),
                topk_weights[tokens, slots].to(dtype)).to(dtype)
    y = torch.zeros((num_tokens, xs[0].size(1)), dtype=dtype, device='cuda')
    if shared_weights is not None:
        y += expert(*xs, *shared_weights, torch.ones(num_tokens, dtype=dtype, device='cuda')).to(dtype)
    for i in range(num_topk):
        y += slot_y[:, i]
    return y


def _calc_token_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # `calc_diff` per token, zero in both counts as equal
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum(dim=-1)
    return 1 - torch.where(denominator == 0, 1, 2 * (x * y).sum(dim=-1) / denominator)


# Sweep: tokens per rank, hidden, intermediate, experts per rank, top-k, shared experts, clamp, routing, extra args
# The heaviest 20 also run with more ranks
# NOTES: only sweep cases set these: bit-exact inputs (with E4M3 subnormal activations), per-128 activation scales 2^U(-n, n),
# the routed weight exponent span per row and 128 K block, exponents moved from the activations to the L1 weights, and the fraction
# of zeroed routed weight groups
_CASE_DEFAULTS = dict(exact_inputs=False, subnormal_probe=False, act_exp_range=0, weight_exp_span=None, weight_exp_shift=0,
                      zero_weight_groups=0)
_EXACT = dict(exact_inputs=True, fast_math=0)
_SWEEP_CASES = (
    (128, 2048, 256, 8, 6, 1, 10, 'random', dict(subnormal_probe=True, fast_math=0)),
    (1, 512, 128, 1, 1, 0, None, 'random', _EXACT),
    (7, 2048, 1536, 8, 6, 1, 10, 'masked', _EXACT),
    (63, 4096, 256, 2, 1, 2, None, 'hot', _EXACT),
    (129, 7168, 3072, 48, 8, 1, 10, 'empty', _EXACT),
    (1000, 512, 3072, 65, 6, 0, 10, 'random', _EXACT),
    (4096, 2048, 128, 8, 8, 2, None, 'masked', _EXACT),
    (8192, 4096, 1536, 1, 1, 1, 10, 'hot', _EXACT),
    (0, 7168, 256, 2, 2, 1, None, 'random', _EXACT),
    (128, 5120, 2304, 65, 8, 2, 10, 'hot', _EXACT),
    (1000, 5120, 2304, 48, 6, 1, 10, 'random', _EXACT),
    (0, 512, 1536, 8, 6, 0, 10, 'masked', {}),
    (1, 2048, 3072, 2, 1, 2, None, 'empty', {}),
    (1, 7168, 2304, 48, 6, 1, 10, 'random', {}),
    (1, 5120, 256, 65, 8, 0, None, 'masked', {}),
    (7, 512, 256, 48, 8, 2, None, 'random', {}),
    (7, 4096, 3072, 1, 1, 0, None, 'masked', {}),
    (7, 7168, 128, 65, 1, 1, 10, 'hot', {}),
    (63, 512, 2304, 8, 1, 1, None, 'empty', {}),
    (63, 2048, 128, 48, 6, 0, 10, 'masked', {}),
    (63, 5120, 1536, 65, 6, 2, 10, 'random', {}),
    (128, 512, 1536, 2, 2, 1, 10, 'masked', {}),
    (128, 4096, 128, 48, 1, 0, None, 'random', {}),
    (128, 7168, 3072, 8, 6, 0, 10, 'random', {}),
    (129, 2048, 2304, 1, 1, 1, None, 'random', {}),
    (129, 4096, 3072, 65, 6, 1, None, 'masked', {}),
    (129, 5120, 128, 8, 8, 0, 10, 'masked', {}),
    (1000, 2048, 256, 1, 1, 2, 10, 'random', {}),
    (1000, 4096, 2304, 2, 2, 0, None, 'masked', {}),
    (1000, 7168, 1536, 8, 6, 2, None, 'hot', {}),
    (4096, 512, 2304, 65, 1, 1, 10, 'masked', {}),
    (4096, 5120, 3072, 2, 1, 1, 10, 'empty', {}),
    (4096, 7168, 256, 48, 8, 0, None, 'masked', {}),
    (8192, 512, 256, 48, 6, 0, 10, 'empty', {}),
    (8192, 2048, 3072, 48, 6, 1, None, 'masked', {}),
    (8192, 7168, 3072, 48, 6, 1, 10, 'random', {}),
    (8192, 5120, 2304, 48, 6, 1, 10, 'random', {}),
    (8192, 7168, 128, 65, 8, 2, 10, 'masked', {}),
    (63, 4096, 1536, 8, 6, 1, None, 'random', dict(act_exp_range=20)),
    (1000, 7168, 2304, 65, 8, 0, None, 'masked', dict(act_exp_range=20)),
    (4096, 2048, 3072, 8, 6, 2, None, 'random', dict(act_exp_range=20)),
    (129, 2048, 1536, 8, 6, 1, 10, 'random', dict(weight_exp_span=0)),
    (1000, 5120, 2304, 8, 6, 1, 10, 'random', dict(shared_gran_k=32)),
    (4096, 7168, 1536, 48, 8, 2, None, 'masked', dict(shared_gran_k=32)),
    (1000, 4096, 256, 2, 2, 0, None, 'random', dict(weight_exp_span=11)),
    (128, 5120, 3072, 48, 6, 1, 10, 'masked', dict(weight_exp_span=3)),
    (4096, 512, 1536, 8, 8, 1, None, 'random', dict(weight_exp_span=7)),
    (7, 7168, 2304, 8, 6, 0, 10, 'random', dict(weight_exp_span=9)),
    (8192, 4096, 2304, 65, 6, 1, 10, 'random', dict(weight_exp_span=10)),
    (1000, 2048, 128, 48, 6, 2, None, 'hot', dict(weight_exp_span=11)),
    (1000, 4096, 256, 8, 6, 1, 10, 'random', dict(weight_exp_span=14)),
    (128, 4096, 1536, 8, 6, 1, None, 'random', dict(weight_exp_shift=120)),
    (1000, 2048, 2304, 48, 6, 1, None, 'masked', dict(weight_exp_shift=120, weight_exp_span=8)),
    (128, 7168, 256, 8, 6, 1, None, 'random', dict(weight_exp_shift=-105)),
    (4096, 5120, 1536, 48, 8, 0, None, 'random', dict(weight_exp_shift=-105, weight_exp_span=11)),
    (129, 2048, 1536, 8, 6, 1, 10, 'random', dict(zero_weight_groups=0.2)),
    (1000, 5120, 2304, 48, 6, 1, 10, 'masked', dict(zero_weight_groups=0.05, weight_exp_span=11)),
    (7, 7168, 256, 8, 6, 0, None, 'random', dict(zero_weight_groups=0.5)),
    (129, 2048, 1536, 8, 6, 0, 0.5, 'random', dict(activation='swiglustep')),
    (1000, 5120, 2304, 48, 6, 0, 2, 'masked', dict(activation='swiglustep')),
    (7, 7168, 3072, 8, 8, 0, 1, 'hot', dict(activation='swiglustep')),
    (63, 4096, 256, 2, 1, 0, 0.5, 'random', dict(activation='swiglustep', **_EXACT)),
    (4096, 2048, 1536, 48, 6, 0, None, 'random', dict(activation='swiglustep')),
)


def _get_sweep_args(args: argparse.Namespace, num_ranks: int):
    cases = _SWEEP_CASES if num_ranks == 1 else \
        sorted(_SWEEP_CASES, key=lambda case: case[0] * case[1] * case[2] * (case[4] + case[5]), reverse=True)[:20]
    for num_tokens, hidden, intermediate_hidden, num_experts_per_rank, num_topk, num_shared_experts, \
            activation_clamp, routing, extra in cases:
        if get_arch_major() != 9 and (extra.get('subnormal_probe') or (extra.get('exact_inputs') and not args.emu_reference)):
            continue
        yield argparse.Namespace(**{
            **_CASE_DEFAULTS, **vars(args),
            'num_tokens': num_tokens, 'num_max_tokens_per_rank': max(num_tokens, 1),
            'hidden': hidden, 'intermediate_hidden': intermediate_hidden,
            'num_experts': num_experts_per_rank * num_ranks, 'num_topk': num_topk,
            'num_shared_experts': num_shared_experts, 'activation_clamp': activation_clamp,
            'routing': 'random' if routing == 'masked' else routing, 'masked_ratio': 0.3 if routing == 'masked' else 0,
            **extra})


def _copy_fp8_sf(dst: torch.Tensor, src: torch.Tensor, num_tokens: int) -> None:
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_case(rank_idx: int, num_ranks: int, group: dist.ProcessGroup, args: argparse.Namespace) -> Tuple[int, bool]:
    global transformed_l1_weights, transformed_l2_weights
    assert args.activation != 'situ' or args.mma_type != 'bf16xbf16', \
        'SiTU is only supported by the FP8/FP4 Mega MoE kernel path'
    deep_gemm.use_deterministic_algorithms(True)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    is_bf16xbf16 = args.mma_type == 'bf16xbf16'
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens is None else args.num_tokens
    num_tokens_per_rank = [int(t) for t in args.num_tokens_per_rank.split(',')] if args.num_tokens_per_rank else None
    num_tokens = num_tokens_per_rank[rank_idx] if num_tokens_per_rank else num_tokens
    if args.empty_rank == rank_idx:
        num_tokens = 0
    num_shared_experts = args.num_shared_experts
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts
    assert num_tokens <= num_max_tokens_per_rank

    norm_gamma = 0.5 + torch.arange(hidden, device="cuda", dtype=torch.float32) / hidden

    # SM90 quantizes activations per 128 channels and checks against the PyTorch reference
    arch_major = get_arch_major()
    gran_k = 128 if arch_major == 9 else 32
    shared_gran_k = args.shared_gran_k or gran_k
    use_emu_reference = arch_major == 9 or args.emu_reference
    is_exact_inputs = args.exact_inputs or args.subnormal_probe
    assert not use_emu_reference or (not is_bf16xbf16 and args.activation in ('swiglu', 'swiglustep'))
    assert not is_exact_inputs or (use_emu_reference and not args.fast_math)
    assert not args.subnormal_probe or arch_major == 9

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared_experts,
        mma_type=args.mma_type,
        activation=args.activation
    )
    # SM90 ranks choose one BLOCK_M from the mean tokens per rank
    num_tokens_for_block_m = ceil_div(sum(num_tokens_per_rank), num_ranks) \
        if num_tokens_per_rank and arch_major == 9 else num_tokens
    block_m = deep_gemm.get_block_m_for_mega_moe(
        num_ranks, num_experts, buffer.num_max_tokens_per_rank, num_tokens_for_block_m, num_topk, args.mma_type)

    # Weights as (FP4 or FP8 values, per-32 UE8M0 exponents), identical on every rank
    weight_generator = torch.Generator(device='cuda')
    weight_generator.manual_seed(0)

    def make_weights(num_groups: int, n: int, k: int,
                     is_fp8: bool, is_l1: bool, is_routed: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_kwargs = dict(generator=weight_generator, device='cuda')
        w = torch.empty((num_groups, n, k if is_fp8 else k // 2),
                        dtype=torch.float8_e4m3fn if is_fp8 else torch.int8, device='cuda')
        exps = torch.empty((num_groups, n, k // 32), dtype=torch.uint8, device='cuda')
        for i in range(num_groups):
            if is_exact_inputs:
                # Values in {0, +-0.5, +-1} with one nonzero per L2 row, and one power-of-two scale per 128 K
                codes = torch.randint(5, (n, k), **rand_kwargs)
                if not is_l1:
                    codes = torch.zeros_like(codes).scatter_(
                        1, torch.randint(k, (n, 1), **rand_kwargs), torch.randint(1, 5, (n, 1), **rand_kwargs))
                block_exps = torch.randint(125, 130, (n, k // 128, 1), **rand_kwargs).repeat(1, 1, 4)
                if is_fp8:
                    w[i] = torch.tensor((0, 0.5, 1, -0.5, -1), device='cuda')[codes].to(torch.float8_e4m3fn)
                else:
                    codes = torch.tensor((0, 1, 2, 9, 10), dtype=torch.uint8, device='cuda')[codes]
                    w[i] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
                exps[i] = block_exps.view(n, -1)
                continue

            # Routed weights are per-32 casts, shared ones per `--shared-gran-k` with the exponent in all its bytes
            cast = per_token_cast_to_fp8 if is_fp8 else per_token_cast_to_fp4
            gran_k_w = 32 if is_routed else shared_gran_k
            w[i], sf = cast(torch.randn((n, k), dtype=torch.bfloat16, **rand_kwargs) * k ** -0.5,
                            use_ue8m0=True, gran_k=gran_k_w)
            e = (sf.view(torch.int) >> 23).repeat_interleave(gran_k_w // 32, dim=1)
            if is_l1:
                e = (e + args.weight_exp_shift).clamp(1, 254)
            if is_routed and args.weight_exp_span is not None:
                # Per row, exponents `max - offset` with offsets spanning exactly `weight_exp_span` in every (row, 128 K) block
                offsets = torch.randint(args.weight_exp_span + 1, (n, k // 128, 4), **rand_kwargs)
                offsets[:, :, 0], offsets[:, :, 1] = 0, args.weight_exp_span
                e = e.amax(dim=1, keepdim=True) - offsets.view(n, -1)
            if is_routed and not is_fp8 and args.zero_weight_groups > 0:
                # Zero groups (half -0) with exponent 0, plus zero rows and 128 K blocks like padding
                is_zero = torch.rand((n, k // 32), **rand_kwargs) < args.zero_weight_groups
                is_zero[::16], is_zero[3::7, :4] = True, True
                zero_byte = torch.where(torch.rand((n, k // 32, 1), **rand_kwargs) < 0.5, 0x00, 0x88).to(torch.uint8)
                w[i] = torch.where(is_zero[:, :, None], zero_byte, w[i].view(torch.uint8).view(n, -1, 16)).view(n, -1).view(torch.int8)
                e = torch.where(is_zero, 0, e)
            exps[i] = e
        return w, exps

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, shared_x, shared_l1_x_sf, topk_idx, topk_weights, l1_weights, l2_weights
        global transformed_l1_weights, transformed_l2_weights
        global shared_l1_weights, shared_l2_weights, transformed_shared_l1_weights, transformed_shared_l2_weights
        global cumulative_local_expert_recv_stats_fused, cumulative_local_expert_recv_stats_baseline
        global initial_cumulative_local_expert_recv_stats_fused, initial_cumulative_local_expert_recv_stats_baseline
        global ref_x, ref_x_hi, all_l1_weights, all_l2_weights, ref_shared_weights, hi_shared_weights
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        if is_bf16xbf16:
            l1_weights = torch.randn(
                (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
            l2_weights = torch.randn(
                (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        routing_bias = torch.zeros(num_experts, dtype=torch.float, device='cuda')
        if args.routing == 'hot':
            routing_bias[0] = float('inf')
        elif args.routing == 'empty':
            routing_bias[1::2] = -float('inf')
        topk_idx = torch.topk(scores + routing_bias, num_topk, dim=-1, largest=True, sorted=False)[1]
        topk_weights = scores.gather(1, topk_idx)
        if is_exact_inputs:
            topk_weights = _exp2(torch.randint(126, 129, topk_weights.shape, device='cuda'))
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_fused = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_baseline.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        if num_shared_experts > 0 and is_bf16xbf16:
            shared_l1_weights = torch.randn(
                (shared_intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
            shared_l2_weights = torch.randn(
                (hidden, shared_intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        else:
            shared_l1_weights = shared_l2_weights = ref_shared_weights = hi_shared_weights = None

        if not is_bf16xbf16:
            # FP8 path: activations with one UE8M0 exponent per (token, `gran_k`)
            assert hidden % 128 == 0 and intermediate_hidden % 128 == 0 and shared_intermediate_hidden % 128 == 0
            if is_exact_inputs:
                # Integers in [-2, 2], or whole blocks of subnormal bytes in the probe
                x_fp8 = torch.randint(-2, 3, (num_tokens, hidden), device='cuda').float().to(torch.float8_e4m3fn)
                x_exps = torch.randint(125, 130, (num_tokens, hidden // gran_k), dtype=torch.int, device='cuda')
                if args.subnormal_probe:
                    is_probe = torch.rand((num_tokens, hidden // gran_k), device='cuda') < 0.5
                    subnormals = torch.randint(8, (num_tokens, hidden), dtype=torch.uint8, device='cuda') | \
                        (torch.randint(2, (num_tokens, hidden), dtype=torch.uint8, device='cuda') << 7)
                    x_fp8 = torch.where(is_probe.repeat_interleave(gran_k, dim=1),
                                        subnormals, x_fp8.view(torch.uint8)).view(torch.float8_e4m3fn)
                ref_x_hi = x_fp8.double() * _exp2(x_exps, torch.double).repeat_interleave(gran_k, dim=1)
                x_scales = _exp2(x_exps)
            else:
                if args.act_exp_range > 0:
                    act_exps = torch.randint(
                        127 - args.act_exp_range, 128 + args.act_exp_range, (num_tokens, hidden // 128), device='cuda')
                    x = (x.float() * _exp2(act_exps).repeat_interleave(128, dim=1)).bfloat16()
                # SM90 uses FP32 scales, except with `--weight-exp-shift` (power-of-two scales)
                use_ue8m0 = arch_major != 9 or args.weight_exp_shift != 0
                x_fp8, x_scales = per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
                x_exps = x_scales.view(torch.int) >> 23
                # `--weight-exp-shift` moves exponents from the activations to the L1 weights
                shifted_x_exps = (x_exps - args.weight_exp_shift).clamp(1, 254)
                ref_x_hi = x.double() * _exp2(shifted_x_exps - x_exps + 127, torch.double).repeat_interleave(gran_k, dim=1)
                x_exps = shifted_x_exps
                if use_ue8m0:
                    x_scales = _exp2(x_exps)
            # SM90 SF words hold the FP32 scales, SM100 ones the UE8M0 exponents of 4 groups
            x_sf = x_scales.contiguous() if arch_major == 9 else x_exps.to(torch.uint8).view(torch.int)
            ref_x = (x_fp8, x_scales)
            x = (x_fp8, x_sf)
            shared_x = (x_fp8, _to_mn_major_sf(x_sf))
            if num_shared_experts > 0:
                shared_l1_x_sf = _to_shared_mega_moe_sf_layout(x_sf, block_m, buffer.shared_l1_acts_sf.shape[0])

            # Routed weights of all experts for the PyTorch reference, the local slice for the kernel
            is_fp8 = args.mma_type == 'fp8xfp8'
            num_weight_experts = num_experts if use_emu_reference else num_experts_per_rank
            all_l1_weights = make_weights(num_weight_experts, intermediate_hidden * 2, hidden, is_fp8, True, True)
            all_l2_weights = make_weights(num_weight_experts, hidden, intermediate_hidden, is_fp8, False, True)
            local_experts = slice(rank_idx * num_experts_per_rank, (rank_idx + 1) * num_experts_per_rank) \
                if use_emu_reference else slice(None)
            l1_weights, l2_weights = ((w[local_experts], _to_mn_major_sf(e[local_experts].view(torch.int)))
                                      for w, e in (all_l1_weights, all_l2_weights))
            if num_shared_experts > 0:
                hi_shared_weights = ref_shared_weights = tuple((w[0], e[0]) for w, e in (
                    make_weights(1, shared_intermediate_hidden * 2, hidden, True, True, False),
                    make_weights(1, hidden, shared_intermediate_hidden, True, False, False)))
                shared_l1_weights, shared_l2_weights = ((w, _to_mn_major_sf(e.view(torch.int))) for w, e in ref_shared_weights)
                if arch_major == 9:
                    # The kernel takes the weights requantized by `transform_weights_for_mega_moe`
                    ref_shared_weights = tuple(
                        (w, sf.view(torch.uint8)) for w, sf in (deep_gemm.requantize_fp8_weights_for_sm90_mega_moe(
                            (w, e.view(torch.int))) for w, e in ref_shared_weights))

        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                l1_weights, l2_weights, activation=args.activation))
        if num_shared_experts > 0:
            transformed_shared_l1_weights, transformed_shared_l2_weights = (
                deep_gemm.transform_weights_for_mega_moe(
                    shared_l1_weights, shared_l2_weights, activation=args.activation))
            if arch_major == 9:
                # Requantizing in the kernel layout gives the same weights as before `transform_weights_for_mega_moe`
                (l1_w, l1_sf), (l2_w, l2_sf) = shared_l1_weights, shared_l2_weights
                for transformed, kernel_layout in (
                        (transformed_shared_l1_weights, (deep_gemm.mega._interleave_weights(l1_w),
                                                         deep_gemm.mega._transpose_sf_for_utccp(deep_gemm.mega._interleave_weights(l1_sf)))),
                        (transformed_shared_l2_weights, (l2_w, deep_gemm.mega._transpose_sf_for_utccp(l2_sf)))):
                    w, sf = deep_gemm.requantize_fp8_weights_for_sm90_mega_moe(kernel_layout, is_sf_transposed_for_utccp=True)
                    assert torch.equal(w.view(torch.uint8), transformed[0].view(torch.uint8)) and torch.equal(sf, transformed[1])
        else:
            transformed_shared_l1_weights = transformed_shared_l2_weights = None

        if args.use_localization:
            def localize(transformed):
                if isinstance(transformed, tuple):
                    return deep_gemm.localize(transformed[0]), transformed[1]
                return deep_gemm.localize(transformed)
            transformed_l1_weights, transformed_l2_weights = localize(transformed_l1_weights), localize(transformed_l2_weights)
            if num_shared_experts > 0:
                transformed_shared_l1_weights, transformed_shared_l2_weights = localize(transformed_shared_l1_weights), localize(transformed_shared_l2_weights)
            deep_gemm.destroy_localizer()

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def copy_inputs_to_buffer():
        if is_bf16xbf16:
            buffer.x[:num_tokens].copy_(x)
        else:
            buffer.x[:num_tokens].copy_(x[0])
            buffer.x_sf[:num_tokens].copy_(x[1])
            if num_shared_experts > 0:
                _copy_fp8_sf(buffer.shared_l1_acts_sf, shared_l1_x_sf, num_tokens)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def run_fused():
        cumulative_local_expert_recv_stats_fused.copy_(initial_cumulative_local_expert_recv_stats_fused)
        copy_inputs_to_buffer()

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            activation_alpha=args.activation_alpha,
            activation_beta=args.activation_beta)
        if args.post_expert_norm:
            kernel_kwargs.update(post_expert_norm_gamma=norm_gamma,
                                 post_expert_norm_eps=1e-8)
        if arch_major == 9:
            kernel_kwargs.update(num_tokens_for_block_m=num_tokens_for_block_m)
        if num_shared_experts > 0:
            kernel_kwargs.update(
                shared_l1_weights=transformed_shared_l1_weights,
                shared_l2_weights=transformed_shared_l2_weights
            )
        (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Activation: type={args.activation}, limit={args.activation_clamp}, '
               f'alpha={args.activation_alpha}, beta={args.activation_beta}',
               once_in_node=True)
    dist_print(f' > Localized weights: {args.use_localization}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Shared experts: {num_shared_experts}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Block M: {block_m}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        return block_m, False

    if args.post_expert_norm:
        assert is_bf16xbf16 and num_shared_experts == 0
        create_inputs()
        topk_weights.copy_(topk_weights.softmax(dim=-1).masked_fill(topk_idx < 0, 0))
        # Small down scales exercise epsilon and prevent a routing-before-norm
        # implementation from passing via approximate scale invariance.
        l1_weights.mul_(0.1)
        l2_weights.mul_(1e-5)
        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights))
        all_w1 = [torch.empty_like(l1_weights) for _ in range(num_ranks)]
        all_w2 = [torch.empty_like(l2_weights) for _ in range(num_ranks)]
        dist.all_gather(all_w1, l1_weights, group=group)
        dist.all_gather(all_w2, l2_weights, group=group)
        all_w1, all_w2 = torch.cat(all_w1), torch.cat(all_w2)
        nvfp4_buffer = deep_gemm.SymmBuffer(
            group, num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, mma_type='fp4xfp4', base=buffer)
        assert nvfp4_buffer.buffer.data_ptr() == buffer.buffer.data_ptr()
        assert nvfp4_buffer.handle is buffer.handle
        def zero_fp4_weights(n, k):
            return (
                torch.zeros((num_experts_per_rank, n, k // 2), dtype=torch.int8, device='cuda'),
                torch.full((num_experts_per_rank, n, k // 64), 0x38383838,
                           dtype=torch.int32, device='cuda').transpose(-1, -2).contiguous().transpose(-1, -2))
        fp4_w1, fp4_w2 = deep_gemm.transform_weights_for_mega_moe(
            zero_fp4_weights(intermediate_hidden * 2, hidden),
            zero_fp4_weights(hidden, intermediate_hidden))
        def run_nvfp4():
            nvfp4_buffer.x[:num_tokens].zero_()
            nvfp4_buffer.x_sf[:num_tokens].fill_(0x38383838)
            nvfp4_buffer.topk_idx[:num_tokens].copy_(topk_idx)
            nvfp4_buffer.topk_weights[:num_tokens].copy_(topk_weights)
            result = torch.empty_like(x)
            deep_gemm.fp4_fp4_mega_moe(
                result, fp4_w1, fp4_w2, nvfp4_buffer,
                post_expert_norm_gamma=norm_gamma, post_expert_norm_eps=1e-8)
            return result
        expected = torch.zeros((num_tokens, hidden), device="cuda", dtype=torch.float32)
        for expert in range(num_experts):
            token, slot = (topk_idx == expert).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue
            gate, up = (x[token].float() @ all_w1[expert].float().T).bfloat16().chunk(2, -1)
            if args.activation_clamp is not None:
                gate = gate.clamp(max=args.activation_clamp)
                up = up.clamp(-args.activation_clamp, args.activation_clamp)
            intermediate = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
            rows = (intermediate.float() @ all_w2[expert].float().T).bfloat16().float()
            scaled = rows * norm_gamma
            normalized = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + 1e-8)
            expected.index_add_(0, token, normalized * topk_weights[token, slot, None])
        for _ in range(3):
            torch.testing.assert_close(run_nvfp4(), torch.zeros_like(x), atol=0, rtol=0)
            actual, _ = run_fused()
            relative_error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)
            assert relative_error < 0.005, relative_error.item()
            torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nvfp4_captured = run_nvfp4()
            captured, _ = run_fused()
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(nvfp4_captured, torch.zeros_like(x), atol=0, rtol=0)
            torch.testing.assert_close(captured.float(), expected, atol=0.04, rtol=0.02)
        dist_print("BF16 post-expert norm: eager and CUDA Graph reference checks passed", once_in_node=True)
        dist.barrier()
        buffer.destroy()
        return block_m, False

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine (not with the PyTorch reference)
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = \
        (None, None, None, False) if use_emu_reference else import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    ep_buffer_type = (getattr(deep_ep, 'EPBuffer', None) or deep_ep.ElasticBuffer) if is_legacy_loaded else None
    ep_buffer = ep_buffer_type(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=not is_bf16xbf16,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
    ) if is_legacy_loaded else None

    # Baseline params differ by mma type
    run_baseline = None
    if is_legacy_loaded:
        if is_bf16xbf16:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False, 'do_expand': True}
            gemm_fn = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
            gemm_kwargs = {'compiled_dims': '', 'use_psum_layout': True}
            activation_kwargs = {'round_scale': False, 'ue8m0_scale': False, 'output_bf16': True}
            get_num_tokens = lambda recv_x: recv_x.size(0)
        else:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False,
                               'do_expand': True, 'use_tma_aligned_col_major_sf': True}
            gemm_fn = deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous
            gemm_kwargs = {'use_psum_layout': True, 'recipe': (1, 1, 32)}
            activation_kwargs = {'round_scale': True, 'ue8m0_scale': True, 'output_bf16': False}
            get_num_tokens = lambda recv_x: recv_x[0].size(0)

        def get_baseline_shared_bias() -> Optional[torch.Tensor]:
            if num_shared_experts == 0:
                return None

            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            if is_bf16xbf16:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.bf16_gemm_nt(x, shared_l1_weights, l1_out)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=128, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    activation=args.activation,
                    alpha=args.activation_alpha, beta=args.activation_beta,
                    round_scale=False, ue8m0_scale=False, output_bf16=True)[-1]
                deep_gemm.bf16_gemm_nt(l2_in, shared_l2_weights, y)
            else:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.fp8_gemm_nt(shared_x, shared_l1_weights, l1_out, recipe=(1, 1, 32), disable_ue8m0_cast=True)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=32, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    activation=args.activation,
                    alpha=args.activation_alpha, beta=args.activation_beta,
                    round_scale=True, ue8m0_scale=True, output_bf16=False)
                deep_gemm.fp8_gemm_nt(l2_in, shared_l2_weights, y, recipe=(1, 1, 32), disable_ue8m0_cast=True)
            return y

        def run_baseline():
            cumulative_local_expert_recv_stats_baseline.copy_(initial_cumulative_local_expert_recv_stats_baseline)
            # Dispatch
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                num_experts=num_experts, expert_alignment=alignment,
                **dispatch_kwargs)
            num_recv_tokens = get_num_tokens(recv_x)

            # L1 GEMM
            l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
            gemm_fn(recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # Gated activation
            activation_result = tilelang_ops.swiglu_apply_weight_to_fp8(
                x=l1_y, topk_weights=recv_topk_weights,
                avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
                num_per_channels=32, use_col_major_scales=True,
                clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                activation=args.activation,
                alpha=args.activation_alpha, beta=args.activation_beta,
                **activation_kwargs)
            l1_y = activation_result[-1] if is_bf16xbf16 else activation_result

            # L2 GEMM
            l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            gemm_fn(l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # Combine
            return (
                ep_buffer.combine(l2_y, handle=handle, bias=get_baseline_shared_bias())[0],
                cumulative_local_expert_recv_stats_baseline
            )

    # Tokens received by each local expert, from the gathered top-k indices
    def get_num_recv_tokens_per_expert() -> torch.Tensor:
        gathered_topk_idx = uneven_all_gather(topk_idx, group=group) - rank_idx * num_experts_per_rank
        return torch.bincount(gathered_topk_idx[(gathered_topk_idx >= 0) & (gathered_topk_idx < num_experts_per_rank)],
                              minlength=num_experts_per_rank).int()

    # Check correctness
    # noinspection PyBroadException
    if use_emu_reference and num_correctness_tests > 0:
        dist_print('Running correctness tests against the PyTorch reference:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, fused_stats = run_fused()
            fused_stats = fused_stats.clone()
            assert torch.equal(run_fused()[0], fused_y), 'Relaunch is not bit-identical'
            assert torch.equal(fused_stats, initial_cumulative_local_expert_recv_stats_fused + get_num_recv_tokens_per_expert())
            ref_y = _moe_reference(
                functools.partial(_emu_expert, gran_k=gran_k, args=args), ref_x, topk_idx, topk_weights,
                all_l1_weights, all_l2_weights, ref_shared_weights, torch.float).bfloat16()
            hi_y = _moe_reference(
                functools.partial(_hi_expert, args=args), (ref_x_hi, ), topk_idx, topk_weights,
                all_l1_weights, all_l2_weights, hi_shared_weights, torch.double)
            diff, hi_diff = calc_diff(fused_y, ref_y), calc_diff(fused_y, hi_y)
            max_token_diff = max(_calc_token_diff(fused_y, ref_y).tolist(), default=0)
            dist_print(f' > Accuracy (rank {rank_idx}): calc_diff={diff:.3e}, max_token_diff={max_token_diff:.3e}, hi_diff={hi_diff:.3e}')
            if is_exact_inputs:
                assert torch.equal(fused_y, ref_y), f'calc_diff={diff}'
            # NOTES: Hopper FP8 WGMMA accumulates with less than FP32 precision (about 1e-5 calc_diff against this FP32 emulation),
            # so random inputs get a looser tolerance
            max_diff, max_token_diff_limit = (1e-4, 1e-2) if arch_major == 9 else (1e-5, 1e-4)
            assert diff < max_diff and max_token_diff < max_token_diff_limit and hi_diff < 2e-3, \
                f'calc_diff={diff}, max_token_diff={max_token_diff}, hi_diff={hi_diff}'
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)

        # Replays of one CUDA graph alternate two input sets, and must match the eager launches bit for bit
        graph_inputs = [*x, topk_weights]
        input_sets = [[t.clone() for t in graph_inputs], [t.view(torch.uint8).roll(1, 0).view(t.dtype) for t in x] + [-topk_weights]]
        eager_ys = []
        for input_set in input_sets:
            for t, s in zip(graph_inputs, input_set):
                t.copy_(s)
            eager_ys.append(run_fused()[0].clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_y = run_fused()[0]
        for i in range(8):
            for t, s in zip(graph_inputs, input_sets[i % 2]):
                t.copy_(s)
            graph.replay()
            assert torch.equal(graph_y, eager_ys[i % 2]), f'CUDA graph replay {i} differs from the eager launch'
        dist_print(once_in_node=True)
    elif is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, fused_stats = run_fused()
            baseline_y, baseline_stats = run_baseline()
            assert torch.equal(fused_stats, baseline_stats)
            assert torch.equal(fused_y, baseline_y), f'calc_diff={calc_diff(fused_y, baseline_y)}'
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens, and whether the pool outlives the L1/L2 ring
    num_recv_tokens_per_expert = get_num_recv_tokens_per_expert()
    num_recv_tokens = num_recv_tokens_per_expert.sum().item()
    is_ring_wrapped = ceil_div(num_recv_tokens_per_expert, block_m).sum().item() > buffer.l1_acts.size(0) // block_m

    # Benchmark
    barrier_kwargs = {'wait_comm_stream': False} if hasattr(deep_ep, 'EPBuffer') else {'use_comm_stream': False}
    barrier_fn = lambda: ep_buffer.barrier(**barrier_kwargs) if ep_buffer else dist.all_reduce(torch.empty(1, device='cuda'))
    trace_path = None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json'
    t_fused = bench_kineto(run_fused, 'mega_moe', barrier=barrier_fn, trace_path=trace_path)
    t_baseline = tilelang_bench(
        run_baseline, _n_warmup=5, _n_repeat=1,
        backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: routed + shared L1/L2, each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    num_routed_flops = 2 * num_recv_tokens * hidden * intermediate_hidden * 3
    num_shared_flops = 2 * num_tokens * hidden * shared_intermediate_hidden * 3
    num_total_flops = num_routed_flops + num_shared_flops

    # HBM bytes: weights + activations + output
    num_touched_experts = (num_recv_tokens_per_expert > 0).sum().item()
    act_elem_size = 2 if is_bf16xbf16 else 1
    routed_weight_elem_size = 2 if is_bf16xbf16 else (1 if args.mma_type == 'fp8xfp8' else 0.5)
    shared_weight_elem_size = 2 if is_bf16xbf16 else 1
    num_routed_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * routed_weight_elem_size      # L1 weights
        + num_touched_experts * hidden * intermediate_hidden * routed_weight_elem_size        # L2 weights
        + num_recv_tokens * hidden * act_elem_size                                            # L1 acts read
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L1 output write
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L2 acts read
        + num_recv_tokens * hidden * 2                                                        # L2 output write (always BF16)
    )
    num_shared_hbm_bytes = 0 if num_shared_experts == 0 else (
        shared_intermediate_hidden * 2 * hidden * shared_weight_elem_size      # Shared L1 weights
        + hidden * shared_intermediate_hidden * shared_weight_elem_size        # Shared L2 weights
        + num_tokens * hidden * act_elem_size                                  # Shared L1 acts read
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L1 output write
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L2 acts read
        + num_tokens * hidden * 2                                              # Shared L2 output write
    )
    num_hbm_bytes = num_routed_hbm_bytes + num_shared_hbm_bytes

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    def print_perf(elapsed: float, ref_time: float, ref_label: str):
        tflops = safe_div(num_total_flops / 1e12, elapsed)
        hbm_gbs = safe_div(num_hbm_bytes / 1e9, elapsed)
        nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, elapsed)
        approx_factor = safe_div(elapsed, elapsed - t_reduction)
        dist_print(f' > EP {rank_idx:2}/{num_ranks} | '
                   f'{tflops:4.0f} TFLOPS | '
                   f'overlap: '
                   f'{tflops * approx_factor:4.0f} TFLOPS, '
                   f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
                   f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
                   f'{elapsed * 1e6:4.0f} us, '
                   f'reduction: {t_reduction * 1e6:4.1f} us | '
                   f'{safe_div(ref_time, elapsed):.2f}x {ref_label}')

    dist_print(f'Performance (w/{"" if num_shared_experts else "o"} shared):', once_in_node=True)
    print_perf(t_fused, t_baseline, f'legacy{"+shared" if num_shared_experts else ""}')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    return block_m, is_ring_wrapped


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if args.sweep:
        results = [test_case(rank_idx, num_ranks, group, case_args) for case_args in _get_sweep_args(args, num_ranks)]
        is_ring_wrapped = torch.tensor([any(is_ring_wrapped for _, is_ring_wrapped in results)], dtype=torch.int, device='cuda')
        dist.all_reduce(is_ring_wrapped, op=dist.ReduceOp.MAX, group=group)
        assert is_ring_wrapped.item(), 'No sweep case wraps the ring'
        if num_ranks == 1 and get_arch_major() == 9:
            assert {block_m for block_m, _ in results} == {16, 32, 64, 128}
    else:
        test_case(rank_idx, num_ranks, group, argparse.Namespace(**_CASE_DEFAULTS, **vars(args)))
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=None, help='Number of tokens per rank (follow max minus removed if unset)')
    parser.add_argument('--num-tokens-per-rank', type=str, default=None, help='Comma-separated tokens of each rank (overrides --num-tokens)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--num-shared-experts', type=int, default=1, help='Number of shared experts (use 0 to disable)')
    parser.add_argument('--activation', type=str, default='swiglu', choices=('swiglu', 'swiglustep', 'situ'))
    parser.add_argument('--activation-clamp', type=float, default=10, help='SwiGLU clamp value')
    parser.add_argument('--activation-alpha', type=float, default=1.0,
                        help='SwiGLU sigmoid scale or SiTU gate tanh scale')
    parser.add_argument('--activation-beta', type=float, default=0.0,
                        help='SwiGLU up bias or SiTU linear tanh scale (0 disables it)')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--routing', type=str, default='random', choices=('random', 'hot', 'empty'),
                        help='Every token picks expert 0 (hot), or odd experts get no tokens (empty)')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--mma-type', type=str, default='fp8xfp4', choices=('fp8xfp4', 'fp8xfp8', 'bf16xbf16'))
    parser.add_argument('--use-localization', action='store_true', help='Localize the weights into the locality domains (requires MLOPart)')

    # Test settings
    parser.add_argument('--post-expert-norm', action='store_true')
    parser.add_argument('--empty-rank', type=int, default=-1)
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--sweep', action='store_true', help='Run the shape sweep (heaviest 20 cases with more than 1 rank)')
    parser.add_argument('--emu-reference', action='store_true', help='Check against the PyTorch reference (always on SM90)')
    parser.add_argument('--shared-gran-k', type=int, default=None, choices=(32, 128),
                        help='K granularity of the shared weight scales (default: 128 on SM90, 32 on SM100)')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
