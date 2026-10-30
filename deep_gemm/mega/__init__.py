import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


_SUPPORTED_ACTIVATIONS = ('swiglu', 'swiglustep', 'situ')


def _validate_activation(activation: str) -> None:
    assert activation in _SUPPORTED_ACTIVATIONS, \
        f'Activation must be one of {_SUPPORTED_ACTIVATIONS}, got `{activation}`'


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None):
        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

        # Init
        _validate_activation(activation)
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


# TODO: remove this function
def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def requantize_fp8_weights_for_sm90_mega_moe(weights: Tuple[torch.Tensor, torch.Tensor],
                                              is_sf_transposed_for_utccp: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # One UE8M0 exponent per (row, 128 K) in all 4 bytes of each SF word, the largest of the 4 per-32 exponents
    # NOTES: with `is_sf_transposed_for_utccp`, the SF rows are in the UTCCP order of `transform_weights_for_mega_moe`
    w, sf = weights
    assert w.dtype == torch.float8_e4m3fn and sf.dtype == torch.int
    exps = torch.stack([(sf >> (8 * i)) & 0xff for i in range(4)], dim=-1)
    base_exps = exps.amax(dim=-1, keepdim=True)
    shifts = exps - base_exps
    if is_sf_transposed_for_utccp:
        # Undo `_transpose_sf_for_utccp`: SF row `4 * j + i` of each 128 rows belongs to weight row `32 * i + j`
        *dims, mn, num_sf, _ = shifts.shape
        shifts = shifts.reshape(*dims, mn // 128, 32, 4, num_sf, 4).transpose(-4, -3).reshape(*dims, mn, num_sf, 4)
    shifts = shifts.repeat_interleave(32, dim=-1).flatten(-2)
    uniform_sf = base_exps.to(torch.uint8).expand(*base_exps.shape[:-1], 4).contiguous().view(torch.int).squeeze(-1)
    return torch.ldexp(w.float(), shifts).to(torch.float8_e4m3fn), sf.clone().copy_(uniform_sf)


def relayout_fp4_weights_for_sm90_mega_moe(w: torch.Tensor) -> torch.Tensor:
    # Within each 64-byte K block of a row, 16-byte chunk `q` holds one word per 32-element group `g`: K `32g + 4q + i` with
    # the magnitude in bits `4i` and the sign in bit `8i + 7`, and K `32g + 16 + 4q + i` in bits `16 + 4i` and `8i + 3`
    assert w.dtype == torch.int8 and w.shape[-1] % 64 == 0
    pairs = w.view(torch.uint8).unflatten(-1, (-1, 4, 2, 4, 2))
    codes = torch.stack((pairs & 0xf, pairs >> 4), dim=-1).flatten(-2)
    first, second = codes[..., 0, :, :], codes[..., 1, :, :]
    magnitudes = torch.cat((first, second), dim=-1) & 0x7
    words = (magnitudes[..., 0::2] | (second & 0x8)) | ((magnitudes[..., 1::2] | (first & 0x8)) << 4)
    return words.transpose(-3, -2).flatten(-4).contiguous().view(torch.int8)


def clamp_fp4_exponent_span_for_sm90_mega_moe(w: torch.Tensor, sf: torch.Tensor,
                                              max_span: int = 11) -> Tuple[torch.Tensor, torch.Tensor]:
    # Raise every 32-element group more than `max_span` exponents below its row maximum (all-zero groups exempt) to
    # `max - max_span`, re-rounding its FP4 codes to nearest even, like Humming's `process_fused_e8m0_scale`
    # NOTES: `w` is packed (not relayouted); the added error is at most `2 ^ (max - max_span - 2)` per element
    assert w.dtype == torch.int8 and sf.dtype == torch.int and w.shape[-1] == sf.shape[-1] * 64
    exps = torch.stack([(sf >> (8 * i)) & 0xff for i in range(4)], dim=-1).flatten(-2)
    codes = w.view(torch.uint8).unflatten(-1, (-1, 16))
    is_nonzero = (codes & 0x77).amax(dim=-1) != 0
    max_exps = torch.where(is_nonzero, exps, 0).amax(dim=-1, keepdim=True)
    shifts = torch.where(is_nonzero, (max_exps - max_span - exps).clamp(min=0), 0)
    if not bool((shifts > 0).any()):
        return w, sf
    # `_REROUND[d][c]`: E2M1 magnitude code `c` times `2 ^ -d`, rounded to nearest even (all zero from `d = 5`)
    table = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 2, 2, 3, 4, 5], [0, 0, 0, 1, 1, 2, 2, 3],
                          [0, 0, 0, 0, 0, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1], [0] * 8], dtype=torch.uint8, device=w.device)
    d = shifts.clamp(max=5).to(torch.long)[..., None]
    rerounded = []
    for nibble in (codes & 0xf, codes >> 4):
        mag = table[d, (nibble & 0x7).to(torch.long)]
        rerounded.append(torch.where(mag != 0, mag | (nibble & 0x8), 0))
    new_w = (rerounded[0] | (rerounded[1] << 4)).flatten(-2).view(torch.int8)
    new_exps = (exps + shifts).to(torch.uint8).unflatten(-1, (-1, 4)).contiguous().view(torch.int).squeeze(-1)
    return new_w, sf.clone().copy_(new_exps)


def transform_fp4_sf_for_sm90_mega_moe(sf: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Per row, the base `max(e) - 6` in bits 7..0 of every SF word, and the 4 per-32 table shifts `e - base + 5` in bits 31..16
    # NOTES: all-zero 32-element groups of `w` (packed, not relayouted) are exempt from the span; without `w`, only all-zero SF words.
    # Rows with a maximum exponent below 7 get a zero scale
    assert sf.dtype == torch.int
    exps = torch.stack([(sf >> (8 * i)) & 0xff for i in range(4)], dim=-1)
    if w is not None:
        assert w.dtype == torch.int8 and w.shape[:-1] == sf.shape[:-1] and w.shape[-1] == sf.shape[-1] * 64
        is_nonzero = ((w.view(torch.uint8) & 0x77).unflatten(-1, (-1, 16)).amax(dim=-1) != 0).unflatten(-1, (-1, 4)).to(exps.device)
    else:
        is_nonzero = (exps != 0).any(dim=-1, keepdim=True).expand_as(exps)
    exps = torch.where(is_nonzero, exps, 0)
    max_exps = exps.amax(dim=(-2, -1), keepdim=True)
    is_zero_row = max_exps < 7
    exps = torch.where(is_nonzero, exps, max_exps)
    max_span = torch.where(is_zero_row, 0, max_exps - exps).amax().item()
    assert max_span <= 11, \
        (f'SM90 Mega MoE needs the UE8M0 exponents of each FP4 weight row to span at most 11, got {max_span}'
         f'{"" if w is not None else " (pass `w`, so that all-zero 32-element groups are exempt)"}')
    shifts = torch.where(is_zero_row, 0, exps - max_exps + 11).to(torch.uint8)
    words = torch.stack((torch.where(is_zero_row[..., 0], 0, max_exps[..., 0] - 6).to(torch.uint8).expand_as(shifts[..., 0]),
                         torch.zeros_like(shifts[..., 0]), shifts[..., 0] | (shifts[..., 1] << 4), shifts[..., 2] | (shifts[..., 3] << 4)),
                        dim=-1)
    return sf.clone().copy_(words.view(torch.int).squeeze(-1))


def _get_weights_arch_major(w: torch.Tensor) -> Optional[int]:
    # The architecture of the weights' GPU; for CPU weights, the current GPU's if there is one
    if w.is_cuda:
        return torch.cuda.get_device_capability(w.device)[0]
    return torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu',
    arch_major: Optional[int] = None
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    # `arch_major` defaults to the weights' GPU, e.g. pass 9 to prepare SM90 weights on the CPU
    _validate_activation(activation)
    if isinstance(l1_weights, tuple) and arch_major is None:
        arch_major = _get_weights_arch_major(l1_weights[0])
    if isinstance(l1_weights, tuple) and arch_major == 9:
        if l1_weights[0].dtype == torch.float8_e4m3fn:
            l1_weights, l2_weights = (requantize_fp8_weights_for_sm90_mega_moe(w) for w in (l1_weights, l2_weights))
        else:
            l1_weights, l2_weights = (clamp_fp4_exponent_span_for_sm90_mega_moe(*w) for w in (l1_weights, l2_weights))
            l1_weights, l2_weights = ((relayout_fp4_weights_for_sm90_mega_moe(w), transform_fp4_sf_for_sm90_mega_moe(sf, w))
                                      for w, sf in (l1_weights, l2_weights))
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Optional[Tuple[int, int, int]] = None,
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     activation_alpha: float = 1.0,
                     activation_beta: float = 0.0,
                     num_tokens_for_block_m: Optional[int] = None):
    # SM90: `num_tokens_for_block_m` (default: this rank's tokens) picks BLOCK_M; pass one value on all ranks
    _validate_activation(activation)
    recipe = recipe or ((1, 1, 128) if torch.cuda.get_device_capability(y.device)[0] == 9 else (1, 1, 32))
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        activation_alpha, activation_beta,
        num_tokens_for_block_m
    )

def fp4_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[torch.Tensor] = None,
                     shared_l2_weights: Optional[torch.Tensor] = None,
                     x_bf16: Optional[torch.Tensor] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 16),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     l1_alphas: Optional[torch.Tensor] = None,
                     l2_alphas: Optional[torch.Tensor] = None,
                     a2_scales: Optional[torch.Tensor] = None,
                     post_expert_norm_gamma: Optional[torch.Tensor] = None,
                     post_expert_norm_eps: float = 0.0,
                     weight_layout: str = "native"):
    """Run NVFP4 MegaMoE with native or explicitly prepared TRT-LLM weights.

    ``trtllm`` expects the shuffled SwiGLU [up; gate] weights and block-swizzled
    packed E4M3 scales produced by TRT-LLM weight preparation, not plain tensors
    with the same shape. L1 alphas remain in [gate, up] order. This path requires
    SM100-family GPUs, hidden sizes divisible by 256, and no fused shared experts.
    Currently scoped to Muse Spark, it requires ``post_expert_norm_gamma`` for
    per-expert RMSNorm before the routing-weighted reduction. The gain must be
    gain-centered by the caller. Tensor metadata cannot verify packed contents
    or gate/up ordering. Native layout retains optional post-expert norm.
    """
    if weight_layout not in ("native", "trtllm"):
        raise ValueError(f"Unsupported weight layout: {weight_layout}")
    # NVFP4 x NVFP4 fused MoE: weights and activations are packed E2M1 with
    # per-16-element E4M3 SFs. `l1_alphas`/`l2_alphas` are optional per-local-expert
    # scales (e.g. modelopt's `weight_scale_2`), applied before the activation
    # function (L1) and before the combine write-back (L2). `l1_alphas` is FP32 with
    # shape `(num_local_experts, 2)` holding separate gate/up factors; `l2_alphas`
    # is FP32 with shape `(num_local_experts, )`.
    # `a2_scales` is an optional FP32 `(num_local_experts, )` per-expert down-proj
    # input scale (modelopt's `input_scale`): the kernel normalizes the in-kernel
    # intermediate NVFP4 requant block SFs by `1 / a2_scale` (keeping them in E4M3's
    # well-represented range) and folds `a2_scale` back into the L2 alpha. Matches
    # flashinfer/TRT-LLM's recipe; omit (or `None`) for the plain unit-scale recipe.
    # Shared experts run in BF16 (no quantization): `shared_l1_weights` is
    # `(2 * shared_intermediate, hidden)`, `shared_l2_weights` is
    # `(hidden, shared_intermediate)`, and `x_bf16` provides the BF16 activations of
    # the local tokens (the buffer's `x` is packed FP4 and cannot be reused).
    # Under CUDA graphs, `x_bf16` must have a stable address, like `y`
    # `post_expert_norm_gamma` turns on the post-expert norm: every expert's
    # output is RMS-normalized before the top-k weighted reduction, instead of the
    # routing weight being applied in the L2 epilogue. It is an FP32 `(hidden, )` gain
    # applied *before* the RMS, so a zero-centered gamma must already have its gain
    # center added in. Incompatible with fused shared experts, whose contribution
    # would bypass the norm.
    assert (shared_l1_weights is None) == (shared_l2_weights is None) == (x_bf16 is None)
    assert post_expert_norm_gamma is None or shared_l1_weights is None, \
        'Post-expert norm cannot be combined with fused shared experts'
    op = _C.fp4_fp4_mega_moe_trtllm if weight_layout == "trtllm" else _C.fp4_fp4_mega_moe
    op(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights, x_bf16,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        l1_alphas, l2_alphas, a2_scales,
        post_expert_norm_gamma, post_expert_norm_eps
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True,
                  activation_alpha: float = 1.0,
                  activation_beta: float = 0.0,
                  post_expert_norm_gamma: Optional[torch.Tensor] = None,
                  post_expert_norm_eps: float = 0.0):
    assert activation in ('swiglu', 'swiglustep'), \
        f'BF16 Mega MoE supports `swiglu` or `swiglustep`, got `{activation}`'
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math,
        activation_alpha, activation_beta,
        post_expert_norm_gamma, post_expert_norm_eps
    )
