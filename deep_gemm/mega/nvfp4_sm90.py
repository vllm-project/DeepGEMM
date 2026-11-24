"""NVFP4 (E2M1 codes, E4M3 scales per 16 K, FP32 global scale) weights for the SM90 Mega MoE, decoded in registers
by `sm90_nvfp4_dequant.cuh`."""
from typing import Tuple

import torch

# Decoded BF16 weights are `e2m1 * e4m3 * 2 ^ -GLOBAL_SCALE_LOG2`, so the global scales carry `2 ^ GLOBAL_SCALE_LOG2`
GLOBAL_SCALE_LOG2 = 13

# The `(rotation, mask)` pieces of the 4 BF16x2 outputs of a code word: `rotr(word, rotation) & mask`, ORed per output.
# A code `s e1 e0 m` sits at BF16 bits 15 and 8..6 (+16 for the high half), which reads as `e2m1 * 2 ^ -126`
_CODE_PIECES = (((0, 0x81c081c0), ), ((4, 0x81c081c0), ), ((10, 0x81c081c0), ), ((14, 0x00400040), (6, 0x81808180)))
_CODE_BIT_OF_BF16_BIT = {15: 3, 8: 2, 7: 1, 6: 0}


def _make_code_word_gather() -> Tuple[torch.Tensor, torch.Tensor]:
    # For word bit `b`: the element `2 * output + half` and the code bit it holds
    elements, code_bits = [None] * 32, [None] * 32
    for output, pieces in enumerate(_CODE_PIECES):
        for rotation, mask in pieces:
            for bit in range(32):
                if mask >> bit & 1:
                    src = (bit + rotation) % 32
                    assert elements[src] is None
                    elements[src], code_bits[src] = 2 * output + bit // 16, _CODE_BIT_OF_BF16_BIT[bit % 16]
    return torch.tensor(elements), torch.tensor(code_bits, dtype=torch.uint8)


_CODE_WORD_ELEMENTS, _CODE_WORD_BITS = _make_code_word_gather()


def _make_sf_encoding() -> torch.Tensor:
    # E4M3 `s = 2 ^ e * (1 + m / 8)`, subnormals renormalised, to `(e + 16) << 3 | m`; zero to 0
    values = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).double()
    mantissas, exponents = torch.frexp(values)
    codes = ((exponents + 15) << 3) | (mantissas * 16 - 8).to(torch.int32)
    return torch.where(values > 0, codes, 0).to(torch.uint8)


_SF_ENCODING = _make_sf_encoding()


def _interleave_gate_up_rows(t: torch.Tensor) -> torch.Tensor:
    # Checkpoint rows `2j` (gate j) and `2j + 1` (up j) to 16-row groups of 8 gate rows and the 8 matching up rows
    e, n, *rest = t.shape
    return t.view(e, n // 16, 8, 2, *rest).transpose(2, 3).reshape(e, n, *rest)


def _relayout_codes(codes: torch.Tensor) -> torch.Tensor:
    # Nibbles `[E, N, K]` to `[E, N, K / 2]` bytes: per 128 K, 16-byte chunk `q` holds the 4 words of k16 step pairs `w`,
    # whose 8 codes are columns `16 * (2w + o / 2) + 8 * (o % 2) + 2q + {0, 1}` of the 4 outputs `o`
    e, n, k = codes.shape
    elements = codes.view(e, n, k // 128, 4, 2, 2, 4, 2).permute(0, 1, 2, 6, 3, 4, 5, 7).reshape(e, n, k // 128, 4, 4, 8)
    bits = (elements[..., _CODE_WORD_ELEMENTS.to(codes.device)] >> _CODE_WORD_BITS.to(codes.device)) & 1
    weights = (1 << torch.arange(8, dtype=torch.uint8, device=codes.device))
    return (bits.view(*bits.shape[:-1], 4, 8) * weights).sum(dim=-1, dtype=torch.uint8).view(e, n, k // 2)


def _relayout_sf(sf: torch.Tensor) -> torch.Tensor:
    # Encoded bytes `[E, N, K / 16]` to `[E, K / 128, N, 2]` words: per K block, 16-row group and `i < 8`, the 4 words
    # `rotl(s[i][2w] | s[i][2w + 1] << 8 | s[i + 8][2w] << 16 | s[i + 8][2w + 1] << 24, 4)`
    e, n, num_sf = sf.shape
    pairs = sf.view(e, n // 16, 2, 8, num_sf // 8, 4, 2).permute(0, 4, 1, 3, 5, 2, 6).to(torch.int64)
    v = (pairs.flatten(-2) << torch.tensor([0, 8, 16, 24], device=sf.device)).sum(dim=-1)
    words = ((v << 4) | (v >> 28)) & 0xffffffff
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32).view(e, num_sf // 8, n, 2)


def _transform(w: torch.Tensor, sf: torch.Tensor, global_scale: torch.Tensor,
               is_l1: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e, n, half_k = w.shape
    if n % 16 != 0 or half_k % 64 != 0 or sf.shape != (e, n, half_k // 8) or global_scale.shape != (e, ):
        raise ValueError(f'Unsupported NVFP4 shapes: weights {tuple(w.shape)}, scales {tuple(sf.shape)}, '
                         f'global scales {tuple(global_scale.shape)}')
    if w.dtype not in (torch.uint8, torch.int8) or sf.dtype != torch.float8_e4m3fn or global_scale.dtype != torch.float:
        raise ValueError(f'Unsupported NVFP4 dtypes: {w.dtype}, {sf.dtype}, {global_scale.dtype}')
    w, sf_bytes = w.view(torch.uint8), sf.view(torch.uint8)
    for i in range(e):
        if bool(((sf_bytes[i] & 0x80) != 0).any()) or bool(((sf_bytes[i] & 0x7f) == 0x7f).any()):
            raise ValueError('NVFP4 scales must be non-negative and not NaN')

    # Row chunks keep every temporary at a few MiB (the largest takes 8 bytes per weight byte), small enough for the
    # caching allocator to reuse under a low `max_split_size_mb`
    num_rows = max(16, (4 << 20) // (half_k * 8) // 16 * 16)
    out_w = torch.empty_like(w)
    out_sf = torch.empty((e, half_k // 64, n, 2), dtype=torch.int32, device=w.device)
    sf_encoding = _SF_ENCODING.to(w.device)
    for i in range(e):
        for r in range(0, n, num_rows):
            w_rows, sf_rows = w[i, r:r + num_rows][None], sf_bytes[i, r:r + num_rows][None]
            if is_l1:
                w_rows, sf_rows = _interleave_gate_up_rows(w_rows), _interleave_gate_up_rows(sf_rows)
            # Codes of zero-scale groups become zero: the decoded scale is nonzero there
            codes = torch.stack((w_rows & 0xf, w_rows >> 4), dim=-1).view(1, w_rows.shape[1], -1, 16)
            codes = torch.where(sf_rows[..., None] == 0, 0, codes).flatten(-2)
            out_w[i, r:r + num_rows] = _relayout_codes(codes)[0]
            out_sf[i, :, r:r + num_rows] = _relayout_sf(sf_encoding[sf_rows.long()])[0]
    return out_w, out_sf, global_scale * 2.0 ** GLOBAL_SCALE_LOG2


def transform_nvfp4_weights_for_sm90_mega_moe(
        l1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        l2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Checkpoint NVFP4 weights `(codes [E, N, K / 2], E4M3 scales [E, N, K / 16], FP32 global scales [E])` to the SM90
    Mega MoE layout `(codes [E, N, K / 2] u8, scales [E, K / 128, N, 2] i32, global scales * 2 ^ 13)`.

    L1 rows alternate gate and up (row `2j` is gate `j`). Raises `ValueError` on unsupported shapes or scales."""
    return _transform(*l1, is_l1=True), _transform(*l2, is_l1=False)
