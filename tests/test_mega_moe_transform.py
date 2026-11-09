import torch

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8


def _make_weights(num_groups: int, n: int, k: int, is_fp4: bool, generator: torch.Generator):
    # Values with per-32 UE8M0 exponents, as (weights, SF words of 4 exponents per 128 K)
    w, sf = zip(*((per_token_cast_to_fp4 if is_fp4 else per_token_cast_to_fp8)(
        torch.randn((n, k), generator=generator) * k ** -0.5, use_ue8m0=True, gran_k=32) for _ in range(num_groups)))
    exps = (torch.stack(sf).view(torch.int) >> 23).to(torch.uint8)
    return torch.stack(w), exps.view(num_groups, n, -1).contiguous().view(torch.int)


def _transform(l1, l2, **kwargs):
    return deep_gemm.transform_weights_for_mega_moe(l1, l2, **kwargs)


def _assert_equal(x, y):
    if isinstance(x, (tuple, list)):
        assert len(x) == len(y)
        for a, b in zip(x, y):
            _assert_equal(a, b)
    else:
        assert x.device.type == 'cpu' or y.device.type == 'cpu' or x.device == y.device
        assert torch.equal(x.cpu().view(torch.uint8), y.cpu().view(torch.uint8))


def test_cpu_transform():
    # SM90 and generic layouts of FP4 and FP8 weights on the CPU, equal to the ones made on the GPU
    generator = torch.Generator().manual_seed(0)
    for is_fp4 in (True, False):
        l1 = _make_weights(2, 512, 256, is_fp4, generator)
        l2 = _make_weights(2, 256, 256, is_fp4, generator)
        for arch_major in (9, 10):
            cpu_result = _transform(l1, l2, arch_major=arch_major)
            assert all(t.device.type == 'cpu' for pair in cpu_result for t in pair)
            if torch.cuda.is_available():
                gpu_result = _transform(*(tuple(t.cuda() for t in w) for w in (l1, l2)), arch_major=arch_major)
                _assert_equal(cpu_result, gpu_result)
        # Without `arch_major`, CPU weights take the current GPU's layout, or the generic one without a GPU
        expected_arch_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 10
        _assert_equal(_transform(l1, l2), _transform(l1, l2, arch_major=expected_arch_major))


def test_weights_device():
    # Weights on another GPU than the current one use that GPU's architecture
    if torch.cuda.device_count() < 2:
        return
    generator = torch.Generator().manual_seed(1)
    l1, l2 = _make_weights(1, 256, 256, True, generator), _make_weights(1, 256, 128, True, generator)
    torch.cuda.set_device(0)
    result = _transform(*(tuple(t.to('cuda:1') for t in w) for w in (l1, l2)))
    assert all(t.device == torch.device('cuda:1') for pair in result for t in pair)
    _assert_equal(result, _transform(l1, l2, arch_major=torch.cuda.get_device_capability(1)[0]))


def test_fp4_sf_zero_groups():
    # SM90 SF words of FP4 weights with zero padding, on the CPU
    exps = torch.full((5, 8), 120, dtype=torch.uint8)
    exps[1] = 0                                   # Zero padding row: base 0 (zero weight scale) without the weights
    exps[2, 1] = 100                              # A group 20 below its row maximum
    exps[3, 4:] = 0                               # A zero padding 128 K block in a nonzero row, also without the weights
    exps[4] = 3                                   # A row with a maximum below 7
    sf = exps.view(torch.int)
    w = torch.full((5, 128), 0x11, dtype=torch.int8)
    w[1] = 0
    w[2, 16:32] = torch.tensor(0x88, dtype=torch.uint8).view(torch.int8)  # ... whose codes are all -0
    w[3, 64:] = 0
    try:
        deep_gemm.transform_fp4_sf_for_sm90_mega_moe(sf)
        assert False, 'A span of 20 must be rejected without the weights'
    except AssertionError as ex:
        assert 'span at most 11' in str(ex)
    words = deep_gemm.transform_fp4_sf_for_sm90_mega_moe(sf, w).view(torch.uint8).view(5, 2, 4)
    assert (words[:, :, 1] == 0).all()
    for i in (0, 2, 3):
        assert words[i].tolist() == [[114, 0, 0xbb, 0xbb]] * 2
    for i in (1, 4):
        assert words[i].tolist() == [[0, 0, 0, 0]] * 2
    exps[2, 1] = 120
    assert torch.equal(deep_gemm.transform_fp4_sf_for_sm90_mega_moe(exps.view(torch.int)), words.view(5, 2, 4).view(torch.int).squeeze(-1))


def test_fp4_exponent_span_clamp():
    # On the CPU: nonzero groups more than 11 exponents below their row maximum move to `max - 11` with re-rounded codes,
    # all other groups keep their bytes
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.double)

    def dequantize(w, sf):
        codes = w.view(torch.uint8)
        codes = torch.stack((codes & 0xf, codes >> 4), dim=-1).flatten(-2).long()
        scales = torch.exp2(sf.view(torch.uint8).double() - 127).repeat_interleave(32, dim=-1)
        return (e2m1[codes & 7] * (1 - 2 * (codes >> 3)) * scales).view(8, 256)

    # Row `r`: group 0 at the row maximum 140, group 1 at `129 - r`, group 2 all -0 at 1, the others at 135;
    # the nonzero groups hold all 16 codes twice
    w = (torch.arange(0, 16, 2) | (torch.arange(1, 16, 2) << 4)).to(torch.uint8).repeat(8, 16)
    w[:, 32:48] = 0x88
    exps = torch.tensor([140, 0, 1, 135, 135, 135, 135, 135], dtype=torch.uint8).repeat(8, 1)
    exps[:, 1] = 129 - torch.arange(8)
    w, sf = w.view(2, 4, 128).view(torch.int8), exps.view(2, 4, 8).view(torch.int)
    new_w, new_sf = deep_gemm.clamp_fp4_exponent_span_for_sm90_mega_moe(w, sf)
    deep_gemm.transform_fp4_sf_for_sm90_mega_moe(new_sf, new_w)

    is_clamped = torch.zeros((8, 8), dtype=torch.bool)
    is_clamped[1:, 1] = True
    assert torch.equal(new_sf.view(torch.uint8).view(8, 8), torch.where(is_clamped, 129, exps).to(torch.uint8))
    assert torch.equal(new_w.view(torch.uint8).view(8, 8, 16)[~is_clamped], w.view(torch.uint8).view(8, 8, 16)[~is_clamped])
    expected = dequantize(w, sf)
    x = expected[1:, 32:64] / 4
    codes = ((x.abs()[..., None] - e2m1).abs() + (torch.arange(8) & 1) * 2 ** -20).argmin(dim=-1)
    expected[1:, 32:64] = e2m1[codes] * x.sign() * 4
    assert torch.equal(dequantize(new_w, new_sf), expected)


def test_fp8_block_sf_layout():
    # The compact layout of `transform_weights_for_mega_moe`, read the way the SM90 kernel reads it (tile `j` takes word `j & 2` for
    # gate rows, `(j & 2) + 1` for up rows, `j & 3` for L2, of the box at `j & ~3`), gives every row the scale of its checkpoint block
    generator = torch.Generator().manual_seed(2)
    mega = deep_gemm.mega
    def read_like_kernel(sf, n, is_l1):
        rows = torch.arange(n)
        tiles = rows // 128
        words = torch.where(rows % 16 < 8, tiles & 2, (tiles & 2) + 1) if is_l1 else tiles & 3
        memory = torch.as_strided(sf, (sf.shape[0], sf.shape[2], sf.stride(-1)), (sf.stride(0), sf.stride(-1), 1))
        return memory[:, :, (tiles & ~3) + words].transpose(-1, -2)
    for num_groups, hidden, intermediate_hidden in ((2, 512, 384), (3, 768, 256), (1, 4096, 2048)):
        def make(n, k):
            w = torch.randn((num_groups, n, k), generator=generator).to(torch.float8_e4m3fn)
            return w, torch.rand((num_groups, n // 128, k // 128), generator=generator) + 0.5
        l1, l2 = make(intermediate_hidden * 2, hidden), make(hidden, intermediate_hidden)
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(l1, l2, arch_major=9)
        assert torch.equal(l1_w.view(torch.uint8), mega._interleave_weights(l1[0]).view(torch.uint8))
        assert torch.equal(l2_w.view(torch.uint8), l2[0].view(torch.uint8))
        assert torch.equal(read_like_kernel(l1_sf, intermediate_hidden * 2, True), mega._interleave_weights(l1[1].repeat_interleave(128, dim=-2)))
        assert torch.equal(read_like_kernel(l2_sf, hidden, False), l2[1].repeat_interleave(128, dim=-2))


if __name__ == '__main__':
    test_cpu_transform()
    test_weights_device()
    test_fp4_sf_zero_groups()
    test_fp4_exponent_span_clamp()
    test_fp8_block_sf_layout()
    print('CUDA available:', torch.cuda.is_available(), '- all transform tests passed')
