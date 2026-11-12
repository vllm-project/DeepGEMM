"""Check architecture-specific MegaMoE tile choices and fallback boundaries."""

import pytest
import torch

import deep_gemm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                    reason='SM120 required')
@pytest.mark.parametrize('ranks,experts,topk,tokens,expected_block_m', [
    (1, 2, 2, 0, 16),
    (1, 2, 2, 10, 16),
    (1, 2, 2, 11, 32),
    (1, 2, 2, 24, 32),
    (1, 2, 2, 25, 64),
    (1, 8, 6, 13, 16),
    (1, 8, 6, 14, 32),
    (1, 8, 6, 32, 32),
    (1, 8, 6, 33, 64),
    (8, 384, 6, 80, 16),
    (8, 384, 6, 81, 32),
    (8, 384, 6, 192, 32),
    (8, 384, 6, 193, 64),
])
def test_sm120_mx_tile_boundaries(ranks, experts, topk, tokens, expected_block_m):
    assert deep_gemm.get_block_m_for_mega_moe(ranks, experts, 512, tokens, topk, 'fp8xfp4') == expected_block_m


@pytest.mark.parametrize('tokens,default_tile', [(0, 16), (31, 16), (32, 16),
                                                (39, 16), (64, 16), (65, 16),
                                                (128, 32), (256, 64)])
def test_nvfp4_tile_range(tokens, default_tile):
    if torch.cuda.get_device_capability()[0] == 12:
        pytest.skip('SM120 does not support NVFP4 MegaMoE')
    tuned_device = (torch.cuda.get_device_capability() == (10, 3)
                    and deep_gemm.get_num_sms() == 148)
    expected = 64 if tuned_device and 32 <= tokens <= 64 else default_tile
    assert deep_gemm.get_block_m_for_mega_moe(4, 256, 2048, tokens, 8, 'fp4xfp4') == expected


@pytest.mark.parametrize('ranks,experts,topk,mma', [
    (8, 256, 8, 'fp4xfp4'),
    (4, 512, 8, 'fp4xfp4'),
    (4, 256, 4, 'fp4xfp4'),
    (4, 256, 8, 'bf16xbf16'),
    (4, 256, 8, 'fp8xfp4'),
])
def test_other_workloads_keep_m16(ranks, experts, topk, mma):
    if torch.cuda.get_device_capability()[0] == 12 and mma != 'fp8xfp4':
        pytest.skip('SM120 MegaMoE supports only FP8 x FP4')
    assert deep_gemm.get_block_m_for_mega_moe(ranks, experts, 2048, 32, topk, mma) == 16


def test_reduced_sms_keep_m16():
    if torch.cuda.get_device_capability()[0] == 12:
        pytest.skip('SM120 does not support NVFP4 MegaMoE')
    original_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(original_sms - 2)
        assert deep_gemm.get_block_m_for_mega_moe(4, 256, 2048, 64, 8, 'fp4xfp4') == 16
    finally:
        deep_gemm.set_num_sms(original_sms)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                    reason='SM120 required')
@pytest.mark.parametrize('mma', ['fp4xfp4', 'bf16xbf16'])
def test_sm120_rejects_unsupported_mma_kinds(mma):
    with pytest.raises(RuntimeError, match='MXFP8FP4'):
        deep_gemm.get_block_m_for_mega_moe(4, 256, 2048, 32, 8, mma)


@pytest.mark.parametrize("mma", ["fp8xfp4", "fp8xfp8"])
@pytest.mark.parametrize("hidden,intermediate", [(512, 256), (256, 512), (512, 384)])
def test_mx_buffer_preserves_128_alignment(mma, hidden, intermediate):
    num_bytes, _ = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
        1, 8, 48, 2, hidden, intermediate, mma, "swiglu", 0
    )
    assert num_bytes > 0


@pytest.mark.parametrize("hidden,intermediate", [(512, 256), (256, 512)])
def test_nvfp4_buffer_accepts_256_alignment(hidden, intermediate):
    num_bytes, _ = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
        1, 8, 48, 2, hidden, intermediate, "fp4xfp4", "swiglu", 0
    )
    assert num_bytes > 0


@pytest.mark.parametrize("hidden,intermediate", [(512, 128), (512, 384)])
def test_nvfp4_buffer_rejects_non_256_alignment(hidden, intermediate):
    with pytest.raises(RuntimeError, match="Assertion error"):
        deep_gemm._C.get_symm_buffer_size_for_mega_moe(
            1, 8, 48, 2, hidden, intermediate, "fp4xfp4", "swiglu", 0
        )
