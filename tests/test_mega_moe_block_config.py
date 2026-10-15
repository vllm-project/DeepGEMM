"""Check the tuned NVFP4 range and preserve the existing fallback tile choices."""

import pytest
import torch

import deep_gemm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('tokens,default_tile', [(0, 16), (31, 16), (32, 16),
                                                (39, 16), (64, 16), (65, 16),
                                                (128, 32), (256, 64)])
def test_nvfp4_tile_range(tokens, default_tile):
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
    assert deep_gemm.get_block_m_for_mega_moe(ranks, experts, 2048, 32, topk, mma) == 16


def test_reduced_sms_keep_m16():
    original_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(original_sms - 2)
        assert deep_gemm.get_block_m_for_mega_moe(4, 256, 2048, 64, 8, 'fp4xfp4') == 16
    finally:
        deep_gemm.set_num_sms(original_sms)
