import functools
import os

import pytest
import torch

from deep_gemm.mega.nvfp4_sm90 import GLOBAL_SCALE_LOG2, transform_nvfp4_weights_for_sm90_mega_moe

_E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])

# Each thread decodes rows `r` and `r + 8` of a 64-row tile with `decode_k32` and stores its A fragments
_DECODE_KERNEL = r'''
#include <torch/extension.h>
#include <deep_gemm/common/sm90_nvfp4_dequant.cuh>

using namespace deep_gemm::sm90_nvfp4;

__global__ void decode_kernel(const uint8_t* codes, const uint32_t* sf, uint32_t* out, uint32_t n, uint32_t k) {
    const uint32_t lane_idx = threadIdx.x % 32, quad_idx = lane_idx % 4;
    const uint32_t expert_idx = blockIdx.x / (n / 64);
    const uint32_t row = blockIdx.x % (n / 64) * 64 + threadIdx.x / 32 * 16 + lane_idx / 4;
    for (uint32_t k_block_idx = 0; k_block_idx < k / 128; ++ k_block_idx) {
        const auto row_codes = [&](const uint32_t& r) {
            return reinterpret_cast<const uint4*>(codes + (expert_idx * n + r) * (k / 2) + k_block_idx * 64)[quad_idx];
        };
        const uint4 codes_r = row_codes(row), codes_r8 = row_codes(row + 8);
        const uint4 sf_unit = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint8_t*>(sf + (expert_idx * (k / 128) + k_block_idx) * n * 2) + get_sf_smem_offset(row));
        #pragma unroll
        for (uint32_t w = 0; w < 4; ++ w) {
            uint32_t a[2][4];
            decode_k32(reinterpret_cast<const uint32_t*>(&codes_r)[w], reinterpret_cast<const uint32_t*>(&codes_r8)[w],
                       reinterpret_cast<const uint32_t*>(&sf_unit)[w], a);
            #pragma unroll
            for (uint32_t s = 0; s < 2; ++ s) {
                const uint32_t col = k_block_idx * 128 + (2 * w + s) * 16 + 2 * quad_idx;
                for (uint32_t i = 0; i < 4; ++ i)
                    out[((expert_idx * n + row + i % 2 * 8) * k + col + i / 2 * 8) / 2] = a[s][i];
            }
        }
    }
}

torch::Tensor decode(const torch::Tensor& codes, const torch::Tensor& sf) {
    const uint32_t e = codes.size(0), n = codes.size(1), k = codes.size(2) * 2;
    auto out = torch::empty({e, n, k}, codes.options().dtype(torch::kBFloat16));
    decode_kernel<<<e * n / 64, 128>>>(codes.data_ptr<uint8_t>(), reinterpret_cast<const uint32_t*>(sf.data_ptr<int>()),
                                       reinterpret_cast<uint32_t*>(out.data_ptr()), n, k);
    return out;
}
'''


@functools.cache
def _decode_module():
    from torch.utils.cpp_extension import load_inline
    include_dir = os.path.join(os.path.dirname(__file__), '..', 'deep_gemm', 'include')
    return load_inline('dg_test_sm90_nvfp4_decode', cpp_sources='torch::Tensor decode(const torch::Tensor&, const torch::Tensor&);',
                       cuda_sources=_DECODE_KERNEL, functions=['decode'], extra_include_paths=[include_dir],
                       extra_cuda_cflags=['-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'])


def _reference(w: torch.Tensor, sf: torch.Tensor, is_l1: bool) -> torch.Tensor:
    # `e2m1 * e4m3 * 2 ^ -13` in BF16 (exact), rows in the kernel order
    codes = torch.stack((w & 0xf, w >> 4), dim=-1).flatten(-2).long()
    values = _E2M1_VALUES[codes] * sf.float().repeat_interleave(16, dim=-1) * 2.0 ** -GLOBAL_SCALE_LOG2
    if is_l1:
        e, n, k = values.shape
        values = values.view(e, n // 16, 8, 2, k).transpose(2, 3).reshape(e, n, k)
    assert torch.equal(values.bfloat16().float(), values)
    return values.bfloat16()


def _make_weights(e: int, n: int, k: int, generator: torch.Generator):
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, generator=generator)
    sf = torch.randint(0, 0x7f, (e, n, k // 16), dtype=torch.uint8, generator=generator).view(torch.float8_e4m3fn)
    return w, sf, torch.rand(e, generator=generator)


def _all_code_scale_pairs():
    # Every 16-group holds the 16 codes in a random order, the 127 valid scales (zero included) cycle over the groups
    generator = torch.Generator().manual_seed(0)
    n, k = 128, 256
    codes = torch.argsort(torch.rand((1, n, k // 16, 16), generator=generator), dim=-1).to(torch.uint8).view(1, n, k // 2, 2)
    sf = (torch.arange(n * k // 16) % 0x7f).to(torch.uint8).view(1, n, k // 16).view(torch.float8_e4m3fn)
    return codes[..., 0] | (codes[..., 1] << 4), sf, torch.ones(1)


def test_transform_shapes_and_scales():
    w, sf, gs = _make_weights(2, 32, 256, torch.Generator().manual_seed(1))
    (w1, sf1, gs1), (w2, sf2, gs2) = transform_nvfp4_weights_for_sm90_mega_moe((w, sf, gs), (w, sf, gs))
    assert w1.shape == w2.shape == w.shape and w1.dtype == torch.uint8
    assert sf1.shape == sf2.shape == (2, 2, 32, 2) and sf1.dtype == torch.int32
    assert torch.equal(gs1, gs * 2 ** GLOBAL_SCALE_LOG2) and torch.equal(gs2, gs1)
    for bad in (0x80, 0xb8, 0x7f, 0xff):
        bad_sf = sf.clone()
        bad_sf.view(torch.uint8)[1, 3, 5] = bad
        with pytest.raises(ValueError):
            transform_nvfp4_weights_for_sm90_mega_moe((w, bad_sf, gs), (w, sf, gs))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason='SM90 only')
@pytest.mark.parametrize('weights', ['all_pairs', 'random'])
def test_decode(weights):
    # The decoded BF16 weights equal the checkpoint's values exactly, and the CPU and GPU transforms are byte-equal
    generator = torch.Generator().manual_seed(2)
    l1, l2 = ((_all_code_scale_pairs(), ) * 2 if weights == 'all_pairs' else
              (_make_weights(3, 192, 384, generator), _make_weights(3, 384, 128, generator)))
    cpu = transform_nvfp4_weights_for_sm90_mega_moe(l1, l2)
    gpu = transform_nvfp4_weights_for_sm90_mega_moe(*(tuple(t.cuda() for t in l) for l in (l1, l2)))
    for cpu_l, gpu_l, (w, sf, _), is_l1 in zip(cpu, gpu, (l1, l2), (True, False)):
        for cpu_t, gpu_t in zip(cpu_l, gpu_l):
            assert torch.equal(cpu_t, gpu_t.cpu())
        decoded = _decode_module().decode(gpu_l[0], gpu_l[1]).cpu()
        assert torch.equal(decoded.float(), _reference(w, sf, is_l1).float())
