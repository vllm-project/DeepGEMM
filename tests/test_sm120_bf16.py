import random
import torch

import deep_gemm
from deep_gemm.testing import (
    calc_diff,
    get_arch_major, test_filter
)
from generators import (
    MajorTypeAB,
    generate_k_grouped_contiguous,
)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_small_n_output_row_stride_and_accumulation() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_num_sms(2)
        for n in (8, 32, 64):
            for dtype in (torch.float32, torch.bfloat16):
                for padding in (0, 64):
                    for accumulation in ('none', 'alias', 'separate'):
                        a = torch.ones((64, 128), dtype=torch.bfloat16, device='cuda')
                        b = torch.ones((n, 128), dtype=torch.bfloat16, device='cuda')
                        storage = torch.full((66, n + padding), -7, dtype=dtype, device='cuda')
                        d = storage[1:65, :n]
                        d.fill_(3)
                        c_storage = torch.full((64, n + 64), -11, dtype=dtype, device='cuda')
                        c = None if accumulation == 'none' else d if accumulation == 'alias' else c_storage[:, :n]
                        if c is not None:
                            c.fill_(3)
                        before_c = c_storage.clone()
                        deep_gemm.bf16_gemm_nt(a, b, d, c=c)
                        assert torch.equal(d.cpu().float(), torch.full((64, n), 128.0 + (3 if c is not None else 0)))
                        assert torch.all(storage[[0, -1]] == -7)
                        assert torch.all(storage[1:65, n:] == -7)
                        assert torch.equal(c_storage, before_c)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_contiguous_grouped_output_row_stride() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_num_sms(2)
        for padding in (0, 8):
            a = torch.ones((128, 128), dtype=torch.bfloat16, device='cuda')
            b = torch.ones((1, 16, 128), dtype=torch.bfloat16, device='cuda')
            storage = torch.full((130, 16 + padding), -7, dtype=torch.bfloat16, device='cuda')
            d = storage[1:129, :16]
            labels = torch.zeros(128, dtype=torch.int32, device='cuda')
            deep_gemm.set_mk_alignment_for_contiguous_layout(128)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, labels)
            assert torch.all(d == 128)
            assert torch.all(storage[[0, -1]] == -7)
            assert torch.all(storage[1:129, 16:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_kgroup_unequal_k_accumulation() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        deep_gemm.set_num_sms(8)
        ks = [128, 256]
        a = torch.cat([torch.full((k, 128), i + 1, dtype=torch.bfloat16, device='cuda') for i, k in enumerate(ks)])
        b = torch.ones((sum(ks), 128), dtype=torch.bfloat16, device='cuda')
        d = torch.full((2, 128, 128), 3, dtype=torch.float32, device='cuda')
        layout = torch.tensor(ks, dtype=torch.int32, device='cuda')
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, layout, d)
        for i, k in enumerate(ks):
            assert torch.all(d[i] == k * (i + 1) + 3)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_masked_physical_capacity() -> None:
    old_sms, old_pdl = deep_gemm.get_num_sms(), deep_gemm.get_pdl()
    try:
        for multiple in (1, 64):
            deep_gemm.set_block_size_multiple_of((1, multiple))
            for groups in (1, 2, 3):
                for capacity, valid in ((65, 65), (128, 65), (127, 127), (128, 128), (129, 129), (257, 257)):
                    for zero_group in (False, True):
                        masks = [0 if zero_group and g == 0 else valid for g in range(groups)]
                        a = torch.stack([torch.full((capacity, 128), g + 1, dtype=torch.bfloat16, device='cuda') for g in range(groups)])
                        b = torch.stack([torch.full((128, 128), 2 * g + 1, dtype=torch.bfloat16, device='cuda') for g in range(groups)])
                        storage = torch.full((groups * capacity * 128 + 128,), -7, dtype=torch.bfloat16, device='cuda')
                        d = storage[64:-64].view(groups, capacity, 128)
                        mask = torch.tensor(masks, dtype=torch.int32, device='cuda')
                        deep_gemm.set_num_sms(2 if capacity > 128 else 8)
                        for pdl in (False, True):
                            deep_gemm.set_pdl(pdl)
                            def call():
                                deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, mask, valid)
                            call()
                            for _ in range(3):
                                call()
                                torch.cuda.synchronize()
                                for g, count in enumerate(masks):
                                    assert torch.all(d[g, :count] == 128 * (g + 1) * (2 * g + 1)), (multiple, groups, capacity, masks, pdl, g)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                call()
                            graph.replay()
                            torch.cuda.synchronize()
                            for g, count in enumerate(masks):
                                assert torch.all(d[g, :count] == 128 * (g + 1) * (2 * g + 1)), (multiple, groups, capacity, masks, pdl, g)
                            assert torch.all(storage[:64] == -7) and torch.all(storage[-64:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_block_size_multiple_of((1, 1))


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_kgroup_zero_and_unequal_k() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_num_sms(2)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        for ks in ([0, 128, 256], [128, 0, 256], [128, 256, 0], [0, 0, 0], [128, 384, 256]):
            m, n = 256, 256
            a = torch.cat([torch.full((k, m), g + 1, dtype=torch.bfloat16, device='cuda') for g, k in enumerate(ks)])
            b = torch.cat([torch.full((k, n), 2 * g + 1, dtype=torch.bfloat16, device='cuda') for g, k in enumerate(ks)])
            storage = torch.full((len(ks) * m * n + 32,), -7, dtype=torch.float32, device='cuda')
            d = storage[16:-16].view(len(ks), m, n)
            d.fill_(3)
            layout = torch.tensor(ks, dtype=torch.int32, device='cuda')
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, layout, d)
            torch.cuda.synchronize()
            for g, k in enumerate(ks):
                assert torch.all(d[g] == k * (g + 1) * (2 * g + 1) + 3), (ks, g)
            assert torch.all(storage[:16] == -7) and torch.all(storage[-16:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_kgroup_unaligned_m_tail_tile_isolation() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_num_sms(2)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        # m=48 is not a multiple of BLOCK_M=64: a group's partial tail M tile must
        # stay inside its own slab (the D descriptor is one flat 2D map over all
        # groups, so TMA cannot clamp at group boundaries). The asymmetric ks make
        # group 0's tile finish last, so a tail-tile spill would overwrite group 1's
        # already-stored head rows.
        m, n, ks = 48, 128, [8192, 128]
        a = torch.cat([torch.full((k, m), g + 1, dtype=torch.bfloat16, device='cuda') for g, k in enumerate(ks)])
        b = torch.cat([torch.full((k, n), 2 * g + 1, dtype=torch.bfloat16, device='cuda') for g, k in enumerate(ks)])
        layout = torch.tensor(ks, dtype=torch.int32, device='cuda')
        for out_dtype in (torch.float32, torch.bfloat16):
            storage = torch.full((len(ks) * m * n + 32,), -7, dtype=out_dtype, device='cuda')
            d = storage[16:-16].view(len(ks), m, n)
            d.fill_(3)
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, layout, d)
            torch.cuda.synchronize()
            start = 0
            for g, k in enumerate(ks):
                ref = (a[start:start + k].float().T @ b[start:start + k].float() + 3).to(out_dtype)
                assert torch.equal(d[g], ref), (out_dtype, ks, g)
                start += k
            assert torch.all(storage[:16] == -7) and torch.all(storage[-16:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_kgroup_descriptor_reuse_at_default_sms() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_num_sms(torch.cuda.get_device_properties(0).multi_processor_count)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        for equal_groups in (False, True):
            random.seed(0)
            torch.manual_seed(0)
            ks = [121] * 3 if equal_groups else [max(1, int(128 * random.uniform(.7, 1.3))) for _ in range(8)]
            _, a, b, _, initial, _, layout, host_ks = generate_k_grouped_contiguous(
                len(ks), 768, 2048, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, ks,
                use_bf16=True, gran_k=128, k_alignment=128, use_psum_layout=False,
                accumulate=True, out_dtype=torch.float32)
            aa, bb, cc = a.cpu().double(), b.cpu().double(), initial.cpu().double()
            refs, start = [], 0
            for g, k in enumerate(host_ks):
                refs.append((aa[start:start+k].T @ bb[start:start+k] + cc[g]).float())
                start += k
            ref = torch.stack(refs)
            for separate in (False, True):
                first = None
                for _ in range(4):
                    out = initial.clone()
                    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                        a, b, out, host_ks, layout, initial if separate else out)
                    actual = out.cpu()
                    assert calc_diff(actual, ref) < 1e-5
                    assert torch.equal(initial.cpu().double(), cc)
                    if first is not None:
                        assert torch.equal(actual, first)
                    first = actual
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_sm120_kgroup_descriptor_reuse_at_default_sms()
    test_sm120_kgroup_zero_and_unequal_k()
    test_sm120_kgroup_unaligned_m_tail_tile_isolation()
    test_sm120_small_n_output_row_stride_and_accumulation()
    test_sm120_contiguous_grouped_output_row_stride()
    test_sm120_kgroup_unequal_k_accumulation()
    test_sm120_masked_physical_capacity()
