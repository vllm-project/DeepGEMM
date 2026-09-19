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


def _constant_fp8_fp4_with_unit_scales(shape, value=1, gran_k=128, fp4=False):
    data = torch.full((*shape[:-1], shape[-1] // 2), 0x22, dtype=torch.int8, device='cuda') if fp4 else \
        torch.full(shape, value, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    sf = torch.ones((*shape[:-1], (shape[-1] + gran_k - 1) // gran_k), dtype=torch.float32, device='cuda')
    return data, sf


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_split_k_accumulation() -> None:
    old_sms = deep_gemm.get_num_sms()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        for sms in (2, 8):
            deep_gemm.set_num_sms(sms)
            for value in (0, 1):
                for accumulation in ('none', 'alias', 'separate'):
                    a, b = quant((64, 4096), value), quant((64, 4096))
                    storage = torch.full((66, 128), -7, dtype=torch.float32, device='cuda')
                    d = storage[1:65, :64]
                    d.fill_(7)
                    c_storage = torch.full((64, 192), -11, dtype=torch.float32, device='cuda')
                    c = None if accumulation == 'none' else d if accumulation == 'alias' else c_storage[:, :64]
                    if c is not None:
                        c.fill_(7)
                    before_c = c_storage.clone()
                    deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, recipe=(1, 1, 128))
                    assert torch.all(d == value * 4096 + (7 if c is not None else 0))
                    assert torch.all(storage[[0, -1]] == -7) and torch.all(storage[1:65, 64:] == -7)
                    assert torch.equal(c_storage, before_c)
    finally:
        deep_gemm.set_num_sms(old_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_mixed_fp8_fp4_scale_tile_k_tail() -> None:
    old_sms = deep_gemm.get_num_sms()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_num_sms(2)
        for k in (128, 384, 640, 512):
            for fp4_a in (False, True):
                a = quant((64, k), fp4=fp4_a)
                b = quant((64, k), fp4=not fp4_a)
                d = torch.empty((64, 64), dtype=torch.bfloat16, device='cuda')
                deep_gemm.fp8_fp4_gemm_nt(a, b, d, recipe_a=(1, 128), recipe_b=(1, 128))
                assert torch.all(d == k)
    finally:
        deep_gemm.set_num_sms(old_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_strided_output_and_accumulation() -> None:
    old_sms = deep_gemm.get_num_sms()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_num_sms(2)
        for n in (8, 9, 64):
            for value in (0, 1):
                for accumulate in (False, True):
                    a, b = quant((64, 128), value), quant((n, 128))
                    storage = torch.full((66, 128), -7, dtype=torch.float32, device='cuda')
                    d = storage[1:65, :n]
                    d.fill_(7)
                    c_storage = torch.full((64, 192), -11, dtype=torch.float32, device='cuda')
                    c = c_storage[:, :n] if accumulate else None
                    if c is not None:
                        c.fill_(3)
                    before_c = c_storage.clone()
                    deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, recipe=(1, 1, 128))
                    assert torch.all(d == value * 128 + (3 if accumulate else 0))
                    assert torch.all(storage[[0, -1]] == -7) and torch.all(storage[1:65, n:] == -7)
                    assert torch.equal(c_storage, before_c)
    finally:
        deep_gemm.set_num_sms(old_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_odd_n_bf16_output_and_accumulation() -> None:
    old_sms = deep_gemm.get_num_sms()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_num_sms(2)
        for n in (8, 9):
            for accumulate in (False, True):
                a, b = quant((64, 128)), quant((n, 128))
                storage = torch.full((66, 16), -7, dtype=torch.bfloat16, device='cuda')
                d = storage[1:65, :n]
                d.fill_(3)
                deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=d if accumulate else None, recipe=(1, 1, 128))
                assert torch.all(d == (131 if accumulate else 128))
                assert torch.all(storage[[0, -1]] == -7) and torch.all(storage[1:65, n:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_batched_strided_output_and_accumulation() -> None:
    old_sms = deep_gemm.get_num_sms()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_num_sms(2)
        for n in (16, 64):
            a = quant((64, 2, 128))
            b = quant((2, n, 128))
            storage = torch.full((66, 2, n + 8), -7, dtype=torch.float32, device='cuda')
            d = storage[1:65, :, :n]
            c_storage = torch.full((2, 64, n), 3, dtype=torch.float32, device='cuda')
            c = c_storage.permute(1, 0, 2)
            before_c = c_storage.clone()
            deep_gemm.fp8_einsum('bhr,hdr->bhd', a, b, d, c=c, recipe=(1, 1, 128))
            assert torch.all(d == 131)
            assert torch.all(storage[[0, -1]] == -7) and torch.all(storage[1:65, :, n:] == -7)
            assert torch.equal(c_storage, before_c)
    finally:
        deep_gemm.set_num_sms(old_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_contiguous_grouped_output_row_stride() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_num_sms(2)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        for padding in (0, 8):
            a, b = quant((128, 128)), quant((1, 16, 128))
            storage = torch.full((130, 16 + padding), -7, dtype=torch.bfloat16, device='cuda')
            d = storage[1:129, :16]
            labels = torch.zeros(128, dtype=torch.int32, device='cuda')
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, labels, recipe=(1, 1, 128))
            assert torch.all(d == 128)
            assert torch.all(storage[[0, -1]] == -7) and torch.all(storage[1:129, 16:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_kgroup_nt_tn_layouts_and_accumulation() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        deep_gemm.set_num_sms(8)
        ks = [128, 256]
        ap = [quant((128, k), i + 1) for i, k in enumerate(ks)]
        bp = [quant((128, k)) for k in ks]
        layout = torch.tensor(ks, dtype=torch.int32, device='cuda')
        for transposed in (False, True):
            if transposed:
                a = (torch.cat([x[0].T.contiguous() for x in ap]), torch.cat([x[1].T.contiguous() for x in ap]))
                b = (torch.cat([x[0].T.contiguous() for x in bp]), torch.cat([x[1].T.contiguous() for x in bp]))
                fn = deep_gemm.k_grouped_fp8_gemm_tn_contiguous
            else:
                a = (torch.cat([x[0].flatten() for x in ap]), torch.cat([x[1] for x in ap], dim=1))
                b = (torch.cat([x[0].flatten() for x in bp]), torch.cat([x[1] for x in bp], dim=1))
                fn = deep_gemm.k_grouped_fp8_gemm_nt_contiguous
            d = torch.full((2, 128, 128), 3, dtype=torch.float32, device='cuda')
            fn(a, b, d, ks, layout, d, recipe=(1, 1, 128))
            for i, k in enumerate(ks):
                assert torch.all(d[i] == k * (i + 1) + 3)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_asymmetric_scale_recipe_swap() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        deep_gemm.set_num_sms(2)
        for accumulate in (False, True):
            a, b = quant((16, 128), gran_k=32), quant((128, 128))
            d = torch.full((16, 128), 3, dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=d if accumulate else None,
                                      recipe_a=(1, 32), recipe_b=(1, 128))
            assert torch.all(d == (131 if accumulate else 128))
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_scale_dtype_validation() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    quant = _constant_fp8_fp4_with_unit_scales
    try:
        deep_gemm.set_num_sms(2)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        for form in ('dense', 'grouped', 'masked'):
            for int_a, int_b in ((False, False), (True, False), (False, True), (True, True)):
                for disable in (False, True):
                    a = quant((1, 128, 128) if form == 'masked' else (128, 128))
                    b = quant((128, 128) if form == 'dense' else (1, 128, 128))
                    if int_a:
                        a = (a[0], deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(a[1]))
                    if int_b:
                        b = (b[0], deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(b[1]))
                    d = torch.full(a[0].shape, -7, dtype=torch.bfloat16, device='cuda')
                    kwargs = dict(recipe=(1, 1, 128), disable_ue8m0_cast=disable)
                    rejected = False
                    try:
                        if form == 'dense':
                            deep_gemm.fp8_fp4_gemm_nt(a, b, d, **kwargs)
                        elif form == 'grouped':
                            labels = torch.zeros(128, dtype=torch.int32, device='cuda')
                            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, labels, **kwargs)
                        else:
                            masks = torch.tensor([128], dtype=torch.int32, device='cuda')
                            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masks, 128, **kwargs)
                    except RuntimeError:
                        rejected = True
                    assert rejected == (disable and not (int_a and int_b)), (form, int_a, int_b, disable)
                    assert torch.all(d == (-7 if rejected else 128))
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_masked_physical_capacity() -> None:
    old_sms, old_pdl = deep_gemm.get_num_sms(), deep_gemm.get_pdl()
    try:
        for groups in (1, 2, 3):
            for capacity, valid in ((68, 68), (128, 68), (127, 127), (128, 128), (129, 129), (257, 257)):
                for zero_group in (False, True):
                    masks = [0 if zero_group and g == 0 else valid for g in range(groups)]
                    a_data = torch.stack([torch.full((capacity, 128), g + 1, dtype=torch.float32, device='cuda') for g in range(groups)]).to(torch.float8_e4m3fn)
                    b_data = torch.stack([torch.full((64, 128), 2 * g + 1, dtype=torch.float32, device='cuda') for g in range(groups)]).to(torch.float8_e4m3fn)
                    a = (a_data, torch.ones((groups, capacity, 1), device='cuda'))
                    b = (b_data, torch.ones((groups, 1, 1), device='cuda'))
                    storage = torch.full((groups * capacity * 64 + 128,), -7, dtype=torch.bfloat16, device='cuda')
                    d = storage[64:-64].view(groups, capacity, 64)
                    mask = torch.tensor(masks, dtype=torch.int32, device='cuda')
                    deep_gemm.set_num_sms(2 if capacity > 128 else 8)
                    for pdl in (False, True):
                        deep_gemm.set_pdl(pdl)
                        def call():
                            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, mask, valid, recipe=(1, 128, 128))
                        call()
                        for _ in range(3):
                            call()
                            torch.cuda.synchronize()
                            for g, count in enumerate(masks):
                                assert torch.all(d[g, :count] == 128 * (g + 1) * (2 * g + 1)), (groups, capacity, masks, pdl, g)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            call()
                        graph.replay()
                        torch.cuda.synchronize()
                        for g, count in enumerate(masks):
                            assert torch.all(d[g, :count] == 128 * (g + 1) * (2 * g + 1)), (groups, capacity, masks, pdl, g)
                        assert torch.all(storage[:64] == -7) and torch.all(storage[-64:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_pdl(old_pdl)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_kgroup_zero_and_unequal_k() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_num_sms(2)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        for ks in ([0, 128, 256], [128, 0, 256], [128, 256, 0], [0, 0, 0], [128, 384, 256]):
            m, n = 256, 256
            layout = torch.tensor(ks, dtype=torch.int32, device='cuda')
            ap = [torch.full((m, k), g + 1, device='cuda').to(torch.float8_e4m3fn) for g, k in enumerate(ks)]
            bp = [torch.full((n, k), 2 * g + 1, device='cuda').to(torch.float8_e4m3fn) for g, k in enumerate(ks)]
            sa = torch.ones((sum(ks) // 128, m), device='cuda')
            sb = torch.ones((sum(ks) // 128, n), device='cuda')
            for transposed in (False, True):
                storage = torch.full((len(ks) * m * n + 32,), -7, dtype=torch.float32, device='cuda')
                d = storage[16:-16].view(len(ks), m, n)
                d.fill_(3)
                if transposed:
                    a, b = (torch.cat([x.T.contiguous() for x in ap]), sa), (torch.cat([x.T.contiguous() for x in bp]), sb)
                    fn = deep_gemm.k_grouped_fp8_gemm_tn_contiguous
                else:
                    a, b = (torch.cat([x.flatten() for x in ap]), sa.T.contiguous()), (torch.cat([x.flatten() for x in bp]), sb.T.contiguous())
                    fn = deep_gemm.k_grouped_fp8_gemm_nt_contiguous
                fn(a, b, d, ks, layout, d, recipe=(1, 1, 128))
                torch.cuda.synchronize()
                for g, k in enumerate(ks):
                    assert torch.all(d[g] == k * (g + 1) * (2 * g + 1) + 3), (ks, transposed, g)
                assert torch.all(storage[:16] == -7) and torch.all(storage[-16:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_kgroup_unaligned_m_tail_tile_isolation() -> None:
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_num_sms(2)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        # m=48 is not a multiple of BLOCK_M=64: a group's partial tail M tile must
        # stay inside its own slab (the D descriptor is one flat 2D map over all
        # groups, so TMA cannot clamp at group boundaries). The asymmetric ks make
        # group 0's tile finish last, so a tail-tile spill would overwrite group 1's
        # already-stored head rows.
        m, n, ks = 48, 128, [8192, 128]
        layout = torch.tensor(ks, dtype=torch.int32, device='cuda')
        ap = [torch.full((m, k), g + 1, device='cuda').to(torch.float8_e4m3fn) for g, k in enumerate(ks)]
        bp = [torch.full((n, k), 2 * g + 1, device='cuda').to(torch.float8_e4m3fn) for g, k in enumerate(ks)]
        sa = torch.ones((sum(ks) // 128, m), device='cuda')
        sb = torch.ones((sum(ks) // 128, n), device='cuda')
        for transposed in (False, True):
            storage = torch.full((len(ks) * m * n + 32,), -7, dtype=torch.float32, device='cuda')
            d = storage[16:-16].view(len(ks), m, n)
            d.fill_(3)
            if transposed:
                a, b = (torch.cat([x.T.contiguous() for x in ap]), sa), (torch.cat([x.T.contiguous() for x in bp]), sb)
                fn = deep_gemm.k_grouped_fp8_gemm_tn_contiguous
            else:
                a, b = (torch.cat([x.flatten() for x in ap]), sa.T.contiguous()), (torch.cat([x.flatten() for x in bp]), sb.T.contiguous())
                fn = deep_gemm.k_grouped_fp8_gemm_nt_contiguous
            fn(a, b, d, ks, layout, d, recipe=(1, 1, 128))
            torch.cuda.synchronize()
            for g, k in enumerate(ks):
                assert torch.all(d[g] == k * (g + 1) * (2 * g + 1) + 3), (ks, transposed, g)
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
                use_ue8m0=True, gran_k=32, k_alignment=128, use_psum_layout=False,
                accumulate=True, out_dtype=torch.float32)
            aa = a[0].cpu().double() * a[1].cpu().double().repeat_interleave(32, 0)
            bb = b[0].cpu().double() * b[1].cpu().double().repeat_interleave(32, 0)
            cc = initial.cpu().double()
            refs, start = [], 0
            for g, k in enumerate(host_ks):
                refs.append((aa[start:start+k].T @ bb[start:start+k] + cc[g]).float())
                start += k
            ref = torch.stack(refs)
            for separate in (False, True):
                first = None
                for _ in range(4):
                    out = initial.clone()
                    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                        a, b, out, host_ks, layout, initial if separate else out, recipe=(1, 1, 32))
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
    test_sm120_split_k_accumulation()
    test_sm120_mixed_fp8_fp4_scale_tile_k_tail()
    test_sm120_dense_strided_output_and_accumulation()
    test_sm120_odd_n_bf16_output_and_accumulation()
    test_sm120_batched_strided_output_and_accumulation()
    test_sm120_contiguous_grouped_output_row_stride()
    test_sm120_kgroup_nt_tn_layouts_and_accumulation()
    test_sm120_asymmetric_scale_recipe_swap()
    test_sm120_scale_dtype_validation()
    test_sm120_masked_physical_capacity()
