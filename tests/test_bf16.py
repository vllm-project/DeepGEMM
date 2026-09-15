import numpy as np
import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes
)
from utils import (
    assert_direct_output_matches_fp32_accumulation,
    assert_psum_zero_padding,
)
from generators import (
    MajorTypeAB, get_arch_major,
    enumerate_normal, enumerate_batched_syrk_symm, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,
)


def test_gemm() -> None:
    print('Testing GEMM:')
    scores = []
    use_alpha_options = (False, True) if get_arch_major() == 10 else (False,)
    for kernel_type, _, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.bfloat16):
        deep_gemm.use_deterministic_algorithms(True)
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        for test_alias in (False, True):
            for use_alpha in use_alpha_options:
                alpha = random.uniform(-1.0, 1.0) if use_alpha else None
                a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                     kernel_type, use_bf16=True, alpha=alpha)
                func_name = f'bf16_gemm_{major_opt.lower() if test_alias else "nt"}'
                if test_alias:
                    a = a if major_a.is_k_major() else a.T
                    b = b if major_b.is_k_major() else b.T
                    assert a.is_contiguous() and b.is_contiguous()
                getattr(deep_gemm, func_name)(a, b, d, c=c, alpha=alpha)
                diff = calc_diff(d, ref_d)
                assert diff < 1e-5, (f'{m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                       f'{use_alpha=}, {alpha=}, {diff:.5f}, alias={test_alias}')
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)

        t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), 'bf16_gemm', suppress_kineto_output=True)
        deep_gemm.use_deterministic_algorithms(False)
        cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:7.1f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{(cublas_t + split_k_t) / t:.2f}x cuBLAS')
        if cublas_t > 0:
            scores.append((cublas_t + split_k_t) / t)
    print(f"Average speedup over cuBLASLt: {float(np.prod(scores)) ** (1.0 / len(scores)):.3f}x\n")


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for _, _, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout, ensure_zero_padding in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for test_alias in (False, True):
            m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                                          use_bf16=True, use_psum_layout=use_psum_layout)
            func_name = f"m_grouped_bf16_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else b.mT
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, grouped_layout, use_psum_layout=use_psum_layout,
                                          ensure_zero_padding=ensure_zero_padding)
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < 1e-5, f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}, alias={test_alias}, {ensure_zero_padding=}'
            if use_psum_layout and ensure_zero_padding:
                assert_psum_zero_padding(a, d, grouped_layout, 'BF16')
        m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                                      use_bf16=True, use_psum_layout=use_psum_layout)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, use_psum_layout=use_psum_layout,
                                                        ensure_zero_padding=ensure_zero_padding)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}, '
              f'psum={use_psum_layout}, zero_pad={ensure_zero_padding}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for _, _, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.bfloat16):
        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m_per_group * 1.2))
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for i in range(num_tests):
            a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k,
                use_bf16=True, use_psum_layout=use_psum_layout)

            def test_func():
                if use_psum_layout:
                    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout,
                                                                use_psum_layout=True, expected_m_for_psum_layout=expected_m_per_group)
                else:
                    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, grouped_layout, expected_m_per_group)

            test_func()
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < 1e-5, f'{max_m=}, {n=}, {k=}, {num_groups=}, {diff:.5f}'


            # Test performance with fixed shapes
            valid_m = int(valid_mask.sum().item())
            t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * (valid_m / (max_m * num_groups)) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'psum={1 if use_psum_layout else 0}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    for num_groups, m, n, major_a, major_b, real_ks_cpu, _, _, _, alignment, use_psum_layout, accumulate, out_dtype in \
            enumerate_k_grouped_contiguous(torch.bfloat16):
        for test_real_ks_cpu in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu):
            total_k, a, b, c, d, ref_d, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, test_real_ks_cpu, use_bf16=True,
                use_psum_layout=use_psum_layout, k_alignment=alignment,
                accumulate=accumulate, out_dtype=out_dtype)
            initial_d = d.clone()
            if not accumulate:
                initial_d.fill_(float('nan'))
            host_ks_options = (host_ks_cpu, None, []) if use_psum_layout else (host_ks_cpu, )
            for test_host_ks_cpu in host_ks_options:
                d.copy_(initial_d)
                deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                    a, b, d, test_host_ks_cpu, grouped_layout, c, use_psum_layout=use_psum_layout)
                if accumulate:
                    diff = calc_diff(d, ref_d)
                    assert diff < 1e-5, (f'{m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                                        f'{test_host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, '
                                        f'{out_dtype=}, {diff:.7f}')
                else:
                    case_label = (f'BF16 K-grouped direct output, {m=}, {n=}, {total_k=}, '
                                  f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, '
                                  f'{out_dtype=}')
                    assert_direct_output_matches_fp32_accumulation(
                        d,
                        lambda output, accumulator: deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                            a, b, output, test_host_ks_cpu, grouped_layout, accumulator,
                            use_psum_layout=use_psum_layout),
                        case_label)

        # Test performance
        _, a, b, c, d, _, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
            num_groups, m, n, major_a, major_b, real_ks_cpu, use_bf16=True,
            use_psum_layout=use_psum_layout, k_alignment=alignment,
            accumulate=accumulate, out_dtype=out_dtype)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, host_ks_cpu, grouped_layout, c, use_psum_layout=use_psum_layout)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        logical_k = sum(real_ks_cpu)
        out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={logical_k:5}, align={alignment:3}, '
              f'psum={int(use_psum_layout)}, acc={int(accumulate)}, {out_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * logical_k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')

    print()


def test_cublaslt_gemm() -> None:
    print('Testing cuBLASLt GEMM:')
    use_alpha_options = (False, True)
    for kernel_type, _, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(dtype=torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        # BF16 accumulation has lower precision than cuBLASLt's FP32 accumulation
        threshold = 1e-5 if (accumulate and out_dtype == torch.bfloat16) else 6e-7
        for use_alpha in use_alpha_options:
            alpha = random.uniform(-1.0, 1.0) if use_alpha else None
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                 kernel_type, use_bf16=True, alpha=alpha)
            deep_gemm.use_deterministic_algorithms(False)
            deep_gemm.bf16_gemm_nt(a, b, d, c=c, alpha=alpha)
            diff = calc_diff(d, ref_d)
            assert diff < threshold, (f'{diff=}, {use_alpha=}, {alpha=}, '
                                      f'({m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=})')

        t_nvjet, t_gemv, t_gemm = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
        t = t_nvjet + t_gemv + t_gemm
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()


def test_cublaslt_batched_syrk() -> None:
    print('Testing cuBLASLt batched SYRK:')
    for num_batches, m, k, out_dtype in enumerate_batched_syrk_symm():
        out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
        rows, cols = min(m, k), max(m, k)
        threshold = 6e-7
        for k_major in (True, False):
            for batch_shape, padding in (((), (3, 5)), ((num_batches,), (3, 5)), ((num_batches,), (0, 0))):
                shape = (rows, cols) if k_major else (cols, rows)
                a = torch.randn(*batch_shape, shape[0] + padding[0], shape[1] + padding[1],
                                device='cuda', dtype=out_dtype)[..., :shape[0], :shape[1]]
                if not k_major:
                    a = a.mT
                major_opt = 'N' if a.stride(-1) == 1 else 'T'

                d = torch.empty(*batch_shape, rows + padding[0], rows + padding[1],
                                device='cuda', dtype=out_dtype)[..., :rows, :rows]
                deep_gemm.batched_syrk(a, d)
                ref_d = (a.float() @ a.float().mT).to(out_dtype)
                diff = calc_diff(d, ref_d)
                assert diff < threshold, (
                    f'{diff=}, ({batch_shape=}, {padding=}, {rows=}, {cols=}, {major_opt=}, {out_dtype=})'
                )

            t_nvjet, t_gemv, t_gemm = bench_kineto(
                lambda: deep_gemm.batched_syrk(a, d),
                ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
            t = t_nvjet + t_gemv + t_gemm
            print(f' > Perf (batch={num_batches}, m={rows:6}, n={rows:6}, k={cols:6}, '
                  f'layout={major_opt}, {out_opt}): '
                  f'{t * 1e6:5.0f} us | '
                  f'{2 * num_batches * rows * rows * cols / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_cublaslt_batched_symm() -> None:
    print('Testing cuBLASLt batched SYMM:')
    for num_batches, m, k, out_dtype in enumerate_batched_syrk_symm():
        out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
        rows, cols = min(m, k), max(m, k)
        threshold = 6e-7
        for k_major_a, k_major_b in ((True, True), (True, False), (False, True), (False, False)):
            for batch_shape, padding in (((), (3, 5)), ((num_batches,), (3, 5)), ((num_batches,), (0, 0))):
                a = torch.randn(*batch_shape, rows + padding[0], rows + padding[1],
                                device='cuda', dtype=out_dtype)[..., :rows, :rows]
                if not k_major_a:
                    a = a.mT
                a.copy_(a + a.mT)

                shape_b = (rows, cols) if k_major_b else (cols, rows)
                b = torch.randn(*batch_shape, shape_b[0] + padding[0], shape_b[1] + padding[1],
                                device='cuda', dtype=out_dtype)[..., :shape_b[0], :shape_b[1]]
                if not k_major_b:
                    b = b.mT
                major_opt  = 'N' if a.stride(-1) == 1 else 'T'
                major_opt += 'N' if b.stride(-1) == 1 else 'T'

                d = torch.empty(*batch_shape, rows + padding[0], cols + padding[1],
                                device='cuda', dtype=out_dtype)[..., :rows, :cols]
                deep_gemm.batched_symm(a, b, d)
                ref_d = (a.float() @ b.float()).to(out_dtype)
                diff = calc_diff(d, ref_d)
                assert diff < threshold, (
                    f'{diff=}, ({batch_shape=}, {padding=}, {rows=}, {cols=}, {major_opt=}, {out_dtype=})'
                )

            t_nvjet, t_gemv, t_gemm = bench_kineto(
                lambda: deep_gemm.batched_symm(a, b, d),
                ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
            t = t_nvjet + t_gemv + t_gemm
            print(f' > Perf (batch={num_batches}, m={rows:6}, n={cols:6}, k={rows:6}, '
                  f'layout={major_opt}, {out_opt}): '
                  f'{t * 1e6:5.0f} us | '
                  f'{2 * num_batches * rows * rows * cols / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_sm120_small_n_output_row_stride_and_accumulation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_contiguous_grouped_output_row_stride() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_kgroup_unequal_k_accumulation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_masked_physical_capacity() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_kgroup_zero_and_unequal_k() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_kgroup_descriptor_reuse_at_default_sms() -> None:
    if get_arch_major() != 12:
        return
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
    test_sm120_kgroup_descriptor_reuse_at_default_sms()
    test_sm120_kgroup_zero_and_unequal_k()
    test_sm120_small_n_output_row_stride_and_accumulation()
    test_sm120_contiguous_grouped_output_row_stride()
    test_sm120_kgroup_unequal_k_accumulation()
    test_sm120_masked_physical_capacity()
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    if get_arch_major() >= 9:
        test_gemm()
        test_m_grouped_gemm_contiguous()
        test_m_grouped_gemm_masked()
        test_k_grouped_gemm_contiguous()

    test_cublaslt_gemm()
    test_cublaslt_batched_syrk()
    test_cublaslt_batched_symm()
