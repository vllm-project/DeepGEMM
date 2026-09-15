import random
import torch

import deep_gemm
from deep_gemm.utils import align, pack_ue8m0_to_int
from deep_gemm.testing import (
    assert_bitwise_equal,
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major
)
from utils import (
    assert_direct_output_matches_fp32_accumulation,
    assert_psum_zero_padding, convert_to_fp8, make_cublas_gemm,
)

from generators import (
    KernelType, MajorTypeAB, QuantConfig, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,
)


def test_gemm() -> None:
    print('Testing GEMM:')
    use_alpha_options = (False, True) if get_arch_major() == 10 else (False,)
    for kernel_type, quant_config, m, n, k, major_a, major_b, accumulate, out_dtype, scores in \
            enumerate_normal(torch.float8_e4m3fn, collect_cublas_scores=True):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=(kernel_type.is_1d1d() and accumulate))

        # SM120 mixed FP8xFP4 is K-major only and the 16U4_ALIGN16B TMA constraint makes
        # `k % 128 == 0` mandatory -- `DG_HOST_ASSERT(!is_mixed_fp4 or k % 128 == 0)` in
        # csrc/apis/sm120_dispatch.hpp. Skip the shapes the kernel cannot take.
        is_mixed_fp4 = quant_config.is_fp4_a != quant_config.is_fp4_b
        if is_mixed_fp4 and get_arch_major() == 12 and k % 128 != 0:
            continue

        for test_alias in (False, True):
            for use_alpha in use_alpha_options:
                alpha = random.uniform(-1.0, 1.0) if use_alpha else None
                a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                     kernel_type, use_ue8m0=use_ue8m0,
                                                     quant_config=quant_config, alpha=alpha)
                func_name = f'fp8_fp4_gemm_{major_opt.lower() if test_alias else "nt"}'
                if test_alias:
                    a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                    b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                    assert a[0].is_contiguous() and b[0].is_contiguous()
                getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast,
                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b, alpha=alpha)
                diff = calc_diff(d, ref_d)
                assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, '
                                                        f'{accumulate=}, {out_dtype=}, {use_alpha=}, {alpha=}, '
                                                        f'{diff:.5f}, alias={test_alias}')

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0, quant_config=quant_config)
        initial_d = d.clone()
        def test_func(a_=a, b_=b, d_=d):
            deep_gemm.fp8_fp4_gemm_nt(a_, b_, d_, c=d_ if accumulate else None,
                                       disable_ue8m0_cast=disable_ue8m0_cast,
                                       recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
        equivalent_d = d.clone()
        test_func(d_=equivalent_d)
        # Bitwise deterministic test
        for _ in range(20):
            d.copy_(initial_d)
            test_func()
            assert torch.equal(d, equivalent_d), f'{m=}, {n=}, {k=}, {accumulate=}'
        if quant_config.is_fp4_a or quant_config.is_fp4_b:
            equivalent_fp8_d = initial_d.clone()
            test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
            # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
            assert calc_diff(equivalent_d, equivalent_fp8_d) < 1e-14, (f'FP4/FP8 mismatch: {m=}, {n=}, {k=}, '
                                                                      f'{accumulate=}')
        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        cublas_func = make_cublas_gemm(a, b, d, c)
        cublas_times = bench_kineto(cublas_func, ('nvjet', 'bstensorop', 'reduce'),
                                    suppress_kineto_output=True, with_multiple_kernels=True)
        cublas_t = sum(cublas_times)
        if cublas_t > 0:
            scores.append(cublas_t / t)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{cublas_t / t:.2f}x cuBLAS speedup')


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for kernel_type, quant_config, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout, ensure_zero_padding in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for test_alias in (False, True):
            m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                                          use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                                          quant_config=quant_config)
            func_name = f"m_grouped_fp8_fp4_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            def test_func(a_=a, b_=b, d_=d):
                getattr(deep_gemm, func_name)(a_, b_, d_, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast,
                                              use_psum_layout=use_psum_layout, ensure_zero_padding=ensure_zero_padding,
                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            equivalent_d = d.clone()
            test_func(d_=equivalent_d)
            # Bitwise deterministic test
            for _ in range(20):
                test_func()
                assert torch.equal(d[valid_mask], equivalent_d[valid_mask]), f'{m=}, {n=}, {k=}, alias={test_alias}'
            if quant_config.is_fp4_a or quant_config.is_fp4_b:
                equivalent_fp8_d = d.clone()
                test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
                # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
                assert calc_diff(equivalent_d[valid_mask], equivalent_fp8_d[valid_mask]) < 1e-14, (
                    f'FP4/FP8 mismatch: {m=}, {n=}, {k=}, alias={test_alias}')
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, '
                                                    f'{diff:.5f}, alias={test_alias}, {ensure_zero_padding=}')
            if use_psum_layout and ensure_zero_padding:
                assert_psum_zero_padding(a, d, grouped_layout, 'FP8/FP4')
        m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                          use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                          quant_config=quant_config)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast, use_psum_layout=use_psum_layout,
                                                           ensure_zero_padding=ensure_zero_padding,
                                                           recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}, '
              f'psum={use_psum_layout}, zero_pad={ensure_zero_padding}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0
        expected_m = int(expected_m_per_group * 1.2)

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for i in range(num_tests):
            a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k,
                                                                                   use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                                   quant_config=quant_config)
            def test_func(a_=a, b_=b, d_=d):
                common = dict(disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                if use_psum_layout:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a_, b_, d_, grouped_layout,
                                                                   use_psum_layout=True, expected_m_for_psum_layout=expected_m, **common)
                else:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a_, b_, d_, grouped_layout, expected_m, **common)

            equivalent_d = d.clone()
            test_func(d_=equivalent_d)
            # Bitwise deterministic test
            for _ in range(20):
                test_func()
                assert torch.equal(d[valid_mask], equivalent_d[valid_mask]), f'{max_m=}, {n=}, {k=}'
            if quant_config.is_fp4_a or quant_config.is_fp4_b:
                equivalent_fp8_d = d.clone()
                test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
                # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
                assert calc_diff(equivalent_d[valid_mask], equivalent_fp8_d[valid_mask]) < 1e-14, (
                    f'FP4/FP8 mismatch: {max_m=}, {n=}, {k=}, {num_groups=}')
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < quant_config.max_diff(), f'{max_m=}, {n=}, {k=}, {kernel_opt}, {num_groups=}, {diff:.5f}'

            # Test performance with fixed shapes
            valid_m = int(valid_mask.sum().item())
            t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'{kernel_opt}, psum={1 if use_psum_layout else 0}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped GEMM:')

    arch_major = get_arch_major()

    # The K-major/MN-major choice is the entry point: K-major is NT, MN-major is TN.
    # SM90 FP8 is K-major, SM120 FP8 yields both, everything else is MN-major -- so select
    # per case rather than per arch (see `enumerate_k_grouped_contiguous`).
    def select_fp8_gemm(major_a: MajorTypeAB):
        return (deep_gemm.k_grouped_fp8_gemm_nt_contiguous if major_a.is_k_major()
                else deep_gemm.k_grouped_fp8_gemm_tn_contiguous)

    test_options = [(torch.float8_e4m3fn, QuantConfig(), select_fp8_gemm)]
    if arch_major == 10:
        test_options.append((torch.float4_e2m1fn_x2, QuantConfig((32, 32, True, True)),
                             lambda major_a: deep_gemm.k_grouped_fp4_gemm_nt_contiguous))
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    for dtype, quant_config, select_gemm in test_options:
        is_fp4 = dtype == torch.float4_e2m1fn_x2
        dtype_opt = 'FP4' if is_fp4 else 'FP8'

        for num_groups, m, n, major_a, major_b, real_ks_cpu, _, _, gran_k, k_alignment, use_psum_layout, accumulate, out_dtype in \
                enumerate_k_grouped_contiguous(dtype):
            recipe = (1, 1, gran_k)
            gemm = select_gemm(major_a)

            for test_real_ks_cpu in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu):
                total_k, a, b, c, d, ref_d, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                    num_groups, m, n, major_a, major_b, test_real_ks_cpu,
                    use_ue8m0=use_ue8m0, gran_k=gran_k,
                    quant_config=quant_config if is_fp4 else None,
                    use_psum_layout=use_psum_layout, k_alignment=k_alignment,
                    accumulate=accumulate, out_dtype=out_dtype)

                initial_d = d.clone()
                if not accumulate:
                    initial_d.fill_(float('nan'))
                equivalent_d = initial_d.clone()
                gemm(a, b, equivalent_d, host_ks_cpu, grouped_layout, equivalent_d if accumulate else None,
                     recipe=recipe, use_psum_layout=use_psum_layout)
                if is_fp4 and accumulate:
                    fp8_a, fp8_b = convert_to_fp8(a), convert_to_fp8(b)
                    fp8_a = (fp8_a[0].T.contiguous(), fp8_a[1])
                    fp8_b = (fp8_b[0].T.contiguous(), fp8_b[1])
                    equivalent_fp8_d = initial_d.clone()
                    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                        fp8_a, fp8_b, equivalent_fp8_d, host_ks_cpu, grouped_layout, equivalent_fp8_d if accumulate else None,
                        recipe=recipe, use_psum_layout=use_psum_layout)
                    mismatch_message = (f'FP4/FP8 mismatch: {m=}, {n=}, {total_k=}, '
                                        f'{test_real_ks_cpu=}, {use_psum_layout=}')
                    assert calc_diff(equivalent_d, equivalent_fp8_d) < 1e-14, mismatch_message

                # Bitwise deterministic test
                host_ks_options = (host_ks_cpu, None, []) if use_psum_layout else (host_ks_cpu, )
                for test_host_ks_cpu in host_ks_options:
                    for stress_idx in range(20):
                        d.copy_(initial_d)
                        gemm(a, b, d, test_host_ks_cpu, grouped_layout, c,
                             recipe=recipe, use_psum_layout=use_psum_layout)
                        assert_bitwise_equal(
                            d, equivalent_d,
                            f'k-grouped self-consistency at {stress_idx=}, {dtype_opt}, {m=}, {n=}, {total_k=}, '
                            f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}'
                        )
                    if not accumulate:
                        case_label = (f'{dtype_opt} K-grouped direct output, {m=}, {n=}, {total_k=}, '
                                      f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, '
                                      f'{out_dtype=}')
                        assert_direct_output_matches_fp32_accumulation(
                            d,
                            lambda output, accumulator: gemm(
                                a, b, output, test_host_ks_cpu, grouped_layout, accumulator,
                                recipe=recipe, use_psum_layout=use_psum_layout),
                            case_label)

                if accumulate:
                    diff = calc_diff(d, ref_d)
                    assert diff < quant_config.max_diff(), (
                        f'{dtype_opt}, {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                        f'{host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}, {diff:.5f}')

                # gran_k=128 requires FP32 SF input; only gran_k=32 accepts
                # the per-group packed INT32 UE8M0 layout.
                if gran_k == 32:
                    sf_ks = [k // gran_k for k in host_ks_cpu]
                    ref_packed_a = torch.cat([
                        pack_ue8m0_to_int(torch.nn.functional.pad(
                            group_sf.T, (0, align(sf_k, 4) - sf_k)).contiguous()).T
                        for group_sf, sf_k in zip(a[1].split(sf_ks), sf_ks) if sf_k > 0
                    ])
                    ref_packed_b = torch.cat([
                        pack_ue8m0_to_int(torch.nn.functional.pad(
                            group_sf.T, (0, align(sf_k, 4) - sf_k)).contiguous()).T
                        for group_sf, sf_k in zip(b[1].split(sf_ks), sf_ks) if sf_k > 0
                    ])
                    packed_a = (
                        a[0], deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                            a[1], grouped_layout, host_ks_cpu, gran_k, k_alignment, use_psum_layout))
                    packed_b = (
                        b[0], deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                            b[1], grouped_layout, host_ks_cpu, gran_k, k_alignment, use_psum_layout))
                    assert torch.equal(packed_a[1], ref_packed_a)
                    assert torch.equal(packed_b[1], ref_packed_b)

                    packed_d = initial_d.clone()
                    gemm(packed_a, packed_b, packed_d, host_ks_cpu, grouped_layout, packed_d if accumulate else None,
                         recipe=recipe, use_psum_layout=use_psum_layout)
                    if accumulate:
                        packed_diff = calc_diff(packed_d, ref_d)
                        assert packed_diff < quant_config.max_diff(), (
                            f'pre-packed INT32 SF: {dtype_opt}, {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                            f'{host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}, '
                            f'{packed_diff:.5f}')
                    else:
                        case_label = (f'pre-packed INT32 SF direct output: {dtype_opt}, {m=}, {n=}, '
                                      f'{total_k=}, {test_real_ks_cpu=}, {host_ks_cpu=}, '
                                      f'{use_psum_layout=}, {out_dtype=}')
                        assert_direct_output_matches_fp32_accumulation(
                            packed_d,
                            lambda output, accumulator: gemm(
                                packed_a, packed_b, output, host_ks_cpu, grouped_layout, accumulator,
                                recipe=recipe, use_psum_layout=use_psum_layout),
                            case_label)

            _, a, b, c, d, _, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, real_ks_cpu,
                use_ue8m0=use_ue8m0, gran_k=gran_k,
                quant_config=quant_config if is_fp4 else None,
                use_psum_layout=use_psum_layout, k_alignment=k_alignment,
                accumulate=accumulate, out_dtype=out_dtype)

            # noinspection PyShadowingNames
            def test_func():
                gemm(a, b, d, host_ks_cpu, grouped_layout, c,
                     recipe=recipe, use_psum_layout=use_psum_layout)

            t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
            logical_k = sum(real_ks_cpu)
            out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
            print(f' > Perf ({dtype_opt}, {num_groups=:2}, m={m:5}, n={n:5}, k={logical_k:5}, gran_k={gran_k:3}, '
                  f'k_alignment={k_alignment:3}, psum={int(use_psum_layout)}, acc={int(accumulate)}, {out_opt}): '
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * m * n * logical_k / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    print()


def _constant_fp8_fp4_with_unit_scales(shape, value=1, gran_k=128, fp4=False):
    data = torch.full((*shape[:-1], shape[-1] // 2), 0x22, dtype=torch.int8, device='cuda') if fp4 else \
        torch.full(shape, value, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    sf = torch.ones((*shape[:-1], (shape[-1] + gran_k - 1) // gran_k), dtype=torch.float32, device='cuda')
    return data, sf


def test_sm120_split_k_accumulation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_mixed_fp8_fp4_scale_tile_k_tail() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_dense_strided_output_and_accumulation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_odd_n_bf16_output_and_accumulation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_batched_strided_output_and_accumulation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_contiguous_grouped_output_row_stride() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_kgroup_nt_tn_layouts_and_accumulation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_asymmetric_scale_recipe_swap() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_scale_dtype_validation() -> None:
    if get_arch_major() != 12:
        return
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


def test_sm120_masked_physical_capacity() -> None:
    if get_arch_major() != 12:
        return
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
    test_sm120_kgroup_descriptor_reuse_at_default_sms()
    test_sm120_kgroup_zero_and_unequal_k()
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
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
    test_k_grouped_gemm_contiguous()
