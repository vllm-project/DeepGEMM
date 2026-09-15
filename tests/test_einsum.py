import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major, test_filter
)
from deep_gemm.utils.math import (
    align, cast_back_from_fp8, ceil_div,
    per_block_cast_to_fp8, per_channel_cast_to_fp8, per_token_cast_to_fp8
)


def assert_bf16_einsum_close(z, ref_z, fp32_ref_fn) -> None:
    # For BF16-in / BF16-out einsums only. SM120: DeepGEMM and the reference differ only in
    # FP32 accumulation order -- both equally accurate against an FP64 reduction -- which puts
    # the BF16 comparison a few ULPs past 1e-10. Relax the like-for-like check on arch 12 and
    # pin correctness to FP32 truth instead. `fp32_ref_fn` is a thunk so the extra FP32 einsum
    # is never run on other arches.
    #
    # Ported from nv_dev, which applies it at exactly these two call sites. It is NOT used
    # where both sides are already FP32 -- see `test_bhd_bhr_hdr`.
    if get_arch_major() != 12:
        assert calc_diff(z, ref_z) < 1e-10
        return
    assert calc_diff(z, ref_z) < 1e-7
    assert calc_diff(z, fp32_ref_fn()) < 1e-5


def test_bmk_bnk_mn() -> None:
    print('Testing "bmk, bnk -> mn":')
    for s in (129, 4096, 8192):
        for m, n, k in [(128, 384, 128), (256, 256, 256), (384, 128, 384)]:
            for dtype in (torch.float, torch.bfloat16):
                a = torch.randn((s, m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((s, n, k), dtype=torch.bfloat16, device='cuda')
                d = torch.randn((m, n), dtype=dtype, device='cuda')
                c = d if dtype == torch.float else None

                # Test correctness
                ref_d = (c if dtype == torch.float else 0) + torch.bmm(a.float(), b.float().mT).sum(0)
                deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)
                assert calc_diff(d, ref_d) < 1e-5

                t = bench_kineto(lambda: deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c), 'bmn_bnk_mn_gemm_impl', suppress_kineto_output=True)
                print(f' > Perf (b={s:4.0f}, {m=}, {n=}, {k=}, {"FP32" if dtype == torch.float else "BF16"}): ',
                    f'{t * 1e6:4.0f} us | '
                    f'{2 * s * m * n * k / t / 1e12:4.0f} TFLOPS | '
                    f'{(count_bytes(a, b) + (d.numel() * 4)) / 1e9 / t:4.0f} GB/s')
    print()


def test_bhr_hdr_bhd():
    print('Testing "bhr, hdr -> bhd":')
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)

            times = []
            for use_cublaslt in (False, True):
                deep_gemm.use_deterministic_algorithms(not use_cublaslt)
                z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
                deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
                assert_bf16_einsum_close(z, ref_z,
                                         lambda: torch.einsum('bhr,hdr->bhd', x.float(), y.float()))
                kernel_name = 'nvjet' if use_cublaslt else 'gemm'
                times.append(bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), kernel_name, suppress_kineto_output=True))
            t, t_cublaslt = times
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


def test_bhd_hdr_bhr():
    print('Testing "bhd, hdr -> bhr":')
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)

            times = []
            for use_cublaslt in (False, True):
                deep_gemm.use_deterministic_algorithms(not use_cublaslt)
                z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)
                deep_gemm.einsum('bhd,hdr->bhr', x, y, z)
                assert_bf16_einsum_close(z, ref_z,
                                         lambda: torch.einsum('bhd,hdr->bhr', x.float(), y.float()))
                kernel_name = 'nvjet' if use_cublaslt else 'gemm'
                times.append(bench_kineto(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z), kernel_name, suppress_kineto_output=True))

            t, t_cublaslt = times
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


def test_bhd_bhr_hdr():
    print('Testing "bhd, bhr -> hdr":')
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            z_0 = torch.randn((h, d, r), device='cuda', dtype=torch.float32) * 10
            ref_z = z_0 + torch.einsum('bhd,bhr->hdr', x.float(), y.float())

            z = z_0.clone()
            deep_gemm.einsum('bhd,bhr->hdr', x, y, z, z)
            # NOT relaxed on arch 12: `z` and `ref_z` are both FP32 here (`ref_z` is
            # `z_0 + einsum(x.float(), y.float())`, the FP32 truth), so the BF16
            # accumulation-order argument behind `assert_bf16_einsum_close` does not apply.
            # nv_dev has no counterpart for this test at all. Leave it strict; if SM120
            # actually needs slack here, that is a finding, not a tolerance to pre-loosen.
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_kineto(lambda: deep_gemm.einsum('bhd,bhr->hdr', x, y, z, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z, z)) / t / 1e9:4.0f} GB/s')
    print()


def test_fp8_bhr_hdr_bhd(use_ue8m0: bool = True):
    print('Testing FP8 "bhr, hdr -> bhd":')
    deep_gemm.use_deterministic_algorithms(False)
    for h, r, d in [(8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)

            x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=use_ue8m0)
            x_fp8 = x_fp8[0].view(b, h, r), x_fp8[1].view(b, h, ceil_div(r, 128))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                     torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float))
            for i in range(h):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)
            z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
            output_dtypes = (torch.bfloat16, torch.float8_e4m3fn) if get_arch_major() == 10 else (torch.bfloat16,)
            for output_dtype in output_dtypes:
                if output_dtype == torch.float8_e4m3fn:
                    output = torch.empty_like(z, dtype=output_dtype)
                    sfd = torch.empty_strided(
                        (b, ceil_div(h * d, 32 * 4)), (1, align(b, 4)), dtype=torch.int32, device='cuda')
                    deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, (output, sfd))
                    restored_output = cast_back_from_fp8(
                        output.view(b, h * d), sfd.contiguous(), gran_k=32, use_packed_ue8m0=True)
                    diff = calc_diff(restored_output, ref_z.reshape(b, h * d))

                    # The fused output must align with the production pipeline (a BF16 D
                    # followed by a per-token cast): the epilogue rounds the accumulator into
                    # BF16 before the same amax/scale/rounding steps, so both the FP8 data
                    # and the packed SF words must match bitwise
                    d_bf16 = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
                    deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, d_bf16)
                    ref_fp8, ref_sfd = per_token_cast_to_fp8(d_bf16.view(b, h * d), use_ue8m0=True,
                                                             gran_k=32, use_packed_ue8m0=True)
                    assert torch.equal(output.view(b, h * d), ref_fp8), f'{b=}, {h=}, {r=}, {d=}'
                    assert torch.equal(sfd.contiguous(), ref_sfd), f'{b=}, {h=}, {r=}, {d=}'
                else:
                    output = z
                    deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, output)
                    diff = calc_diff(output, ref_z)
                assert diff < (2e-3 if output_dtype == torch.float8_e4m3fn else 1e-3), \
                    f'{b=}, {h=}, {r=}, {d=}, {output_dtype=}, {diff=:.5f}'

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, z), 'gemm_', suppress_kineto_output=True)
            t_cublaslt = bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


@test_filter(lambda: get_arch_major() >= 10)
def test_fp8_bhd_hdr_bhr(use_ue8m0: bool = True):
    deep_gemm.use_deterministic_algorithms(False)
    print('Testing FP8 "bhd, hdr -> bhr":')
    for h, r, d in [(8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)

            x_fp8 = per_token_cast_to_fp8(x.view(-1, d), use_ue8m0=use_ue8m0)
            x_fp8 = x_fp8[0].view(b, h, d), x_fp8[1].view(b, h, ceil_div(d, 128))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                     torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float))
            for i in range(h):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)
            z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)

            deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, z)
            assert calc_diff(z, ref_z) < 1e-3

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, z), 'gemm_', suppress_kineto_output=True)
            t_cublaslt = bench_kineto(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


@test_filter(lambda: get_arch_major() >= 10)
def test_fp8_bhd_bhr_hdr(use_ue8m0: bool = True):
    print('Testing FP8 "bhd, bhr -> hdr":')
    for h, r, d in [(8, 4096, 1024)]:
        for b in (4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            z_0 = torch.randn((h, d, r), device='cuda', dtype=torch.float32) * 10
            ref_z = z_0 + torch.einsum('bhd,bhr->hdr', x, y)

            x_fp8 = per_channel_cast_to_fp8(x.view(b, -1), use_ue8m0=use_ue8m0)
            y_fp8 = per_channel_cast_to_fp8(y.view(b, -1), use_ue8m0=use_ue8m0)
            x_fp8 = (x_fp8[0].view(b, h, d), x_fp8[1].view(ceil_div(b, 128), h, d))
            y_fp8 = (y_fp8[0].view(b, h, r), y_fp8[1].view(ceil_div(b, 128), h, r))
            z = z_0.clone()
            deep_gemm.fp8_einsum('bhd,bhr->hdr', x_fp8, y_fp8, z, c=z, recipe=(1, 1, 128))
            assert calc_diff(z, ref_z) < 1e-3

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhd,bhr->hdr', x_fp8, y_fp8, z, c=z, recipe=(1, 1, 128)), 'gemm_', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z, z)) / t / 1e9:4.0f} GB/s')
    print()


def test_sm120_batched_output_row_stride() -> None:
    if get_arch_major() != 12:
        return
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_num_sms(2)
        for expr in ('bhd,hdr->bhr', 'bhr,hdr->bhd'):
            for padding in (0, 8):
                a = torch.ones((128, 2, 128), dtype=torch.bfloat16, device='cuda')
                a[:, 1].fill_(2)
                b_shape = (2, 128, 32) if expr == 'bhd,hdr->bhr' else (2, 32, 128)
                b = torch.ones(b_shape, dtype=torch.bfloat16, device='cuda')
                storage = torch.full((130, 2, 32 + padding), -7, dtype=torch.bfloat16, device='cuda')
                d = storage[1:129, :, :32]
                deep_gemm.einsum(expr, a, b, d)
                assert torch.all(d[:, 0] == 128)
                assert torch.all(d[:, 1] == 256)
                assert torch.all(storage[[0, -1]] == -7)
                assert torch.all(storage[1:129, :, 32:] == -7)
    finally:
        deep_gemm.set_num_sms(old_sms)


def test_sm120_reduction_empty_dimensions_and_fp32_accumulation_contract() -> None:
    if get_arch_major() != 12:
        return
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_num_sms(2)
        for m, n in ((0, 128), (128, 0), (128, 128)):
            a = torch.ones((1, m, 128), dtype=torch.bfloat16, device='cuda')
            b = torch.ones((1, n, 128), dtype=torch.bfloat16, device='cuda')
            for dtype in (torch.float32, torch.bfloat16):
                storage = torch.full((m * n + 32,), -7, dtype=dtype, device='cuda')
                d = storage[16:16 + m * n].view(m, n)
                d.fill_(3)
                deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=d if dtype == torch.float32 else None)
                assert torch.all(d == (131 if dtype == torch.float32 else 128))
                assert torch.all(storage[:16] == -7) and torch.all(storage[-16:] == -7)
            d = torch.empty((m, n), dtype=torch.float32, device='cuda')
            try:
                deep_gemm.einsum('bmk,bnk->mn', a, b, d)
            except RuntimeError:
                pass
            else:
                raise AssertionError('FP32 reduction must require C=D')
    finally:
        deep_gemm.set_num_sms(old_sms)


if __name__ == '__main__':
    test_sm120_batched_output_row_stride()
    test_sm120_reduction_empty_dimensions_and_fp32_accumulation_contract()
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_bmk_bnk_mn()
    test_bhr_hdr_bhd()
    test_bhd_hdr_bhr()
    test_bhd_bhr_hdr()

    test_fp8_bhr_hdr_bhd()
    test_fp8_bhd_hdr_bhr()
    test_fp8_bhd_bhr_hdr()
