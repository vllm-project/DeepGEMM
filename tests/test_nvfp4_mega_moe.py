"""NVFP4 MegaMoE correctness and MiniMax M3 microbenchmark.

python tests/test_nvfp4_mega_moe.py --tokens 1 17 129 --hidden 512 --intermediate-hidden 256 --experts 8
python tests/test_nvfp4_mega_moe.py --benchmark --output benchmarks/minimax_m3_nvfp4.json
"""
import argparse
import hashlib
import inspect
import json
import os
import socket
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import (
    align,
    per_token_cast_to_fp4, cast_back_from_fp4,
    per_token_cast_to_fp8, cast_back_from_fp8,
)
from deep_gemm.utils.nvfp4 import per_token_cast_to_nvfp4, cast_back_from_nvfp4
from deep_gemm.utils.dist import init_dist


def mn_major(sf):
    return sf.transpose(-1, -2).contiguous().transpose(-1, -2)


def shared_sf_layout(sf, block_m, dst):
    """Scatter logical rows into the kernel's padded UTCCP pages."""
    rows = torch.arange(sf.size(0), device=sf.device)
    local = rows % block_m
    indices = rows // block_m * align(block_m, 128) + local // 128 * 128 + local % 32 * 4 + local % 128 // 32
    dst.zero_()
    dst[indices] = sf


def quantize(x, mode, scale=1.0):
    if mode == 'nvfp4':
        return per_token_cast_to_nvfp4(x, scale)
    if mode == 'fp4':
        return per_token_cast_to_fp4(x, True, 32, True)
    return per_token_cast_to_fp8(x, True, 32, True)


def dequantize(x, mode, scale=1.0):
    x = x[0], x[1].contiguous()
    if mode == 'nvfp4':
        return cast_back_from_nvfp4(*x, scale)
    if mode == 'fp4':
        return cast_back_from_fp4(*x, gran_k=32, use_packed_ue8m0=True)
    return cast_back_from_fp8(*x, gran_k=32, use_packed_ue8m0=True)


def make_weights(experts, hidden, intermediate, mode, scale):
    weights = []
    for n, k in ((intermediate * 2, hidden), (hidden, intermediate)):
        q = torch.empty((experts, n, k // (2 if mode in ('fp4', 'nvfp4') else 1)),
                        dtype=torch.int8 if mode in ('fp4', 'nvfp4') else torch.float8_e4m3fn, device='cuda')
        sf = torch.empty((experts, n, k // (64 if mode == 'nvfp4' else 128)), dtype=torch.int32, device='cuda')
        for e in range(experts):
            x = torch.randn((n, k), dtype=torch.bfloat16, device='cuda') * (k ** -0.5)
            q[e], sf[e] = quantize(x, mode, scale)
        weights.append((q, mn_major(sf)))
    return weights


def activation(x, weights, args):
    gate, up = x.to(torch.bfloat16).float().chunk(2, -1)
    gate = gate.clamp(max=args.clamp)
    up = up.clamp(-args.clamp, args.clamp)
    return gate * torch.sigmoid(gate * args.alpha) * (up + args.beta) * weights


def reference(x, shared_x, indices, weights, w, sw, mode, args, rank, world):
    """Independent eager reference using decoded FP4/FP8 and FP32 matmuls."""
    all_x, all_idx, all_weights = [], [], []
    for tensor, output in ((x, all_x), (indices, all_idx), (weights, all_weights)):
        output.extend(torch.empty_like(tensor) for _ in range(world))
        dist.all_gather(output, tensor)
    gx, gi, gw = torch.cat(all_x), torch.cat(all_idx), torch.cat(all_weights)
    y = torch.zeros_like(gx, dtype=torch.float32)
    local_experts = args.experts // world
    act_mode = 'nvfp4' if mode == 'nvfp4' else 'fp8'
    for e in range(local_experts):
        rows, slots = torch.where(gi == e + rank * local_experts)
        if rows.numel() == 0:
            continue
        w1 = dequantize((w[0][0][e], w[0][1][e]), mode, args.weight_scale if mode == 'nvfp4' else 1)
        w2 = dequantize((w[1][0][e], w[1][1][e]), mode, args.weight_scale if mode == 'nvfp4' else 1)
        mid = activation(gx[rows] @ w1.T, gw[rows, slots, None], args)
        mid = dequantize(quantize(mid, act_mode, args.mid_scale), act_mode, args.mid_scale)
        y.index_add_(0, rows, (mid @ w2.T).to(torch.bfloat16).float())
    dist.all_reduce(y)
    y = y[rank * x.size(0):(rank + 1) * x.size(0)]
    if sw is not None:
        w1, w2 = (weight.float() for weight in sw) if args.shared_dtype == 'bf16' else (dequantize(pair, 'fp8') for pair in sw)
        mid = activation(shared_x @ w1.T, 1.0, args)
        mid = mid.to(torch.bfloat16).float() if args.shared_dtype == 'bf16' else dequantize(quantize(mid, 'fp8'), 'fp8')
        y += (mid @ w2.T).to(torch.bfloat16).float()
    return y.to(torch.bfloat16)


def measure(fn, cold, repeats):
    """CUDA graph timing; cache flush is outside each measured event interval."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device='cuda') if cold else None
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(repeats)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for start, end in zip(starts, ends):
            if cache is not None:
                cache.zero_()
            start.record()
            fn()
            end.record()
    samples = []
    for i in range(4):
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        if i:
            samples.extend(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    return {'median_us': statistics.median(samples), 'min_us': min(samples), 'max_us': max(samples),
            'samples_us': samples}


def run(local_rank, local_world, args):
    rank, world, group = init_dist(local_rank, local_world)
    started_utc = datetime.now(timezone.utc).isoformat()
    def gpu_process_snapshot():
        try:
            return subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'],
                text=True, timeout=10).strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            return None
    initial_gpu_processes = gpu_process_snapshot() if local_rank == 0 else None
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.num_sms:
        deep_gemm.set_num_sms(args.num_sms)
    torch.manual_seed(args.seed + rank)
    mode = {'nvfp4': 'nvfp4', 'fp8xfp4': 'fp4', 'fp8xfp8': 'fp8'}[args.mma_type]
    nv = mode == 'nvfp4'
    assert nv or args.shared_dtype == 'mxfp8', 'BF16 shared experts require the NVFP4 API'
    w = make_weights(args.experts // world, args.hidden, args.intermediate_hidden, mode, args.weight_scale)
    transformed = deep_gemm.transform_weights_for_mega_moe(*w)
    sw = None
    shared_transformed = (None, None)
    if args.shared_experts:
        # Shared weights are replicated across all EP ranks, independent of sharding.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(args.seed + 1000000)
            if args.shared_dtype == 'bf16':
                sw = [torch.randn((n, k), dtype=torch.bfloat16, device='cuda') * k ** -0.5
                      for n, k in ((2 * args.intermediate_hidden * args.shared_experts, args.hidden),
                                   (args.hidden, args.intermediate_hidden * args.shared_experts))]
            else:
                sw = [(q[0][0], q[1][0]) for q in make_weights(1, args.hidden, args.intermediate_hidden * args.shared_experts, 'fp8', 1)]
        shared_transformed = deep_gemm.transform_weights_for_mega_moe(*sw)
    l1_alpha = torch.full((args.experts // world,), args.input_scale * args.weight_scale, dtype=torch.float32, device='cuda') if nv else None
    l2_alpha = torch.full_like(l1_alpha, args.mid_scale * args.weight_scale) if nv else None
    results = []
    for m in args.tokens:
        torch.manual_seed(args.seed + rank + m)
        buffer_args = (group, args.experts, max(m, 1), args.topk, args.hidden,
                       args.intermediate_hidden, args.shared_experts)
        buf = (deep_gemm.NVFP4SymmBuffer(
            *buffer_args, shared_dtype=torch.bfloat16 if args.shared_dtype == 'bf16' else torch.float8_e4m3fn)
            if nv else deep_gemm.SymmBuffer(*buffer_args, mma_type=args.mma_type))
        x = torch.randn((m, args.hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((m, args.experts), dtype=torch.float32, device='cuda').sigmoid()
        weights, indices = scores.topk(args.topk, -1)
        weights = weights / weights.sum(-1, keepdim=True) * args.routing_scale
        if args.masked:
            indices[torch.rand_like(weights) < args.masked] = -1
            weights.masked_fill_(indices < 0, 0)
        qx = quantize(x, 'nvfp4' if nv else 'fp8', args.input_scale)
        buf.x[:m].copy_(qx[0])
        buf.x_sf[:m].copy_(qx[1])
        buf.topk_idx[:m].copy_(indices)
        buf.topk_weights[:m].copy_(weights)
        shared_x = None
        block_args = (world, args.experts, buf.num_max_tokens_per_rank, m, args.topk)
        block_m = (deep_gemm.get_block_m_for_nvfp4_mega_moe(*block_args) if nv else
                   deep_gemm.get_block_m_for_mega_moe(*block_args, args.mma_type))
        if args.shared_experts:
            if args.shared_dtype == 'bf16':
                buf.shared_l1_acts[:m].copy_(x)
                shared_x = x.float()
                assert buf.shared_l1_acts_sf is None and buf.shared_l2_acts_sf is None
            else:
                sq = quantize(x, 'fp8')
                buf.shared_l1_acts[:m].copy_(sq[0])
                shared_sf_layout(sq[1], block_m, buf.shared_l1_acts_sf)
                shared_x = dequantize(sq, 'fp8')
        y = torch.empty_like(x)
        stats = torch.zeros(args.experts // world, dtype=torch.int32, device='cuda')
        kwargs = dict(shared_l1_weights=shared_transformed[0], shared_l2_weights=shared_transformed[1],
                      cumulative_local_expert_recv_stats=stats,
                      activation_clamp=args.clamp, activation_alpha=args.alpha, activation_beta=args.beta,
                      fast_math=not args.exact_math)
        if nv:
            kwargs.update(l1_alpha=l1_alpha, l2_alpha=l2_alpha, l2_activation_scale=args.mid_scale)
        kernel = deep_gemm.nvfp4_mega_moe if nv else deep_gemm.fp8_fp4_mega_moe
        fn = lambda: kernel(y, *transformed, buf, **kwargs)
        if args.check_api and m == args.tokens[0]:
            # Reject mismatched buffers/recipes, malformed global scales and outputs
            # before a CUDA launch can reinterpret the wrong storage format.
            assert 'l1_alpha' not in inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
            assert 'recipe' not in inspect.signature(deep_gemm.nvfp4_mega_moe).parameters
            native_args = (y, *transformed, None, None, None, buf.buffer,
                           buf.handle.buffer_ptrs, rank, buf.num_max_tokens_per_rank,
                           args.experts, args.topk)
            bad_calls = [lambda: deep_gemm._C.fp8_fp4_mega_moe(
                *native_args, (1, 1, 16), 'swiglu', args.clamp,
                not args.exact_math, args.alpha, args.beta)]
            if nv:
                bad_calls += [
                    lambda: kernel(y.float(), *transformed, buf, **kwargs),
                    lambda: kernel(y[:, :-1], *transformed, buf, **kwargs),
                    lambda: deep_gemm.fp8_fp4_mega_moe(y, *transformed, buf),
                    lambda: kernel(y, *transformed, buf, **dict(kwargs, l1_alpha=l1_alpha.double())),
                    lambda: kernel(y, *transformed, buf, **dict(kwargs, l2_activation_scale=0.0)),
                ]
                # Prove that Python dispatches to its own native entry point,
                # without a hidden dependency on the legacy FP8/FP4 binding.
                with patch.object(deep_gemm._C, 'nvfp4_mega_moe') as native_nv, \
                     patch.object(deep_gemm._C, 'fp8_fp4_mega_moe') as native_mx:
                    fn()
                    native_nv.assert_called_once()
                    native_mx.assert_not_called()
            else:
                bad_calls.append(lambda: deep_gemm.nvfp4_mega_moe(y, *transformed, buf))
            for call in bad_calls:
                try:
                    call()
                except (AssertionError, RuntimeError):
                    pass
                else:
                    raise AssertionError('Invalid MegaMoE input was accepted')
        fn()
        torch.cuda.synchronize()
        row = {'m_per_rank': m, 'block_m': block_m}
        if not args.skip_check:
            ref = reference(dequantize(qx, 'nvfp4' if nv else 'fp8', args.input_scale),
                            shared_x, indices, weights, w, sw, mode, args, rank, world)
            error = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
            diff = (y.float() - ref.float()).abs().max() if m else torch.zeros((), device='cuda')
            row.update(relative_l2_error=error.item(), max_abs_error=diff.item())
            assert torch.isfinite(y).all(), row
            assert error < 0.005, row
            # Validate counts across rank boundaries as well as output values.
            count = torch.bincount(indices[indices >= 0], minlength=args.experts).int()
            dist.all_reduce(count)
            assert torch.equal(stats, count.chunk(world)[rank]), (stats, count)
            # Repeated launches exercise signal cleanup and accumulation.
            first = y.clone()
            fn()
            torch.cuda.synchronize()
            assert torch.equal(stats, count.chunk(world)[rank] * 2)
            torch.testing.assert_close(y, first, atol=0.03125, rtol=0.02)
            if nv and args.input_scale == args.weight_scale == args.mid_scale == 1.0:
                kernel(y, *transformed, buf, **dict(kwargs, l1_alpha=None, l2_alpha=None,
                                                   cumulative_local_expert_recv_stats=None))
                torch.testing.assert_close(y, first, atol=0.03125, rtol=0.02)
                assert torch.equal(stats, count.chunk(world)[rank] * 2)
        if args.benchmark:
            row.update(measure(fn, not args.warm_cache, args.repeats))
            row['tflops'] = 6 * m * args.hidden * args.intermediate_hidden * (args.topk + args.shared_experts) / (row['median_us'] * 1e6)
        gathered = [None] * world
        dist.all_gather_object(gathered, row)
        if rank == 0:
            print(json.dumps({'ranks': [{k: v for k, v in r.items() if k != 'samples_us'} for r in gathered]}), flush=True)
            results.append({'m_per_rank': m, 'ranks': gathered})
        dist.barrier()
        buf.destroy()
    workers = [None] * world
    dist.all_gather_object(workers, {
        'rank': rank, 'local_rank': local_rank, 'hostname': socket.gethostname(), 'pid': os.getpid(),
        'gpu': torch.cuda.get_device_name(), 'gpu_uuid': str(torch.cuda.get_device_properties(local_rank).uuid),
        'gpu_processes_at_start': initial_gpu_processes,
        'gpu_processes_at_end': gpu_process_snapshot() if local_rank == 0 else None,
    })
    if rank == 0 and args.output:
        impl = 'nvfp4' if nv else 'fp8_fp4'
        source_paths = ('csrc/jit_kernels/heuristics/mega_moe.hpp', 'csrc/apis/mega_moe.hpp',
                        f'csrc/jit_kernels/impls/sm100_{impl}_mega_moe.hpp',
                        f'deep_gemm/include/deep_gemm/impls/sm100_{impl}_mega_moe.cuh',
                        'deep_gemm/include/deep_gemm/layout/mega_moe.cuh',
                        'deep_gemm/include/deep_gemm/ptx/tcgen05.cuh',
                        'tests/test_nvfp4_mega_moe.py')
        if nv:
            source_paths += ('csrc/apis/nvfp4_mega_moe.hpp',
                             'csrc/jit_kernels/heuristics/nvfp4_mega_moe.hpp',
                             'deep_gemm/include/deep_gemm/layout/nvfp4_mega_moe.cuh',
                             'deep_gemm/include/deep_gemm/ptx/nvfp4.cuh',
                             'deep_gemm/mega/nvfp4.py', 'deep_gemm/utils/nvfp4.py')
        payload = {'config': vars(args), 'gpu': torch.cuda.get_device_name(), 'sm_count': torch.cuda.get_device_properties(0).multi_processor_count,
                   'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'started_utc': started_utc, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'world_size': world, 'workers': workers,
                   'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'slurm_nodes': os.getenv('SLURM_JOB_NODELIST'),
                   'gpu_processes_at_start': initial_gpu_processes,
                   'gpu_processes_at_end': gpu_process_snapshot(),
                   'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
                   'environment': {k: v for k, v in os.environ.items() if k.startswith('DG_NVFP4_MOE_') or k == 'CUDA_VISIBLE_DEVICES'},
                   'results': results}
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + '\n')
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, nargs='+', default=[2 ** i for i in range(14)])
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=3072)
    p.add_argument('--experts', type=int, default=128)
    p.add_argument('--topk', type=int, default=4)
    p.add_argument('--shared-experts', type=int, default=1)
    p.add_argument('--shared-dtype', choices=['mxfp8', 'bf16'], default='mxfp8')
    p.add_argument('--routing-scale', type=float, default=2.0)
    p.add_argument('--num-processes', type=int, default=1)
    p.add_argument('--num-sms', type=int, default=0)
    p.add_argument('--mma-type', choices=('nvfp4', 'fp8xfp4', 'fp8xfp8'), default='nvfp4')
    p.add_argument('--input-scale', type=float, default=1.0)
    p.add_argument('--weight-scale', type=float, default=1.0)
    p.add_argument('--mid-scale', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=1.702)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--clamp', type=float, default=7.0)
    p.add_argument('--masked', type=float, default=0)
    p.add_argument('--exact-math', action='store_true')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--skip-check', action='store_true')
    p.add_argument('--check-api', action='store_true')
    p.add_argument('--warm-cache', action='store_true')
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.num_processes == 1:
        run(0, 1, args)
    else:
        torch.multiprocessing.spawn(run, args=(args.num_processes, args), nprocs=args.num_processes)
