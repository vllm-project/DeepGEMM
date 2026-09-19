import bisect
import itertools
import random

import pytest
import torch

import deep_gemm
from deep_gemm.testing import (
    get_arch_major, test_filter
)


def metadata_reference(lengths, slots, indices=None):
    starts, atom_lengths = [], []
    token = 0
    while token < len(lengths):
        paired = indices is not None and token + 1 < len(lengths) and indices[token] == indices[token + 1]
        starts.append(token)
        atom_lengths.append(lengths[token + int(paired)])
        token += 2 if paired else 1
    prefix = list(itertools.accumulate((length + 127) // 128 for length in atom_lengths))
    total = prefix[-1]
    if total == 0 and indices is None:
        return [[len(lengths), 0]] * (slots + 1)
    q, r = divmod(total, slots)
    result = []
    for sm in range(slots + 1):
        position = sm * q + max(0, sm - (slots - r))
        atom = bisect.bisect_right(prefix, position)
        before = prefix[atom - 1] if atom else 0
        result.append([starts[atom] if atom < len(starts) else len(lengths), position - before])
    return result


def check_outputs(call, reference, valid):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = call()
        second = call()
    stream.synchronize()
    torch.testing.assert_close(first[valid].double(), reference[valid], rtol=1e-5, atol=1e-5)
    assert torch.equal(first[valid], second[valid])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured[valid].double(), reference[valid], rtol=1e-5, atol=1e-5)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('head_dim', [32, 64, 128])
@pytest.mark.parametrize('paged', [False, True])
def test_fp8_mqa_logits_head_dimension_mapping(head_dim, paged):
    batch, heads, tokens = 4, 16, 256
    q_cpu = torch.randint(-2, 3, (batch, heads, head_dim)).float()
    kv_cpu = torch.randint(-2, 3, (tokens, head_dim)).float()
    weights_cpu = torch.randint(-2, 3, (batch, heads)).float()
    q = q_cpu.to(torch.float8_e4m3fn).cuda()
    kv = kv_cpu.to(torch.float8_e4m3fn).cuda()
    weights = weights_cpu.cuda()
    lengths = [0, 1, 65, 193]
    valid = torch.arange(tokens, device='cuda')[None, :] < torch.tensor(lengths, device='cuda')[:, None]
    reference = torch.einsum('mhd,nd->mhn', q_cpu.double(), kv_cpu.double()).relu()
    reference = (reference * weights_cpu.double()[:, :, None]).sum(1).cuda()
    if paged:
        fused = torch.empty((4, 64 * (head_dim + 4)), dtype=torch.uint8, device='cuda')
        fused[:, :64 * head_dim] = kv.reshape(4, 64 * head_dim).view(torch.uint8)
        scales = torch.ones((4, 64), dtype=torch.float32, device='cuda')
        fused[:, 64 * head_dim:] = scales.view(torch.uint8)
        cache = fused.view(4, 64, 1, head_dim + 4)
        context = torch.tensor(lengths, dtype=torch.int32, device='cuda').view(batch, 1)
        table = torch.arange(4, dtype=torch.int32, device='cuda').repeat(batch, 1)
        metadata = torch.tensor(metadata_reference(lengths, deep_gemm.get_num_sms()), dtype=torch.int32, device='cuda')
        call = lambda: deep_gemm.fp8_fp4_paged_mqa_logits(
            (q.view(batch, 1, heads, head_dim), None), cache, weights, context, table, metadata,
            tokens, clean_logits=False, logits_dtype=torch.float32)
    else:
        starts = torch.zeros(batch, dtype=torch.int32, device='cuda')
        ends = torch.tensor(lengths, dtype=torch.int32, device='cuda')
        scales = torch.ones(tokens, dtype=torch.float32, device='cuda')
        call = lambda: deep_gemm.fp8_fp4_mqa_logits(
            (q, None), (kv, scales), weights, starts, ends, clean_logits=False,
            max_seqlen_k=tokens, logits_dtype=torch.float32)
    check_outputs(call, reference, valid)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('batch', [33, 65])
@pytest.mark.parametrize('varlen', [False, True])
def test_metadata_prefix_visibility(batch, varlen):
    slots = 8
    lengths = [[0, 1, 128, 129, 256, 384][i % 6] for i in range(batch)]
    indices = [i // 2 for i in range(batch)] if varlen else None
    if varlen:
        lengths = [lengths[i // 2 * 2] + i % 2 for i in range(batch)]
    context = torch.tensor(lengths, dtype=torch.int32, device='cuda').view(batch, 1)
    indices_gpu = torch.tensor(indices, dtype=torch.int32, device='cuda') if varlen else None
    expected = torch.tensor(metadata_reference(lengths, slots, indices), dtype=torch.int32, device='cuda')
    call = lambda: deep_gemm.get_paged_mqa_logits_metadata(context, 64, slots, indices=indices_gpu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            actual = call()
    stream.synchronize()
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('page_kv', [32, 64])
@pytest.mark.parametrize('heads', [16, 32, 64])
@pytest.mark.parametrize('next_n,varlen', [(1, False), (2, False), (3, False), (1, True)])
@pytest.mark.parametrize('head_dim', [32, 64, 128])
def test_fp8_paged_mqa_page_geometry(page_kv, heads, next_n, varlen, head_dim):
    """Check native FP8 MMA with shuffled pages, padded strides and partial tails."""
    torch.manual_seed(17)
    lengths = [0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257]
    batch, tokens = len(lengths), 384
    pages_per_request = tokens // page_kv
    num_pages = batch * pages_per_request
    q_cpu = torch.randint(-4, 5, (batch, next_n, heads, head_dim)).float() / 2
    k_cpu = torch.randint(-4, 5, (num_pages, page_kv, head_dim)).float() / 2
    scale_cpu = 2.0 ** torch.randint(-3, 3, (num_pages, page_kv)).float()
    weights_cpu = torch.randint(-4, 5, (batch * next_n, heads)).float() / 8
    q = q_cpu.to(torch.float8_e4m3fn).cuda()
    weights = weights_cpu.cuda()
    # Distinct physical pages, including a gap between pages as in hybrid caches.
    row_bytes = page_kv * (head_dim + 4)
    fused = torch.zeros((num_pages, row_bytes + 128), dtype=torch.uint8, device='cuda')
    fused[:, :page_kv * head_dim] = k_cpu.to(torch.float8_e4m3fn).cuda().reshape(num_pages, -1).view(torch.uint8)
    fused[:, page_kv * head_dim:row_bytes] = scale_cpu.cuda().view(torch.uint8)
    cache = fused[:, :row_bytes].view(num_pages, page_kv, 1, head_dim + 4)
    table_cpu = torch.randperm(num_pages).view(batch, pages_per_request)
    indices_cpu = torch.arange(batch, dtype=torch.int32) // 2 if varlen else None
    if varlen:
        # Adjacent paired tokens share one request's page table; its second row
        # must be at least as long as the first for the paired scheduler atom.
        table_cpu = table_cpu[indices_cpu.long()].clone()
    context_cpu = torch.tensor(lengths, dtype=torch.int32)[:, None]
    context_cpu = (context_cpu - next_n + 1 + torch.arange(next_n, dtype=torch.int32)).clamp_min(0)
    context = context_cpu.cuda()
    table = table_cpu.to(device='cuda', dtype=torch.int32)
    indices = indices_cpu.cuda() if varlen else None
    reference = torch.empty((batch * next_n, tokens), dtype=torch.float64)
    for b in range(batch):
        keys = k_cpu[table_cpu[b]].reshape(tokens, head_dim).double()
        scales = scale_cpu[table_cpu[b]].reshape(tokens).double()
        score = torch.einsum('nhd,kd->nhk', q_cpu[b].double(), keys).relu()
        score = (score * weights_cpu[b * next_n:(b + 1) * next_n].double()[..., None]).sum(1)
        reference[b * next_n:(b + 1) * next_n] = score * scales
    valid = torch.arange(tokens, device='cuda')[None, :] < context.flatten()[:, None]

    def call():
        metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context, page_kv, deep_gemm.get_num_sms(), indices=indices)
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, None), cache, weights, context, table, metadata, tokens,
            clean_logits=False, logits_dtype=torch.float32, indices=indices)

    check_outputs(call, reference.cuda(), valid)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    for paged in (False, True):
        for head_dim in (32, 64, 128):
            test_fp8_mqa_logits_head_dimension_mapping(head_dim, paged)
    for varlen in (False, True):
        for batch in (33, 65):
            test_metadata_prefix_visibility(batch, varlen)
