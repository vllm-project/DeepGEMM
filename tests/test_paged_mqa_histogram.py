"""FP32 histogram producer: live-score parity, accumulation and graph replay."""
import pytest
import torch
import deep_gemm as dg

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100",
)


def histogram_reference(scores, lens):
    values = scores.float().contiguous()
    bits = values.view(torch.int32).long() & 0xFFFFFFFF
    mag = bits & 0x7FFFFFFF
    negative = ((bits >> 31) != 0) & (mag != 0)
    half = values.half().view(torch.int16).long() & 0xFFFF
    code = (half & 0x7FFF) >> 6
    adjusted = (mag - negative.long()).clamp(max=0x435F0000).int().view(torch.float32)
    wide = (adjusted.floor().long() + 288).clamp(min=304)
    code = torch.where(code >= 304, wide, code)
    bins = torch.where(negative, 512 + code, 511 - code)
    valid = torch.arange(scores.shape[1], device="cuda")[None, :] < lens
    valid &= ~values.isnan()
    out = torch.zeros((scores.shape[0], 1024), device="cuda", dtype=torch.int32)
    out.scatter_add_(1, bins.clamp(0, 1023), valid.int())
    return out


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("varlen", [False, True])
def test_histogram_live_scores_and_graph(n, varlen):
    torch.manual_seed(n)
    width, requests, pages = 8192, 3, 128
    rows = requests * n
    q = torch.randn((requests, n, 32, 128), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn((rows, 32), device="cuda") / 8
    cache = torch.empty((requests * pages, 64 * 132), device="cuda", dtype=torch.uint8)
    cache[:, :64 * 128] = torch.randn((requests * pages, 64 * 128), device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    cache[:, 64 * 128:] = torch.ones((requests * pages, 64), device="cuda").view(torch.uint8)
    cache = cache.view(-1, 64, 1, 132)
    table = torch.randperm(requests * pages, device="cuda").int().view(requests, pages)
    lens = torch.full((requests, n), width, dtype=torch.int32, device="cuda")
    ids = None
    if varlen:
        ids = torch.arange(requests, device="cuda", dtype=torch.int32).repeat_interleave(n)
        q = q.view(rows, 1, 32, 128)
        table = table.repeat_interleave(n, 0)
        lens = lens.view(rows, 1)
    hist = torch.zeros((rows, 1024), device="cuda", dtype=torch.int32)

    def produce(histogram=None):
        schedule = dg.get_paged_mqa_logits_metadata(lens, 64, dg.get_num_sms(), indices=ids)
        return dg.fp8_fp4_paged_mqa_logits((q, None), cache, weights, lens, table,
            schedule, width, indices=ids, histogram=histogram)

    produce(hist)
    plain = produce()
    assert torch.equal(hist, histogram_reference(plain, lens.view(-1, 1)))
    produce(hist)
    assert torch.equal(hist, 2 * histogram_reference(plain, lens.view(-1, 1)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hist.zero_()
        result = produce(hist)
    for length in (0, 1, 255, 256, 257, 8191):
        lens.copy_((length - n + 1 + torch.arange(rows, device="cuda").int() % n)
                   .clamp_min(0).view_as(lens))
        if varlen:
            # Split request runs on replay, including partial Q blocks at the tail.
            ids.copy_(torch.arange(rows, device="cuda").int() // (n if length % 2 else 1))
        graph.replay()
        plain = produce()
        live = torch.arange(width, device="cuda")[None, :] < lens.view(-1, 1)
        assert torch.equal(result.masked_fill(~live, 0), plain.masked_fill(~live, 0))
        assert torch.equal(hist, histogram_reference(result, lens.view(-1, 1)))
