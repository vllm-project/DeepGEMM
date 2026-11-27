import argparse
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils.dist import dist_print, init_dist
from test_mega_moe_nvfp4_sm90 import _make_routing, _make_weights, _reference


# noinspection PyShadowingNames
def test(args: argparse.Namespace):
    # Back-to-back launches on one buffer alternate between two inputs, and with BLOCK_M 16 the ring is reused within a launch,
    # so a task that reads a tile before the producer's TMA store is visible gets a stale one and the output differs bitwise
    _, _, group = init_dist(0, 1)
    torch.manual_seed(0)
    hidden, intermediate_hidden, num_experts, num_topk = args.hidden, args.intermediate_hidden, args.num_experts, args.num_topk

    checkpoint = (_make_weights(num_experts, intermediate_hidden * 2, hidden), _make_weights(num_experts, hidden, intermediate_hidden))
    (l1_w, l1_sf, l1_gs), (l2_w, l2_sf, l2_gs) = deep_gemm.transform_nvfp4_weights_for_sm90_mega_moe(*checkpoint)
    buffer = deep_gemm.SymmBuffer(group, num_experts, args.num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, mma_type='bf16xnvfp4')
    situ = dict(activation='situ', activation_alpha=4.0, activation_beta=25.0)

    def stage(x, topk_idx, topk_weights):
        buffer.x[:x.shape[0]].copy_(x)
        buffer.topk_idx[:x.shape[0]].copy_(topk_idx)
        buffer.topk_weights[:x.shape[0]].copy_(topk_weights)

    def launch(y, num_tokens_for_block_m):
        deep_gemm.bf16_nvfp4_mega_moe(y, (l1_w, l1_sf), (l2_w, l2_sf), buffer, l1_gs, l2_gs, **situ, num_tokens_for_block_m=num_tokens_for_block_m)

    num_failures = 0
    for case in args.cases.split(','):
        num_tokens, routing, block_m_tokens = case.split(':')
        num_tokens, block_m_tokens = int(num_tokens), int(block_m_tokens) or None
        inputs = [((torch.randn((num_tokens, hidden)) * 0.5).bfloat16(), _make_routing(routing, num_tokens, num_experts, num_topk),
                   torch.rand((num_tokens, num_topk)) + 0.5) for _ in range(2)]
        expected = []
        for x, topk_idx, topk_weights in inputs:
            stage(x, topk_idx, topk_weights)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16)
            launch(y, block_m_tokens)
            torch.cuda.synchronize()
            ref = _reference(x, topk_idx, topk_weights, checkpoint, **situ)
            diff = (y.float() - ref.float()).norm().item() / ref.float().norm().item()
            assert y.isfinite().all() and diff < 0.01, f'{case}: rel-L2 {diff:.5f}'
            expected.append(y)
        assert not torch.equal(*expected)

        outputs = torch.empty((args.num_launches, num_tokens, hidden), dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(args.num_launches):
                stage(*inputs[i % 2])
                launch(outputs[i], block_m_tokens)

        num_mismatches = [0, 0]
        for replay in range(args.num_rounds * 2):
            if replay % 2 == 0:
                for i in range(args.num_launches):
                    stage(*inputs[i % 2])
                    launch(outputs[i], block_m_tokens)
            else:
                graph.replay()
            torch.cuda.synchronize()
            for i in range(args.num_launches):
                num_mismatches[replay % 2] += int(not torch.equal(outputs[i], expected[i % 2]))
        num_checked = args.num_rounds * args.num_launches
        dist_print(f' > {case}: {num_mismatches[0]}/{num_checked} eager and {num_mismatches[1]}/{num_checked} graph launches differ')
        num_failures += sum(num_mismatches)

    assert num_failures == 0, f'{num_failures} launches differ'
    dist_print(' > All launches bitwise equal')
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Repeat SM90 NVFP4 Mega MoE launches with alternating inputs, compared bitwise')
    parser.add_argument('--cases', type=str, default='1024:blocks:1,8192:random:1',
                        help='Comma-separated tokens:routing:num_tokens_for_block_m (0 for the default BLOCK_M)')
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--hidden', type=int, default=2304, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=1152, help='Intermediate hidden size')
    parser.add_argument('--num-experts', type=int, default=128, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of expert selections')
    parser.add_argument('--num-launches', type=int, default=32, help='Launches per round (and per graph)')
    parser.add_argument('--num-rounds', type=int, default=20, help='Eager rounds and graph replays per case')
    test(parser.parse_args())
