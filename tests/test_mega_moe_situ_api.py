from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from deep_gemm.mega import fp8_fp4_mega_moe


@pytest.mark.parametrize('arch_major, recipe', [(9, (1, 1, 128)), (10, (1, 1, 32))])
def test_situ_vllm_parameter_names(arch_major, recipe):
    buffer = SimpleNamespace(
        buffer=torch.empty(1),
        handle=SimpleNamespace(buffer_ptrs=[0]),
        group=SimpleNamespace(rank=lambda: 0),
        num_max_tokens_per_rank=1,
        num_experts=1,
        num_topk=1,
    )
    weights = (torch.empty(1), torch.empty(1))
    y = torch.empty((1, 1))

    with patch('deep_gemm.mega.torch.cuda.get_device_capability', return_value=(arch_major, 0)), \
         patch('deep_gemm.mega._C.fp8_fp4_mega_moe') as native:
        fp8_fp4_mega_moe(
            y, weights, weights, buffer,
            activation='situ', activation_clamp=None, fast_math=True,
            situ_beta=4.0, situ_linear_beta=25.0,
        )
        native.assert_called_once()
        args = native.call_args.args
        assert args[12] == recipe
        assert args[-3:] == (4.0, 25.0, None)

        native.reset_mock()
        fp8_fp4_mega_moe(
            y, weights, weights, buffer, activation='situ', situ_beta=4.0,
        )
        assert native.call_args.args[-3:] == (4.0, 0.0, None)

        native.reset_mock()
        fp8_fp4_mega_moe(
            y, weights, weights, buffer, activation='situ',
            activation_alpha=4.0, activation_beta=25.0,
        )
        assert native.call_args.args[-3:] == (4.0, 25.0, None)

        native.reset_mock()
        fp8_fp4_mega_moe(
            y, weights, weights, buffer, activation='situ',
            activation_alpha=4.0, activation_beta=25.0,
            situ_beta=4.0, situ_linear_beta=25.0,
        )
        assert native.call_args.args[-3:] == (4.0, 25.0, None)


@pytest.mark.parametrize('kwargs', [
    dict(activation_alpha=4.0, situ_beta=3.0),
    dict(activation_beta=25.0, situ_linear_beta=24.0),
])
def test_situ_parameter_conflicts(kwargs):
    with pytest.raises(ValueError, match='Conflicting activation_'):
        fp8_fp4_mega_moe(None, None, None, None, activation='situ', **kwargs)
