import itertools
import os
from unittest import SkipTest, TestCase, main, mock

import torch
from torch import nn
from torch import dtype as _dtype

from rtp_llm.models_py.modules.fusion import gated_mlp
from rtp_llm.models_py.modules.fusion.gated_mlp import fusion_mode
from rtp_llm.models_py.modules.fusion.quant_activation import QuantizedActivation
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.modules.hybrid.test.dense_mlp_ref import DenseMLP as DenseMLPRef
from rtp_llm.ops import ActivationType, ParallelismConfig
from rtp_llm.utils.model_weight import W


class _FakeParallelismConfig:
    def get_ffn_tp_size(self):
        return 1


class _FakeUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 1


class _FakeDownBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return x + 100


class _FakeDown(_FakeDownBase):
    def __init__(self):
        super().__init__()
        self.quantize_calls = 0

    def quantize_fused_silu_and_mul(self, x):
        self.quantize_calls += 1
        return x + 10


class _FakeFullUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = 128
        self.N = 256
        self.weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
        self.weight_scales = torch.empty((2, 1), dtype=torch.float32)
        self.bias = None


class _FakeFullDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = 128
        self.last_input = None

    def quantize_fused_silu_and_mul(self, x):
        return x

    def forward(self, x):
        self.last_input = x
        return torch.tensor([[7.0]])


class DenseMlpFusionModeTest(TestCase):
    def _make_mlp(self, enable_quantize: bool = True) -> DenseMLP:
        mlp = DenseMLP.__new__(DenseMLP)
        nn.Module.__init__(mlp)
        mlp.is_gated = True
        mlp.up_proj = _FakeUp()
        mlp.down_proj = _FakeDown() if enable_quantize else _FakeDownBase()
        mlp.act_fn = nn.Identity()
        mlp.parallelism_config = _FakeParallelismConfig()
        mlp.fusion_mode = "off"
        return mlp

    def test_legacy_env_enables_silu_quant(self):
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_DENSE_SILU_MUL_QUANT_FUSION": "1",
            },
            clear=True,
        ):
            self.assertEqual(fusion_mode(), "silu_quant")

    def test_new_env_overrides_legacy_env(self):
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_DENSE_SILU_MUL_QUANT_FUSION": "1",
                "RTP_LLM_DENSE_MLP_FUSION": "off",
            },
            clear=True,
        ):
            self.assertEqual(fusion_mode(), "off")

    def test_off_mode_uses_eager_path(self):
        mlp = self._make_mlp(enable_quantize=True)
        x = torch.tensor([[1.0]])
        mlp.fusion_mode = "off"
        out = DenseMLP.forward(mlp, x)
        self.assertEqual(out.item(), 102.0)
        self.assertEqual(mlp.down_proj.quantize_calls, 0)

    def test_silu_quant_mode_uses_quantized_path(self):
        mlp = self._make_mlp(enable_quantize=True)
        x = torch.tensor([[1.0]])
        mlp.fusion_mode = "silu_quant"
        out = DenseMLP.forward(mlp, x)
        self.assertEqual(out.item(), 112.0)
        self.assertEqual(mlp.down_proj.quantize_calls, 1)

    def test_silu_quant_mode_falls_back_when_backend_is_missing(self):
        mlp = self._make_mlp(enable_quantize=False)
        x = torch.tensor([[1.0]])
        mlp.fusion_mode = "silu_quant"
        out = DenseMLP.forward(mlp, x)
        self.assertEqual(out.item(), 102.0)

    def test_serving_fusion_mode_rejects_full(self):
        with mock.patch.dict(
            os.environ,
            {"RTP_LLM_DENSE_MLP_FUSION": "full"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "RTP_LLM_DENSE_MLP_FUSION"):
                fusion_mode()

    def test_gate_up_staged_producer_requires_op(self):
        mlp = self._make_mlp(enable_quantize=True)
        x = torch.tensor([[1.0]])
        mlp.fusion_mode = "gate_up_staged_producer"
        with self.assertRaisesRegex(
            RuntimeError, "cutlass_gated_mlp_staged_producer_sm120_fp8"
        ):
            DenseMLP.forward(mlp, x)

    def test_gate_up_staged_producer_wraps_output(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        x = torch.empty((2, 128), dtype=torch.bfloat16)
        data = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        scale = torch.empty((1, 2), dtype=torch.float32).transpose(0, 1)

        def fake_op(input, weight, weight_scales, bias):
            self.assertIs(input, x)
            self.assertIs(weight, up_proj.weight)
            self.assertIs(weight_scales, up_proj.weight_scales)
            self.assertIsNone(bias)
            return data, scale

        with mock.patch.object(
            gated_mlp, "_gate_up_staged_producer_op", return_value=fake_op
        ):
            out = gated_mlp.try_gated_mlp_fusion(
                "gate_up_staged_producer", up_proj, down_proj, x, is_gated=True
            )

        self.assertEqual(out.item(), 7.0)
        self.assertIsInstance(down_proj.last_input, QuantizedActivation)
        self.assertIs(down_proj.last_input.data, data)
        self.assertIs(down_proj.last_input.scale, scale)
        self.assertEqual(down_proj.last_input.orig_dtype, torch.bfloat16)

    def test_gate_up_producer_wraps_output(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        x = torch.empty((2, 128), dtype=torch.bfloat16)
        data = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        scale = torch.empty((1, 2), dtype=torch.float32).transpose(0, 1)

        def fake_op(input, weight, weight_scales, bias):
            self.assertIs(input, x)
            self.assertIs(weight, up_proj.weight)
            self.assertIs(weight_scales, up_proj.weight_scales)
            self.assertIsNone(bias)
            return data, scale

        def fake_compute_op(name):
            return fake_op if name == gated_mlp.GATE_UP_PRODUCER_OP_NAME else None

        with mock.patch.object(
            gated_mlp,
            "_compute_op",
            side_effect=fake_compute_op,
        ):
            out = gated_mlp.try_gated_mlp_fusion(
                "gate_up_producer", up_proj, down_proj, x, is_gated=True
            )

        self.assertEqual(out.item(), 7.0)
        self.assertIsInstance(down_proj.last_input, QuantizedActivation)
        self.assertIs(down_proj.last_input.data, data)
        self.assertIs(down_proj.last_input.scale, scale)

    def test_gate_up_producer_requires_op(self):
        mlp = self._make_mlp(enable_quantize=True)
        x = torch.tensor([[1.0]])
        mlp.fusion_mode = "gate_up_producer"
        with self.assertRaisesRegex(
            RuntimeError, "cutlass_gated_mlp_producer_sm120_fp8"
        ):
            DenseMLP.forward(mlp, x)

    def test_gate_up_staged_producer_rejects_bad_stride(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        x = torch.empty((2, 128), dtype=torch.bfloat16)
        data = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        scale = torch.empty((2, 1), dtype=torch.float32).contiguous()

        with mock.patch.object(
            gated_mlp,
            "_gate_up_staged_producer_op",
            return_value=lambda *_: (data, scale),
        ):
            with self.assertRaisesRegex(ValueError, "column-major"):
                gated_mlp.try_gate_up_staged_producer(
                    up_proj, down_proj, x, is_gated=True
                )

    def test_gate_up_staged_producer_empty_input_is_not_compatible(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        x = torch.empty((0, 128), dtype=torch.bfloat16)

        with mock.patch.object(
            gated_mlp,
            "_gate_up_staged_producer_op",
            return_value=lambda *_: self.fail("producer should not be called"),
        ):
            self.assertIsNone(
                gated_mlp.try_gate_up_staged_producer(
                    up_proj, down_proj, x, is_gated=True
                )
            )

    def test_gate_up_staged_producer_bad_input_dim_is_not_compatible(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        x = torch.empty((2, 1, 128), dtype=torch.bfloat16)

        with mock.patch.object(
            gated_mlp,
            "_gate_up_staged_producer_op",
            return_value=lambda *_: self.fail("producer should not be called"),
        ):
            self.assertIsNone(
                gated_mlp.try_gate_up_staged_producer(
                    up_proj, down_proj, x, is_gated=True
                )
            )

    def test_gate_up_staged_producer_shape_mismatch_is_not_compatible(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        up_proj.K = 256
        x = torch.empty((2, 128), dtype=torch.bfloat16)

        with mock.patch.object(
            gated_mlp,
            "_gate_up_staged_producer_op",
            return_value=lambda *_: self.fail("producer should not be called"),
        ):
            self.assertIsNone(
                gated_mlp.try_gate_up_staged_producer(
                    up_proj, down_proj, x, is_gated=True
                )
            )

    def test_gate_up_staged_producer_unaligned_hidden_is_not_compatible(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        up_proj.K = 64
        up_proj.weight = torch.empty((256, 64), dtype=torch.float8_e4m3fn)
        up_proj.weight_scales = torch.empty((2, 1), dtype=torch.float32)
        x = torch.empty((2, 64), dtype=torch.bfloat16)

        with mock.patch.object(
            gated_mlp,
            "_gate_up_staged_producer_op",
            return_value=lambda *_: self.fail("producer should not be called"),
        ):
            self.assertIsNone(
                gated_mlp.try_gate_up_staged_producer(
                    up_proj, down_proj, x, is_gated=True
                )
            )

    def test_gate_up_staged_producer_bad_weight_scale_is_not_compatible(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        up_proj.weight_scales = torch.empty((1, 2), dtype=torch.float32)
        x = torch.empty((2, 128), dtype=torch.bfloat16)

        with mock.patch.object(
            gated_mlp,
            "_gate_up_staged_producer_op",
            return_value=lambda *_: self.fail("producer should not be called"),
        ):
            self.assertIsNone(
                gated_mlp.try_gate_up_staged_producer(
                    up_proj, down_proj, x, is_gated=True
                )
            )

    def test_gate_up_staged_producer_noncontiguous_weight_is_not_compatible(self):
        up_proj = _FakeFullUp()
        down_proj = _FakeFullDown()
        up_proj.weight = torch.empty((4, 256), dtype=torch.float8_e4m3fn).t()
        x = torch.empty((2, 128), dtype=torch.bfloat16)

        with mock.patch.object(
            gated_mlp,
            "_gate_up_staged_producer_op",
            return_value=lambda *_: self.fail("producer should not be called"),
        ):
            self.assertIsNone(
                gated_mlp.try_gate_up_staged_producer(
                    up_proj, down_proj, x, is_gated=True
                )
            )

    def test_invalid_fusion_mode_fails_fast(self):
        with mock.patch.dict(
            os.environ,
            {"RTP_LLM_DENSE_MLP_FUSION": "gated_gemm"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "RTP_LLM_DENSE_MLP_FUSION"):
                fusion_mode()


class MLPTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096, 5120]
    HIDDEN_SIZES = [768, 2048, 4096, 5120, 8192]

    #
    # DTYPES = [torch.bfloat16]
    # NUM_TOKENS = [5120]
    # HIDDEN_SIZES = [512]
    #

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_mlp_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        parallelism_config = ParallelismConfig()
        parallelism_config.tp_size = 1
        parallelism_config.tp_rank = 0

        weights = {}
        weights[W.ffn_w1] = torch.randn(hidden_size, 4 * hidden_size, dtype=dtype)
        torch.nn.init.xavier_uniform_(weights[W.ffn_w1])
        weights[W.ffn_w3] = torch.randn(hidden_size, 4 * hidden_size, dtype=dtype)
        torch.nn.init.xavier_uniform_(weights[W.ffn_w3])
        weights[W.ffn_w2] = torch.randn(4 * hidden_size, hidden_size, dtype=dtype)
        torch.nn.init.xavier_uniform_(weights[W.ffn_w2])

        qwen3_mlp = DenseMLPRef(
            weights[W.ffn_w1],
            weights[W.ffn_w3],
            weights[W.ffn_w2],
            ActivationType.Swiglu,
        )
        qwen3_mlp_fused = DenseMLP(
            ActivationType.Swiglu, parallelism_config, weights, quant_config=None
        )

        x = torch.randn(num_tokens, hidden_size, dtype=dtype)

        # for _ in range(5):
        #     out = qwen3_mlp(x)
        #     out = qwen3_mlp_fused(x)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        # for _ in range(10):
        #     out = qwen3_mlp(x)
        #     out = qwen3_mlp_fused(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        self.assertTrue(
            torch.allclose(qwen3_mlp(x), qwen3_mlp_fused(x), atol=1e-2, rtol=1e-2)
        )

    def test_mlp(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.DTYPES
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_mlp_test(*params)


if __name__ == "__main__":
    main()
