import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class InputAwareLinearTest(unittest.TestCase):
    def test_selected_columns_match_manual_linear(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from rmm.linear import InputAwareLinear

        base = nn.Linear(4, 2, bias=True)
        inputs = torch.tensor([[[10.0, 1.0, 5.0, 0.0], [10.0, 1.0, 5.0, 0.0]]])
        wrapped = InputAwareLinear(base, keep_ratio=0.5)

        actual = wrapped(inputs)
        expected = F.linear(inputs[..., [0, 2]], base.weight[:, [0, 2]], base.bias)
        torch.testing.assert_close(actual, expected)

    def test_qkv_and_mlp_targets_are_independent_and_restored(self):
        import torch.nn as nn

        from rmm.linear import InputAwareLinear, projection_pruning

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.k_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 4)
                self.o_proj = nn.Linear(4, 4)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 8)
                self.up_proj = nn.Linear(4, 8)
                self.down_proj = nn.Linear(8, 4)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
                self.mlp = MLP()

        model = Model()
        original_q = model.attn.q_proj
        with projection_pruning(model, targets=["qkv"], keep_ratio=0.5) as handle:
            self.assertEqual(len(handle.names), 3)
            self.assertIsInstance(model.attn.q_proj, InputAwareLinear)
            self.assertIsInstance(model.mlp.up_proj, nn.Linear)
            self.assertIsInstance(model.attn.o_proj, nn.Linear)
        self.assertIs(model.attn.q_proj, original_q)

        with projection_pruning(model, targets=["mlp"], keep_ratio=0.5) as handle:
            self.assertEqual(len(handle.names), 3)
            self.assertIsInstance(model.mlp.gate_proj, InputAwareLinear)
            self.assertIsInstance(model.attn.q_proj, nn.Linear)


if __name__ == "__main__":
    unittest.main()
