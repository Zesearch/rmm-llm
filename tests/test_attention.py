import importlib.util
import math
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class AttentionTest(unittest.TestCase):
    def test_full_keep_matches_dense_attention(self):
        import torch

        from rmm.attention import rmm_attention_forward
        from rmm.config import RMMConfig

        class Module:
            num_key_value_groups = 1
            training = False

        torch.manual_seed(7)
        query = torch.randn(2, 3, 4, 8)
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        scale = 1.0 / math.sqrt(8)

        actual, _ = rmm_attention_forward(
            Module(), query, key, value, None, scale, config=RMMConfig()
        )
        weights = torch.softmax(query @ key.transpose(2, 3) * scale, dim=-1)
        expected = (weights @ value).transpose(1, 2).contiguous()
        torch.testing.assert_close(actual, expected)

    def test_pruned_output_shape(self):
        import torch

        from rmm.attention import rmm_attention_forward
        from rmm.config import RMMConfig

        class Module:
            num_key_value_groups = 2
            training = False

        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 2, 5, 8)
        value = torch.randn(1, 2, 5, 8)
        output, weights = rmm_attention_forward(
            Module(),
            query,
            key,
            value,
            None,
            1.0 / math.sqrt(8),
            config=RMMConfig(0.5, 0.5),
        )
        self.assertEqual(output.shape, (1, 3, 4, 8))
        self.assertEqual(weights.shape, (1, 4, 3, 5))


if __name__ == "__main__":
    unittest.main()

