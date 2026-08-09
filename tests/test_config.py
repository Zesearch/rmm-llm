import unittest

from rmm.config import RMMConfig
from rmm.quantization import ModelLoadConfig


class ConfigTest(unittest.TestCase):
    def test_keep_ratio_sweep(self):
        config = RMMConfig(enabled=False)
        config.set_keep_ratio(0.5)
        self.assertTrue(config.enabled)
        self.assertEqual(config.dimension_keep_ratio, 0.5)
        self.assertEqual(config.token_keep_ratio, 0.5)

    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):
            RMMConfig(dimension_keep_ratio=0.0)

    def test_invalid_quantization(self):
        config = ModelLoadConfig("model", quantization="unknown")
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()

