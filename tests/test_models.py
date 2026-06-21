import unittest

import torch

from jepa_iv.config import JEPAConfig
from jepa_iv.masking import block_mask_indices
from jepa_iv.models import SurfaceJEPA, patchify, unpatchify


class ModelTests(unittest.TestCase):
    def test_patch_round_trip(self) -> None:
        x = torch.arange(2 * 20 * 12, dtype=torch.float32).reshape(2, 20, 12)
        tokens = patchify(x, (4, 3))
        reconstructed = unpatchify(tokens, (20, 12), (4, 3))
        self.assertTrue(torch.equal(x, reconstructed))

    def test_jepa_forward_shapes(self) -> None:
        config = JEPAConfig(embed_dim=16, encoder_depth=1, encoder_heads=4, predictor_depth=1)
        model = SurfaceJEPA(config)
        x = torch.randn(3, 20, 12)
        context, target = block_mask_indices(3, (5, 4), config.mask_ratio)
        pred, expected = model(x, context, target)
        self.assertEqual(pred.shape, expected.shape)
        self.assertEqual(pred.shape[0], 3)
        self.assertEqual(pred.shape[-1], 16)


if __name__ == "__main__":
    unittest.main()
