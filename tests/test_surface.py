import unittest

import numpy as np
import pandas as pd

from jepa_iv.config import DataConfig, SurfaceGridConfig
from jepa_iv.surface import SurfaceScaler, add_mid_prices, interpolate_surface


class SurfaceTests(unittest.TestCase):
    def test_mid_prices(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01")],
                "expiry": [pd.Timestamp("2024-02-01")],
                "option_type": ["call"],
                "strike": [100.0],
                "bid": [1.0],
                "ask": [1.4],
                "last": [1.2],
                "volume": [100],
                "open_interest": [100],
                "underlying_price": [100.0],
            }
        )
        out = add_mid_prices(frame)
        self.assertAlmostEqual(out["mid"].iloc[0], 1.2)

    def test_scaler_round_trip(self) -> None:
        x = np.random.default_rng(0).normal(size=(10, 20, 12))
        scaler = SurfaceScaler.fit(x)
        self.assertTrue(np.allclose(scaler.inverse_transform(scaler.transform(x)), x))

    def test_interpolate_surface_shape(self) -> None:
        grid = SurfaceGridConfig()
        m, t = np.meshgrid(grid.moneyness_grid, grid.tenor_years, indexing="ij")
        day = pd.DataFrame({"moneyness": m.ravel(), "time_to_expiry": t.ravel(), "iv": 0.2 + 0.01 * m.ravel()})
        surface = interpolate_surface(day, grid)
        self.assertEqual(surface.shape, grid.shape)
        self.assertFalse(np.isnan(surface).any())


if __name__ == "__main__":
    unittest.main()
