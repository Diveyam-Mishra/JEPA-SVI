import unittest

import numpy as np

from jepa_iv.hedging import delta_hedging_backtest
from jepa_iv.svi import evaluate_svi_surface, fit_svi_surface


class SVIAndHedgingTests(unittest.TestCase):
    def test_svi_surface_fit_is_positive(self) -> None:
        moneyness = np.linspace(0.8, 1.2, 20)
        tenors = np.array([0.1, 0.25, 0.5])
        k = np.log(moneyness)
        surface = np.column_stack(
            [0.18 + 0.05 * k**2 + 0.01 * j for j in range(len(tenors))]
        )
        params = fit_svi_surface(surface, moneyness, tenors)
        fitted = evaluate_svi_surface(params, moneyness, tenors)
        self.assertEqual(fitted.shape, surface.shape)
        self.assertTrue(np.isfinite(fitted).all())
        self.assertTrue((fitted > 0).all())

    def test_delta_backtest_outputs_aligned_pnl(self) -> None:
        result = delta_hedging_backtest(
            "call",
            spots=np.array([100.0, 101.0, 99.0, 102.0]),
            strikes=np.array([100.0, 100.0, 100.0, 100.0]),
            tenors=np.array([0.20, 0.19, 0.18, 0.17]),
            predicted_iv=np.array([0.20, 0.21, 0.20, 0.22]),
            realised_iv=np.array([0.20, 0.20, 0.21, 0.21]),
            transaction_cost_bps=1.0,
        )
        self.assertEqual(len(result.pnl), 3)
        self.assertEqual(len(result.net_pnl), 3)
        self.assertGreaterEqual(result.pnl_variance, 0.0)


if __name__ == "__main__":
    unittest.main()
