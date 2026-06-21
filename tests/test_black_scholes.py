import unittest

from jepa_iv.black_scholes import black_scholes_price, greeks, implied_volatility


class BlackScholesTests(unittest.TestCase):
    def test_implied_vol_recovers_known_price(self) -> None:
        price = black_scholes_price("call", 100.0, 105.0, 0.75, 0.04, 0.27)
        iv = implied_volatility(price, "call", 100.0, 105.0, 0.75, 0.04)
        self.assertAlmostEqual(iv, 0.27, places=8)

    def test_greeks_are_reasonable(self) -> None:
        g = greeks("put", 100.0, 95.0, 0.5, 0.03, 0.2)
        self.assertGreaterEqual(g.delta, -1.0)
        self.assertLessEqual(g.delta, 0.0)
        self.assertGreater(g.vega, 0.0)


if __name__ == "__main__":
    unittest.main()
