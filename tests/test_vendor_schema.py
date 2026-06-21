import unittest

import pandas as pd

from jepa_iv.data import normalize_option_frame


class VendorSchemaTests(unittest.TestCase):
    def test_normalize_vendor_eod_schema(self) -> None:
        frame = pd.DataFrame(
            {
                "[QUOTE_READTIME]": ["2023-01-03 16:00:00"],
                "[QUOTE_DATE]": ["2023-01-03"],
                "[EXPIRE_DATE]": ["2023-02-17"],
                "[UNDERLYING_LAST]": [380.0],
                "[STRIKE]": [375.0],
                "[DTE]": [45.0],
                "[C_BID]": [8.1],
                "[C_ASK]": [8.3],
                "[C_LAST]": [8.2],
                "[C_IV]": [0.22],
                "[C_VOLUME]": [125.0],
                "[P_BID]": [3.1],
                "[P_ASK]": [3.3],
                "[P_LAST]": [3.2],
                "[P_IV]": [0.24],
                "[P_VOLUME]": [95.0],
            }
        )
        normalized = normalize_option_frame(frame)
        self.assertEqual(set(normalized["option_type"]), {"call", "put"})
        self.assertEqual(len(normalized), 2)
        self.assertIn("iv", normalized.columns)
        self.assertIn("dte_days", normalized.columns)

    def test_normalize_vendor_eod_drops_crossed_quotes(self) -> None:
        frame = pd.DataFrame(
            {
                "[QUOTE_READTIME]": ["2023-01-03 16:00:00", "2023-01-03 16:00:00"],
                "[QUOTE_DATE]": ["2023-01-03", "2023-01-03"],
                "[EXPIRE_DATE]": ["2023-02-17", "2023-02-17"],
                "[UNDERLYING_LAST]": [380.0, 380.0],
                "[STRIKE]": [375.0, 380.0],
                "[DTE]": [45.0, 45.0],
                "[C_BID]": [8.1, 5.0],
                "[C_ASK]": [8.3, 4.0],
                "[C_LAST]": [8.2, 4.5],
                "[C_IV]": [0.22, 0.21],
                "[C_VOLUME]": [125.0, 100.0],
                "[P_BID]": [3.1, 6.0],
                "[P_ASK]": [3.3, 5.0],
                "[P_LAST]": [3.2, 5.5],
                "[P_IV]": [0.24, 0.23],
                "[P_VOLUME]": [95.0, 80.0],
            }
        )
        normalized = normalize_option_frame(frame)
        self.assertEqual(len(normalized), 2)
        self.assertTrue((normalized["ask"] >= normalized["bid"]).all())


if __name__ == "__main__":
    unittest.main()
