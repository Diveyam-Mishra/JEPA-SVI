import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jepa_iv.data import store_raw_options


class DataTests(unittest.TestCase):
    def test_store_raw_options_creates_parent_dir_and_file(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01 10:00:00")],
                "expiry": [pd.Timestamp("2024-02-01")],
                "option_type": ["call"],
                "strike": [100.0],
                "bid": [1.0],
                "ask": [1.2],
                "last": [1.1],
                "volume": [10],
                "open_interest": [50],
                "underlying_price": [100.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "options.parquet"
            written = store_raw_options(frame, path)
            self.assertEqual(written, path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
