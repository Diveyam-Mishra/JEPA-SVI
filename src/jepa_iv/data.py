from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

REQUIRED_OPTION_COLUMNS = {
    "timestamp",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "underlying_price",
}

HISTORICAL_EOD_COLUMNS = {
    "[QUOTE_DATE]",
    "[EXPIRE_DATE]",
    "[UNDERLYING_LAST]",
    "[STRIKE]",
    "[C_BID]",
    "[C_ASK]",
    "[C_LAST]",
    "[C_IV]",
    "[C_VOLUME]",
    "[P_BID]",
    "[P_ASK]",
    "[P_LAST]",
    "[P_IV]",
    "[P_VOLUME]",
}


class OptionChainProvider(Protocol):
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return raw option-chain rows for every available observation date."""


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_dates: np.ndarray
    validation_dates: np.ndarray
    test_dates: np.ndarray


class LocalParquetProvider:
    """Adapter for already-collected option-chain Parquet files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        del symbol
        frame = load_parquet_input(self.path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        mask = (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
        return frame.loc[mask].copy()


class YFinanceOptionProvider:
    """Exploratory Yahoo Finance adapter.

    Yahoo option-chain history is sparse and not appropriate as the only source for a
    publishable study. This adapter exists so the rest of the pipeline can run
    against accessible data while institutional data adapters share the same API.
    """

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the data extra to use yfinance: uv sync --extra data") from exc

        ticker = yf.Ticker(symbol)
        rows: list[pd.DataFrame] = []
        snapshot_ts = pd.Timestamp.utcnow().floor("s").tz_localize(None)
        if not start <= snapshot_ts <= end:
            raise ValueError("yfinance adapter only provides current option chains, not full history")
        for expiry in ticker.options:
            chain = ticker.option_chain(expiry)
            for option_type, side in (("call", chain.calls), ("put", chain.puts)):
                side = side.copy()
                side["option_type"] = option_type
                side["expiry"] = pd.Timestamp(expiry)
                rows.append(side)
        if not rows:
            return pd.DataFrame(columns=sorted(REQUIRED_OPTION_COLUMNS))
        frame = pd.concat(rows, ignore_index=True)
        info = ticker.history(period="1d")
        underlying = float(info["Close"].iloc[-1])
        return pd.DataFrame(
            {
                "timestamp": snapshot_ts,
                "expiry": pd.to_datetime(frame["expiry"]),
                "option_type": frame["option_type"],
                "strike": frame["strike"].astype(float),
                "bid": frame["bid"].astype(float),
                "ask": frame["ask"].astype(float),
                "last": frame["lastPrice"].astype(float),
                "volume": frame["volume"].fillna(0).astype(int),
                "open_interest": frame["openInterest"].fillna(0).astype(int),
                "underlying_price": underlying,
            }
        )


def validate_option_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_OPTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"option frame missing required columns: {sorted(missing)}")

    clean = frame.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"]).dt.tz_localize(None)
    clean["expiry"] = pd.to_datetime(clean["expiry"]).dt.tz_localize(None)
    clean["option_type"] = clean["option_type"].str.lower()
    if not clean["option_type"].isin(["call", "put"]).all():
        raise ValueError("option_type must contain only 'call' or 'put'")

    numeric_cols = [
        "strike",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "underlying_price",
    ]
    clean[numeric_cols] = clean[numeric_cols].apply(pd.to_numeric, errors="coerce")
    critical = ["strike", "bid", "ask", "underlying_price", "expiry", "timestamp"]
    if clean[critical].isna().any().any():
        bad_cols = clean[critical].columns[clean[critical].isna().any()].tolist()
        raise ValueError(f"critical option fields contain missing values: {bad_cols}")
    if (clean["strike"] <= 0).any() or (clean["underlying_price"] <= 0).any():
        raise ValueError("strike and underlying_price must be positive")
    if (clean["ask"] < clean["bid"]).any():
        raise ValueError("ask must be greater than or equal to bid")
    return clean


def normalize_option_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if REQUIRED_OPTION_COLUMNS.issubset(frame.columns):
        return validate_option_frame(frame)
    if not HISTORICAL_EOD_COLUMNS.issubset(frame.columns):
        missing = REQUIRED_OPTION_COLUMNS.difference(frame.columns)
        raise ValueError(
            "unsupported option schema; missing standard columns "
            f"{sorted(missing)} and historical EOD schema was not detected"
        )

    base = frame.copy()
    quote_timestamp = pd.to_datetime(base["[QUOTE_READTIME]"], errors="coerce")
    if quote_timestamp.isna().all():
        quote_timestamp = pd.to_datetime(base["[QUOTE_DATE]"], errors="coerce")
    expiry = pd.to_datetime(base["[EXPIRE_DATE]"], errors="coerce")
    dte_days = pd.to_numeric(base.get("[DTE]"), errors="coerce")

    def make_side(prefix: str, option_type: str) -> pd.DataFrame:
        side = pd.DataFrame(
            {
                "timestamp": quote_timestamp,
                "expiry": expiry,
                "option_type": option_type,
                "strike": pd.to_numeric(base["[STRIKE]"], errors="coerce"),
                "bid": pd.to_numeric(base[f"[{prefix}_BID]"], errors="coerce"),
                "ask": pd.to_numeric(base[f"[{prefix}_ASK]"], errors="coerce"),
                "last": pd.to_numeric(base[f"[{prefix}_LAST]"], errors="coerce"),
                "volume": pd.to_numeric(base[f"[{prefix}_VOLUME]"], errors="coerce").fillna(0),
                # This vendor schema does not include OI. Reuse volume as a liquidity proxy
                # so the downstream filters remain operable until a richer source is used.
                "open_interest": pd.to_numeric(base[f"[{prefix}_VOLUME]"], errors="coerce").fillna(0),
                "underlying_price": pd.to_numeric(base["[UNDERLYING_LAST]"], errors="coerce"),
                "iv": pd.to_numeric(base[f"[{prefix}_IV]"], errors="coerce"),
                "dte_days": dte_days,
            }
        )
        side = side[(side["bid"].notna()) & (side["ask"].notna())].copy()
        side = side[(side["bid"] >= 0) & (side["ask"] >= 0)].copy()
        side = side[side["ask"] >= side["bid"]].copy()
        return side

    normalized = pd.concat([make_side("C", "call"), make_side("P", "put")], ignore_index=True)
    return validate_option_frame(normalized)


def resolve_parquet_inputs(path_or_pattern: str | Path) -> list[Path]:
    raw = str(path_or_pattern)
    if any(token in raw for token in ["*", "?", "["]):
        matches = [Path(match) for match in glob(raw)]
        if not matches:
            raise FileNotFoundError(f"no parquet files matched pattern: {raw}")
        return sorted(matches)
    path = Path(raw)
    if path.is_dir():
        matches = sorted(path.glob("*.parquet"))
        if not matches:
            raise FileNotFoundError(f"no parquet files found in directory: {path}")
        return matches
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def load_parquet_input(path_or_pattern: str | Path) -> pd.DataFrame:
    paths = resolve_parquet_inputs(path_or_pattern)
    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return normalize_option_frame(combined)


def store_raw_options(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = normalize_option_frame(frame)
    validated.to_parquet(path, index=False)
    return path


def chronological_split(
    surfaces: np.ndarray,
    dates: np.ndarray,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> DatasetSplit:
    if len(surfaces) != len(dates):
        raise ValueError("surfaces and dates must have the same length")
    if len(surfaces) < 3:
        raise ValueError("at least three surfaces are required for a temporal split")
    order = np.argsort(dates)
    surfaces = surfaces[order]
    dates = dates[order]
    train_end = max(1, int(len(surfaces) * train_ratio))
    val_end = max(train_end + 1, int(len(surfaces) * (train_ratio + validation_ratio)))
    val_end = min(val_end, len(surfaces) - 1)
    return DatasetSplit(
        train=surfaces[:train_end],
        validation=surfaces[train_end:val_end],
        test=surfaces[val_end:],
        train_dates=dates[:train_end],
        validation_dates=dates[train_end:val_end],
        test_dates=dates[val_end:],
    )
