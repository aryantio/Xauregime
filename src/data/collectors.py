"""
Data collection layer.

One interface (`Collector.collect() -> wide daily DataFrame`) with two
implementations:
  - FredYahooCollector : live macro (FRED) + prices (yfinance)
  - SampleCollector    : offline synthetic data for dev/demo
Pick via config `data.source`.
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Collector(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """Return a daily, forward-filled DataFrame indexed by date with columns:
        gold, silver, copper, dxy, dgs10, dfii10, t10yie, vix, oil
        """
        ...

    @staticmethod
    def _align(frame: pd.DataFrame, start: str) -> pd.DataFrame:
        frame = frame.sort_index()
        frame = frame[frame.index >= pd.Timestamp(start)]
        idx = pd.date_range(frame.index.min(), frame.index.max(), freq="B")
        return frame.reindex(idx).ffill()


def get_collector(cfg) -> Collector:
    return {"live": FredYahooCollector, "sample": SampleCollector}[cfg.data.source](cfg)


class FredYahooCollector(Collector):
    """Live data. Requires internet and a free FRED API key."""

    def collect(self) -> pd.DataFrame:
        from fredapi import Fred
        import yfinance as yf

        key = os.environ.get(self.cfg.data.fred_api_key_env)
        if not key:
            raise RuntimeError(
                f"Set {self.cfg.data.fred_api_key_env} env var. "
                "Get a free key at https://fredaccount.stlouisfed.org/apikey"
            )
        fred = Fred(api_key=key)
        start = self.cfg.data.start_date

        macro = {}
        for col, sid in self.cfg.data.fred_series.items():
            macro[col] = fred.get_series(sid, observation_start=start)
        macro = pd.DataFrame(macro)

        prices = {}
        for col, tkr in self.cfg.data.yf_tickers.items():
            s = yf.download(tkr, start=start, progress=False, auto_adjust=True)["Close"]
            prices[col] = s.squeeze()
        prices = pd.DataFrame(prices)

        frame = prices.join(macro, how="outer")
        return self._align(frame, start)


class SampleCollector(Collector):
    """Synthetic but plausible data so the full pipeline runs offline.

    NOT for trading — demo only.
    """

    def collect(self) -> pd.DataFrame:
        cfg = self.cfg
        rng = np.random.default_rng(7)
        idx = pd.date_range(cfg.data.start_date, periods=2300, freq="B")
        n = len(idx)

        t = np.arange(n)
        growth = np.sin(t / 240) + 0.3 * np.sin(t / 90) + rng.normal(0, 0.15, n).cumsum() * 0.02
        infl = np.cos(t / 300) + 0.2 * np.sin(t / 70) + rng.normal(0, 0.15, n).cumsum() * 0.02

        real = 1.0 - 0.6 * infl + 0.4 * growth + rng.normal(0, 0.05, n).cumsum() * 0.01
        breakeven = 2.2 + 0.5 * infl + rng.normal(0, 0.03, n)
        dgs10 = real + breakeven
        vix = np.clip(18 - 4 * growth + rng.normal(0, 2, n), 9, 70)
        oil = np.clip(70 * np.exp(0.15 * infl + 0.1 * growth + rng.normal(0, 0.02, n).cumsum()), 20, 160)
        dxy = 100 + 5 * real - 3 * infl + rng.normal(0, 0.3, n).cumsum() * 0.05

        gold_ret = -0.9 * np.diff(real, prepend=real[0]) + rng.normal(0, 0.008, n)
        gold = 1500 * np.exp(np.cumsum(gold_ret))
        silver_ret = 1.6 * gold_ret + 0.5 * np.diff(growth, prepend=growth[0]) + rng.normal(0, 0.012, n)
        silver = 18 * np.exp(np.cumsum(silver_ret))
        copper = 3.5 * np.exp(np.cumsum(0.6 * np.diff(growth, prepend=growth[0]) + rng.normal(0, 0.01, n)))

        frame = pd.DataFrame(
            {
                "gold": gold, "silver": silver, "copper": copper, "dxy": dxy,
                "dgs10": dgs10, "dfii10": real, "t10yie": breakeven,
                "vix": vix, "oil": oil,
            },
            index=idx,
        )
        return self._align(frame, cfg.data.start_date)
