"""Market intelligence: deterministic synthetic prices or a local CSV. No network access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_desk.artifacts import MarketData, Provenance
from quant_desk.constitution import Constitution
from quant_desk.roles import Agent, Task

ASSETS = ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO")
TRADING_DAYS = 252


def synthetic_prices(seed: int, days: int = 2520, assets: tuple[str, ...] = ASSETS) -> pd.DataFrame:
    """Regime-switching GBM: each asset flips between up-trend, down-trend and calm regimes.

    Regimes are persistent (Markov chain, ~200-day mean stay), so trend-following has a
    planted edge in this synthetic world. A shared market factor adds cross-asset correlation.
    """
    rng = np.random.default_rng(seed)
    n = len(assets)
    drift = np.array([0.40, -0.35, 0.0]) / TRADING_DAYS  # per regime, annual -> daily
    vol = np.array([0.18, 0.28, 0.12]) / np.sqrt(TRADING_DAYS)
    stay = 0.995
    scale = rng.uniform(0.8, 1.3, size=n)  # per-asset volatility multiplier

    regime = rng.integers(0, 3, size=n)
    log_ret = np.empty((days, n))
    for t in range(days):
        switch = rng.random(n) > stay
        regime = np.where(switch, (regime + rng.integers(1, 3, size=n)) % 3, regime)
        sigma = vol[regime] * scale
        shock = 0.3 * rng.standard_normal() + np.sqrt(0.91) * rng.standard_normal(n)
        log_ret[t] = drift[regime] - 0.5 * sigma**2 + sigma * shock

    dates = pd.bdate_range("2015-01-01", periods=days, name="date")
    start = rng.uniform(20, 200, size=n)
    prices = start * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices.round(4), index=dates, columns=list(assets))


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load long-format closes (columns: date, asset, close) into a wide price table.

    Dates where any asset is missing are dropped so every row is a complete cross-section.
    """
    df = pd.read_csv(path)
    missing = {"date", "asset", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    if df.duplicated(["date", "asset"]).any():
        raise ValueError(f"{path}: duplicate (date, asset) rows")
    if (df["close"] <= 0).any():
        raise ValueError(f"{path}: closes must be positive")
    wide = df.pivot(index="date", columns="asset", values="close").sort_index().dropna()
    wide.columns = [str(c) for c in wide.columns]
    if len(wide) < 2:
        raise ValueError(f"{path}: need at least two complete dates")
    return wide.astype(float)


def gather_market_data(
    agent: Agent, constitution: Constitution, *, seed: int, csv_path: str | Path | None = None
) -> MarketData:
    constitution.require_role(agent, Task.GATHER_MARKET_DATA)
    if csv_path is None:
        prices, source = synthetic_prices(seed), f"synthetic regime-switching GBM (seed={seed})"
    else:
        prices, source = load_csv(csv_path), f"csv:{Path(csv_path).name}"
    market = MarketData(Provenance.of(agent), prices, source)
    note = f"{prices.shape[1]} assets x {len(prices)} days from {source}"
    return constitution.publish(market, agent, note)
