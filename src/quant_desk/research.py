"""Quant researcher: strategy hypotheses and a vectorised, lookahead-free backtest."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quant_desk.artifacts import BacktestResult, Hypothesis, MarketData, Provenance
from quant_desk.constitution import Constitution
from quant_desk.market import TRADING_DAYS
from quant_desk.roles import Agent, Task

Signal = Callable[..., pd.DataFrame]


def tsmom(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Long if the price is above its level `lookback` bars ago, short if below."""
    return np.sign(np.log(prices / prices.shift(lookback)))


def ma_cross(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Long when the fast moving average is above the slow one, short otherwise."""
    return np.sign(prices.rolling(fast).mean() - prices.rolling(slow).mean())


def zrev(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean reversion: fade the z-score of price against its rolling mean, capped at +/-1."""
    mean, std = prices.rolling(window).mean(), prices.rolling(window).std()
    return (-(prices - mean) / std / 2).clip(-1, 1)


# Positions returned at bar t use closes up to and including t only.
STRATEGIES: dict[str, Signal] = {"tsmom": tsmom, "ma_cross": ma_cross, "zrev": zrev}

GRID: tuple[tuple[str, dict[str, int], str], ...] = (
    ("tsmom", {"lookback": 20}, "short-horizon trend persists"),
    ("tsmom", {"lookback": 60}, "medium-horizon trend persists"),
    ("tsmom", {"lookback": 120}, "long-horizon trend persists"),
    ("ma_cross", {"fast": 10, "slow": 50}, "fast/slow crossover catches regime shifts"),
    ("ma_cross", {"fast": 20, "slow": 100}, "slower crossover, fewer whipsaws"),
    ("ma_cross", {"fast": 50, "slow": 200}, "classic golden/death cross"),
    ("zrev", {"window": 5}, "weekly overreactions revert"),
    ("zrev", {"window": 20}, "monthly overreactions revert"),
)


def signals(hypothesis: Hypothesis, prices: pd.DataFrame) -> pd.DataFrame:
    raw = STRATEGIES[hypothesis.family](prices, **dict(hypothesis.params))
    return raw.fillna(0.0)


def backtest(positions: pd.DataFrame, prices: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Equal-weight daily net returns across assets.

    The position decided at the close of bar t-1 earns the return from t-1 to t (signals are
    shifted one bar). Changing the position at bar t costs `cost_bps` per unit of turnover;
    the book starts flat, so the first bar pays to enter its position.
    """
    asset_ret = prices / prices.shift(1) - 1
    gross = (positions.shift(1) * asset_ret).fillna(0.0)
    turnover = positions.diff().fillna(positions).abs()  # first bar trades from flat
    net = (gross - turnover * cost_bps / 1e4).mean(axis=1)
    return net.rename("net_return")


def sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve, counting the starting capital (1.0)
    as a peak so that losses from the very first bar are included."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return float((1 - equity / peak).max()) if len(returns) else 0.0


def propose_hypotheses(agent: Agent, constitution: Constitution) -> tuple[Hypothesis, ...]:
    constitution.require_role(agent, Task.PROPOSE_HYPOTHESES)
    hypotheses = []
    for family, params, rationale in GRID:
        name = f"{family}(" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"
        hyp = Hypothesis(Provenance.of(agent), name, family, tuple(params.items()), rationale)
        hypotheses.append(constitution.publish(hyp, agent, name))
    return tuple(hypotheses)


def run_backtest(
    agent: Agent,
    constitution: Constitution,
    hypothesis: Hypothesis,
    market: MarketData,
    cost_bps: float,
) -> BacktestResult:
    constitution.require_role(agent, Task.RUN_BACKTEST)
    positions = signals(hypothesis, market.prices)
    returns = backtest(positions, market.prices, cost_bps)
    result = BacktestResult(
        Provenance.of(agent, hypothesis, market), hypothesis.name, positions, returns, cost_bps
    )
    note = f"{hypothesis.name}: full-sample Sharpe {sharpe(returns):.2f}"
    return constitution.publish(result, agent, note)
