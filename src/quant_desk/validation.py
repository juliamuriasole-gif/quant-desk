"""Validator: an independent check of a backtest, never run by the agent that produced it.

Checks (all must pass): exact reproduction, lookahead (truncation) test, in-sample edge,
deflated Sharpe ratio across all trials, out-of-sample edge and OOS sub-period consistency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from statistics import NormalDist

import numpy as np
import pandas as pd

from quant_desk.artifacts import (
    BacktestResult,
    Hypothesis,
    MarketData,
    Provenance,
    ValidationReport,
    Verdict,
)
from quant_desk.constitution import Constitution
from quant_desk.research import backtest, sharpe, signals
from quant_desk.roles import Agent, Task

EULER_GAMMA = 0.5772156649015329


def _per_period_sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(returns.mean() / std) if std > 0 else 0.0


def deflated_sharpe(returns: pd.Series, n_trials: int) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014), per-period Sharpe units.

    Probability that the true Sharpe exceeds SR0, the maximum Sharpe expected from `n_trials`
    skill-less trials, correcting for sample length, skew and fat tails. For V[SR] we use the
    null sampling variance of the Sharpe estimator, 1/(T-1), instead of the empirical variance
    across trials: our grid contains mirror-image ideas (momentum vs mean reversion) whose
    genuine opposite signs inflate the cross-trial variance until nothing can ever pass.
    """
    nd, t = NormalDist(), len(returns)
    if t < 3:
        return 0.0  # too short to say anything: never passes
    sr = _per_period_sharpe(returns)
    sr0 = 0.0
    if n_trials > 1:
        sr0 = math.sqrt(1 / (t - 1)) * (
            (1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / n_trials)
            + EULER_GAMMA * nd.inv_cdf(1 - 1 / (n_trials * math.e))
        )
    skew, kurt = float(returns.skew()), float(returns.kurt()) + 3
    var = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if not var > 0:  # degenerate (e.g. constant returns): no evidence of skill
        return 0.0
    return nd.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(var))


def lookahead_free(hypothesis: Hypothesis, prices: pd.DataFrame, checkpoints: int = 5) -> bool:
    """A causal signal computed on data up to bar t must equal the full-sample signal up to t."""
    full = signals(hypothesis, prices).to_numpy()
    for t in np.linspace(len(prices) // 4, len(prices) - 2, checkpoints).astype(int):
        truncated = signals(hypothesis, prices.iloc[: t + 1]).to_numpy()
        if not np.allclose(truncated, full[: t + 1], equal_nan=True):
            return False
    return True


def validate(
    agent: Agent,
    constitution: Constitution,
    result: BacktestResult,
    hypothesis: Hypothesis,
    market: MarketData,
    trials: Sequence[BacktestResult],
    *,
    oos_fraction: float = 0.4,
    folds: int = 3,
    min_dsr: float = 0.95,
) -> ValidationReport:
    constitution.require_role(agent, Task.VALIDATE)
    constitution.assert_independent(agent, result)
    constitution.require_published(agent, result, hypothesis, market, *trials)

    split = int(len(result.returns) * (1 - oos_fraction))
    is_ret, oos_ret = result.returns.iloc[:split], result.returns.iloc[split:]
    is_sr, oos_sr = sharpe(is_ret), sharpe(oos_ret)
    dsr = deflated_sharpe(is_ret, len(trials))
    edges = np.linspace(0, len(oos_ret), folds + 1).astype(int)
    fold_srs = [sharpe(oos_ret.iloc[a:b]) for a, b in pairwise(edges)]
    positive_folds = sum(s > 0 for s in fold_srs)

    replay = signals(hypothesis, market.prices)
    reproduced = (
        hypothesis.content_hash in result.provenance.parents
        and np.allclose(replay.to_numpy(), result.positions.to_numpy())
        and np.allclose(backtest(replay, market.prices, result.cost_bps), result.returns)
    )
    checks = [
        (reproduced, "backtest reproduces from the hypothesis and data"),
        (lookahead_free(hypothesis, market.prices), "signals are causal (truncation test)"),
        (is_sr > 0, f"in-sample Sharpe {is_sr:.2f} > 0"),
        (dsr >= min_dsr, f"deflated Sharpe {dsr:.3f} >= {min_dsr} over {len(trials)} trials"),
        (oos_sr > 0, f"out-of-sample Sharpe {oos_sr:.2f} > 0"),
        (
            positive_folds * 2 > folds,
            f"OOS sub-periods: {positive_folds}/{folds} positive "
            f"({', '.join(f'{s:.2f}' for s in fold_srs)})",
        ),
    ]
    reasons = tuple(("ok: " if ok else "FAIL: ") + text for ok, text in checks)
    verdict = Verdict.PASS if all(ok for ok, _ in checks) else Verdict.FAIL
    report = ValidationReport(
        Provenance.of(agent, result),
        result.content_hash,
        verdict,
        str(oos_ret.index[0].date()),
        is_sr,
        oos_sr,
        dsr,
        len(trials),
        reasons,
    )
    note = (
        f"checked {result.kind}:{result.content_hash[:12]} {result.hypothesis} "
        f"by {result.provenance.author_id}: {verdict.value}"
    )
    return constitution.publish(report, agent, note)
