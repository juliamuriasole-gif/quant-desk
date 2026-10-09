"""Portfolio adviser: inverse-volatility blend of risk-approved strategies into paper weights."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_desk.artifacts import Allocation, BacktestResult, Provenance, RiskDecision
from quant_desk.constitution import Constitution
from quant_desk.market import TRADING_DAYS
from quant_desk.roles import Agent, Role, Task


def allocate(
    agent: Agent,
    constitution: Constitution,
    approved: Sequence[tuple[BacktestResult, RiskDecision]],
    *,
    target_vol: float = 0.10,
    window: int = TRADING_DAYS,
) -> Allocation:
    """Weight strategies by 1/vol over the trailing `window`, scale the blend to `target_vol`,
    then cap each asset at the per-asset limit and the book at the gross limit."""
    constitution.require_role(agent, Task.ALLOCATE)
    for result, decision in approved:
        if not (
            decision.approved
            and decision.subject_hash == result.content_hash
            and decision.provenance.role is Role.RISK_CONTROLLER
        ):
            raise constitution.violation(agent, f"{result.hypothesis} lacks risk approval")

    limits = constitution.limits
    results = [r for r, _ in approved]
    if not results:
        return constitution.publish(
            Allocation(Provenance.of(agent), "-", (), (), ()),
            agent,
            "no approved strategies: all cash",
        )
    vols = np.array([r.returns.iloc[-window:].std() for r in results])
    strat_w = (1 / vols) / (1 / vols).sum()
    blend = sum(w * r.returns.iloc[-window:] for w, r in zip(strat_w, results, strict=True))
    n_assets = results[0].positions.shape[1]
    exposure = (
        sum(w * r.positions.iloc[-1] for w, r in zip(strat_w, results, strict=True)) / n_assets
    )
    weights = exposure * target_vol / (blend.std() * np.sqrt(TRADING_DAYS))
    weights = weights.clip(-limits.max_asset_weight, limits.max_asset_weight)
    gross = weights.abs().sum()
    if gross > limits.max_gross_exposure:
        weights *= limits.max_gross_exposure / gross
    weights = np.sign(weights) * np.floor(weights.abs() * 1e4 + 1e-6) / 1e4  # 4 dp, toward 0

    allocation = Allocation(
        # Approvals are recorded by hash, not as parents: an approver is not an author, so the
        # risk controller can still independently check this allocation.
        Provenance.of(agent, *results),
        str(pd.Timestamp(results[0].positions.index[-1]).date()),
        tuple(d.content_hash for _, d in approved),
        tuple((r.hypothesis, round(float(w), 4)) for r, w in zip(results, strat_w, strict=True)),
        tuple((str(a), float(w) + 0.0) for a, w in weights.items()),  # + 0.0 drops -0.0
    )
    note = (
        f"{len(results)} strateg{'y' if len(results) == 1 else 'ies'}, gross {allocation.gross:.2%}"
    )
    return constitution.publish(allocation, agent, note)
