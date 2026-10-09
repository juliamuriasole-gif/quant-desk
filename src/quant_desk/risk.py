"""Risk controller: approves or vetoes strategies and allocations against the frozen limits."""

from __future__ import annotations

from quant_desk.artifacts import (
    Allocation,
    Artifact,
    BacktestResult,
    Provenance,
    RiskDecision,
    ValidationReport,
    Verdict,
)
from quant_desk.constitution import Constitution
from quant_desk.research import max_drawdown, sharpe
from quant_desk.roles import Agent, Task

TOLERANCE = 1e-9


def _decide(
    agent: Agent,
    constitution: Constitution,
    subject: Artifact,
    checks: list[tuple[bool, str]],
    *parents: Artifact,
) -> RiskDecision:
    approved = all(ok for ok, _ in checks)
    reasons = tuple(("ok: " if ok else "breach: ") + text for ok, text in checks)
    decision = RiskDecision(Provenance.of(agent, *parents), subject.content_hash, approved, reasons)
    note = (
        f"checked {subject.kind}:{subject.content_hash[:12]} by {subject.provenance.author_id}: "
        + ("APPROVED" if approved else "VETOED")
    )
    return constitution.publish(decision, agent, note)


def assess_strategy(
    agent: Agent, constitution: Constitution, result: BacktestResult, report: ValidationReport
) -> RiskDecision:
    constitution.require_role(agent, Task.ASSESS_RISK)
    constitution.assert_independent(agent, report)
    if report.subject_hash != result.content_hash:
        raise constitution.violation(agent, "validation report is for a different backtest")
    limits = constitution.limits
    oos = result.returns.loc[report.oos_start :]
    oos_sr, oos_dd = sharpe(oos), max_drawdown(oos)
    checks = [
        (report.verdict is Verdict.PASS, f"validator verdict {report.verdict.value}"),
        (
            oos_sr >= limits.min_oos_sharpe,
            f"OOS Sharpe {oos_sr:.2f} vs min {limits.min_oos_sharpe}",
        ),
        (
            oos_dd <= limits.max_oos_drawdown,
            f"OOS max drawdown {oos_dd:.1%} vs max {limits.max_oos_drawdown:.0%}",
        ),
    ]
    return _decide(agent, constitution, result, checks, result, report)


def assess_allocation(
    agent: Agent, constitution: Constitution, allocation: Allocation
) -> RiskDecision:
    constitution.require_role(agent, Task.ASSESS_RISK)
    constitution.assert_independent(agent, allocation)
    limits = constitution.limits
    largest = max((abs(w) for _, w in allocation.asset_weights), default=0.0)
    checks = [
        (
            allocation.gross <= limits.max_gross_exposure + TOLERANCE,
            f"gross exposure {allocation.gross:.2%} vs max {limits.max_gross_exposure:.0%}",
        ),
        (
            largest <= limits.max_asset_weight + TOLERANCE,
            f"largest asset weight {largest:.2%} vs max {limits.max_asset_weight:.0%}",
        ),
    ]
    return _decide(agent, constitution, allocation, checks, allocation)
