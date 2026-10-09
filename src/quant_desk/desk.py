"""Desk lead: assigns roles first, then runs the desk end-to-end and writes the report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_desk.artifacts import (
    Allocation,
    BacktestResult,
    Hypothesis,
    MarketData,
    PaperFill,
    RiskDecision,
    ValidationReport,
)
from quant_desk.constitution import Constitution, RiskLimits
from quant_desk.errors import ConstitutionViolation
from quant_desk.execution import execute_paper
from quant_desk.market import gather_market_data
from quant_desk.portfolio import allocate
from quant_desk.report import render
from quant_desk.research import propose_hypotheses, run_backtest
from quant_desk.risk import assess_allocation, assess_strategy
from quant_desk.roles import Agent, Role, Task
from quant_desk.validation import validate

STAFF: tuple[tuple[str, Role], ...] = (
    ("intel-1", Role.MARKET_INTELLIGENCE),
    ("researcher-1", Role.QUANT_RESEARCHER),
    ("validator-1", Role.VALIDATOR),
    ("risk-1", Role.RISK_CONTROLLER),
    ("adviser-1", Role.PORTFOLIO_ADVISER),
    ("exec-1", Role.EXECUTION_OPERATOR),
)


@dataclass(frozen=True)
class DeskRun:
    constitution: Constitution
    staff: tuple[Agent, ...]
    market: MarketData
    hypotheses: tuple[Hypothesis, ...]
    results: tuple[BacktestResult, ...]
    reports: tuple[ValidationReport, ...]
    strategy_decisions: tuple[RiskDecision, ...]
    allocation: Allocation
    allocation_decision: RiskDecision
    fills: tuple[PaperFill, ...]
    drill: str | None  # outcome of the lead's attempt to override a veto, if one happened


def assign_roles(lead: Agent, constitution: Constitution) -> tuple[Agent, ...]:
    constitution.require_role(lead, Task.ASSIGN_ROLES)
    staff = tuple(Agent(agent_id, role) for agent_id, role in STAFF)
    for agent in staff:
        constitution.record(lead, "assigned role", note=f"{agent.agent_id} -> {agent.role.value}")
    return (lead, *staff)


def run_desk(
    *,
    seed: int,
    out_dir: Path,
    csv_path: Path | None = None,
    cost_bps: float = 5.0,
    limits: RiskLimits | None = None,
) -> DeskRun:
    out_dir.mkdir(parents=True, exist_ok=True)
    constitution = Constitution(limits or RiskLimits())
    lead = Agent("lead-1", Role.DESK_LEAD)
    constitution.record(lead, "desk opened", note=f"risk limits frozen: {constitution.limits}")
    staff = assign_roles(lead, constitution)
    _, intel, researcher, validator, risk, adviser, operator = staff

    market = gather_market_data(intel, constitution, seed=seed, csv_path=csv_path)
    hypotheses = propose_hypotheses(researcher, constitution)
    results = tuple(run_backtest(researcher, constitution, h, market, cost_bps) for h in hypotheses)
    reports = tuple(
        validate(validator, constitution, r, h, market, results)
        for r, h in zip(results, hypotheses, strict=True)
    )
    decisions = tuple(
        assess_strategy(risk, constitution, r, rep) for r, rep in zip(results, reports, strict=True)
    )

    drill = None
    vetoed = [d for d in decisions if not d.approved]
    if vetoed:  # constitution drill: the lead leans on the risk controller and must be refused
        try:
            constitution.override_veto(lead, vetoed[0])
        except ConstitutionViolation as exc:
            drill = str(exc)

    approved = [(r, d) for r, d in zip(results, decisions, strict=True) if d.approved]
    allocation = allocate(adviser, constitution, approved)
    allocation_decision = assess_allocation(risk, constitution, allocation)
    ledger = out_dir / "ledger.jsonl"
    ledger.touch()
    fills: tuple[PaperFill, ...] = ()
    if allocation_decision.approved:
        fills = execute_paper(
            operator, constitution, allocation, allocation_decision, market, ledger
        )

    run = DeskRun(
        constitution,
        staff,
        market,
        hypotheses,
        results,
        reports,
        decisions,
        allocation,
        allocation_decision,
        fills,
        drill,
    )
    constitution.require_role(lead, Task.WRITE_REPORT)
    constitution.record(lead, "wrote desk report", note="report.md")
    (out_dir / "report.md").write_text(render(run))
    return run
