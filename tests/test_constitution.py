import dataclasses

import pytest

from quant_desk.artifacts import Allocation, Provenance, RiskDecision
from quant_desk.constitution import RiskLimits
from quant_desk.errors import ConstitutionViolation
from quant_desk.execution import execute_paper
from quant_desk.portfolio import allocate
from quant_desk.risk import assess_allocation, assess_strategy
from quant_desk.roles import Role
from quant_desk.validation import validate


def _self_authored(result, agent_id):
    """The same backtest, but with `agent_id` somewhere in its provenance chain."""
    prov = dataclasses.replace(result.provenance, chain=result.provenance.chain | {agent_id})
    return dataclasses.replace(result, provenance=prov)


def test_validator_cannot_grade_its_own_work(constitution, staff, researched, market):
    validator = staff[Role.VALIDATOR]
    hypotheses, results = researched
    tainted = _self_authored(results[0], validator.agent_id)
    with pytest.raises(ConstitutionViolation, match="no agent grades its own work"):
        validate(validator, constitution, tainted, hypotheses[0], market, results)
    assert constitution.audit[-1].action == "VIOLATION BLOCKED"
    # the untouched artifact (researcher's work only) is fine
    validate(validator, constitution, results[0], hypotheses[0], market, results)


def test_risk_controller_cannot_assess_its_own_work(constitution, staff, researched, market):
    risk = staff[Role.RISK_CONTROLLER]
    hypotheses, results = researched
    report = validate(
        staff[Role.VALIDATOR], constitution, results[0], hypotheses[0], market, results
    )
    with pytest.raises(ConstitutionViolation):
        assess_strategy(risk, constitution, results[0], _self_authored(report, risk.agent_id))


def test_lead_cannot_override_risk_veto_and_attempt_is_logged(constitution, staff):
    veto = RiskDecision(Provenance.of(staff[Role.RISK_CONTROLLER]), "abc", False, ("breach",))
    for actor in (staff[Role.DESK_LEAD], staff[Role.RISK_CONTROLLER]):
        with pytest.raises(ConstitutionViolation, match="cannot be overridden"):
            constitution.override_veto(actor, veto)
    assert [e.actor_id for e in constitution.audit] == ["desk_lead-1", "risk_controller-1"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        veto.approved = True


def test_risk_limits_are_frozen_once_the_desk_starts(constitution):
    with pytest.raises(dataclasses.FrozenInstanceError):
        constitution.limits.max_gross_exposure = 5.0
    with pytest.raises(ConstitutionViolation):
        constitution.limits = RiskLimits(max_gross_exposure=5.0)
    with pytest.raises(ConstitutionViolation):
        constitution.audit = []
    assert constitution.limits == RiskLimits()


def test_risk_limits_reject_nonsense():
    with pytest.raises(ValueError):
        RiskLimits(max_asset_weight=2.0, max_gross_exposure=1.0)


@pytest.fixture
def allocation(constitution, staff):
    return Allocation(
        Provenance.of(staff[Role.PORTFOLIO_ADVISER]),
        as_of="2024-01-04",
        approvals=(),
        strategy_weights=(("demo", 1.0),),
        asset_weights=(("X", 0.25),),
    )


def test_execution_requires_an_approving_risk_decision(
    constitution, staff, allocation, tiny_market, tmp_path
):
    operator, risk = staff[Role.EXECUTION_OPERATOR], staff[Role.RISK_CONTROLLER]
    market = tiny_market
    ledger = tmp_path / "ledger.jsonl"
    other = dataclasses.replace(allocation, as_of="2024-01-05")
    forged = RiskDecision(
        Provenance.of(staff[Role.PORTFOLIO_ADVISER]), allocation.content_hash, True, ()
    )
    bad = {
        "no risk decision attached": None,
        "vetoed": RiskDecision(Provenance.of(risk), allocation.content_hash, False, ()),
        "different allocation": RiskDecision(Provenance.of(risk), other.content_hash, True, ()),
        "not issued by a risk controller": forged,
    }
    for message, decision in bad.items():
        with pytest.raises(ConstitutionViolation, match=message):
            execute_paper(operator, constitution, allocation, decision, market, ledger)
    assert not ledger.exists()

    approval = assess_allocation(risk, constitution, allocation)
    fills = execute_paper(operator, constitution, allocation, approval, market, ledger)
    assert [(f.asset, f.side, f.weight) for f in fills] == [("X", "BUY", 0.25)]
    assert fills[0].notional == 25_000.0 and fills[0].price == 99.0


def test_allocation_over_limits_is_vetoed(constitution, staff, allocation):
    greedy = dataclasses.replace(allocation, asset_weights=(("X", 0.9), ("Y", 0.5)))
    decision = assess_allocation(staff[Role.RISK_CONTROLLER], constitution, greedy)
    assert not decision.approved
    assert sum(r.startswith("breach") for r in decision.reasons) == 2


def test_adviser_refuses_strategies_without_risk_approval(constitution, staff, researched):
    _, results = researched
    veto = RiskDecision(
        Provenance.of(staff[Role.RISK_CONTROLLER]), results[0].content_hash, False, ()
    )
    with pytest.raises(ConstitutionViolation, match="lacks risk approval"):
        allocate(staff[Role.PORTFOLIO_ADVISER], constitution, [(results[0], veto)])
