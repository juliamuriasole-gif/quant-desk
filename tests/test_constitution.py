import dataclasses

import pytest

from quant_desk.artifacts import Allocation, Provenance, RiskDecision, Verdict
from quant_desk.constitution import RiskLimits
from quant_desk.errors import ConstitutionViolation
from quant_desk.execution import execute_paper
from quant_desk.portfolio import allocate
from quant_desk.research import propose_hypotheses, run_backtest
from quant_desk.risk import assess_allocation, assess_strategy
from quant_desk.roles import Agent, Role
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
    blocked = [e.actor_id for e in constitution.audit if e.action == "VIOLATION BLOCKED"]
    assert blocked == ["desk_lead-1", "risk_controller-1"]
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


def _decision(constitution, risk, subject, approved):
    """A risk decision published by the risk controller itself."""
    decision = RiskDecision(Provenance.of(risk, subject), subject.content_hash, approved, ())
    return constitution.publish(decision, risk)


def _allocation(constitution, adviser, sources, approvals, weights):
    """An allocation built by hand (not via `allocate`) and published by the adviser."""
    allocation = Allocation(
        Provenance.of(adviser, *sources),
        as_of="2024-01-04",
        approvals=tuple(d.content_hash for d in approvals),
        strategy_weights=(("demo", 1.0),),
        asset_weights=weights,
    )
    return constitution.publish(allocation, adviser)


@pytest.fixture
def strategy(constitution, staff, tiny_market):
    researcher = staff[Role.QUANT_RESEARCHER]
    hypothesis = propose_hypotheses(researcher, constitution)[0]
    return run_backtest(researcher, constitution, hypothesis, tiny_market, 5.0)


@pytest.fixture
def allocation(constitution, staff, strategy):
    approval = _decision(constitution, staff[Role.RISK_CONTROLLER], strategy, True)
    return _allocation(
        constitution, staff[Role.PORTFOLIO_ADVISER], [strategy], [approval], (("X", 0.25),)
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
        # role spoofed in a hand-made Provenance: right shape, but never published by risk
        "never published": RiskDecision(
            Provenance("adviser", Role.RISK_CONTROLLER, (), frozenset()),
            allocation.content_hash,
            True,
            (),
        ),
    }
    for message, decision in bad.items():
        with pytest.raises(ConstitutionViolation, match=message):
            execute_paper(operator, constitution, allocation, decision, market, ledger)
    assert not ledger.exists()

    approval = assess_allocation(risk, constitution, allocation)
    fills = execute_paper(operator, constitution, allocation, approval, market, ledger)
    assert [(f.asset, f.side, f.weight) for f in fills] == [("X", "BUY", 0.25)]
    assert fills[0].notional == 25_000.0 and fills[0].price == 99.0


def test_allocation_over_limits_is_vetoed(constitution, staff, strategy):
    risk, adviser = staff[Role.RISK_CONTROLLER], staff[Role.PORTFOLIO_ADVISER]
    approval = _decision(constitution, risk, strategy, True)
    weights = (("X", 0.9), ("Y", 0.5))
    greedy = _allocation(constitution, adviser, [strategy], [approval], weights)
    decision = assess_allocation(risk, constitution, greedy)
    assert not decision.approved
    assert sum(r.startswith("breach") for r in decision.reasons) == 2


def test_adviser_refuses_strategies_without_risk_approval(constitution, staff, researched):
    _, results = researched
    veto = RiskDecision(
        Provenance.of(staff[Role.RISK_CONTROLLER]), results[0].content_hash, False, ()
    )
    with pytest.raises(ConstitutionViolation, match="lacks risk approval"):
        allocate(staff[Role.PORTFOLIO_ADVISER], constitution, [(results[0], veto)])


def test_hand_built_allocation_needs_risk_approved_sources(constitution, staff, strategy):
    """Bypassing `allocate` does not help: risk re-checks what the allocation is built from."""
    risk, adviser = staff[Role.RISK_CONTROLLER], staff[Role.PORTFOLIO_ADVISER]
    veto = _decision(constitution, risk, strategy, False)
    weights = (("X", 0.25),)
    cases = {
        "built on a vetoed strategy": _allocation(
            constitution, adviser, [strategy], [veto], weights
        ),
        "no approvals at all": _allocation(constitution, adviser, [strategy], [], weights),
        "weights from nowhere": _allocation(constitution, adviser, [], [], weights),
    }
    for case, allocation in cases.items():
        decision = assess_allocation(risk, constitution, allocation)
        assert not decision.approved, case
        assert decision.reasons[0].startswith("breach"), case


def test_audit_log_is_read_only(constitution, staff):
    assert isinstance(constitution.audit, tuple)
    with pytest.raises(AttributeError):
        constitution.audit.clear()


def test_only_rostered_agents_can_act(constitution, staff, researched, market):
    lead = staff[Role.DESK_LEAD]
    impostor = Agent("risk-9", Role.RISK_CONTROLLER)
    renamed = Agent(staff[Role.VALIDATOR].agent_id, Role.RISK_CONTROLLER)  # wrong role for id
    _, results = researched
    for agent in (impostor, renamed):
        with pytest.raises(ConstitutionViolation, match="not on the desk roster"):
            assess_strategy(agent, constitution, results[0], None)
    with pytest.raises(ConstitutionViolation, match="already has a role"):
        constitution.enrol(lead, Agent(staff[Role.VALIDATOR].agent_id, Role.RISK_CONTROLLER))
    with pytest.raises(ConstitutionViolation, match="already open"):
        constitution.open_desk(lead)


def test_forged_or_edited_artifacts_are_refused(constitution, staff, researched, market):
    validator, risk = staff[Role.VALIDATOR], staff[Role.RISK_CONTROLLER]
    hypotheses, results = researched
    # the validator hand-builds a "researcher" backtest so it can grade it itself
    laundered = dataclasses.replace(results[1], cost_bps=0.0)
    with pytest.raises(ConstitutionViolation, match="never published"):
        validate(validator, constitution, laundered, hypotheses[1], market, results)
    # nobody may publish under someone else's name either
    with pytest.raises(ConstitutionViolation, match="cannot publish"):
        constitution.publish(laundered, validator)
    # a FAIL report edited to PASS is not the report the validator published
    report = validate(validator, constitution, results[0], hypotheses[0], market, results)
    edited = dataclasses.replace(report, verdict=Verdict.PASS)
    with pytest.raises(ConstitutionViolation, match="never published"):
        assess_strategy(risk, constitution, results[0], edited)


def test_risk_only_accepts_reports_from_a_validator(constitution, staff, researched, market):
    researcher, validator = staff[Role.QUANT_RESEARCHER], staff[Role.VALIDATOR]
    hypotheses, results = researched
    real = validate(validator, constitution, results[0], hypotheses[0], market, results)
    own = dataclasses.replace(real, provenance=Provenance.of(researcher, results[0]))
    constitution.publish(own, researcher)  # the researcher's own "validation"
    with pytest.raises(ConstitutionViolation, match="not issued by a validator"):
        assess_strategy(staff[Role.RISK_CONTROLLER], constitution, results[0], own)
