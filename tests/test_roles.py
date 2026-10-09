import pytest

from quant_desk.errors import ConstitutionViolation, RoleViolation
from quant_desk.research import propose_hypotheses
from quant_desk.roles import ALLOWED_TASKS, Agent, Role, Task
from quant_desk.validation import validate


def test_every_role_has_tasks_and_every_task_has_a_role():
    assert set(ALLOWED_TASKS) == set(Role)
    assert set().union(*ALLOWED_TASKS.values()) == set(Task)


def test_agent_needs_a_role_at_construction():
    with pytest.raises(TypeError):
        Agent("nobody", "quant researcher")  # a string is not a Role


def test_agent_refuses_task_outside_its_role():
    lead = Agent("lead-1", Role.DESK_LEAD)
    assert lead.can(Task.ASSIGN_ROLES)
    with pytest.raises(RoleViolation, match="may not 'run backtest'"):
        lead.require(Task.RUN_BACKTEST)
    assert issubclass(RoleViolation, ConstitutionViolation)


def test_role_functions_enforce_role_and_log_the_attempt(constitution, staff, researched, market):
    with pytest.raises(RoleViolation):
        propose_hypotheses(staff[Role.VALIDATOR], constitution)
    hypotheses, results = researched
    with pytest.raises(RoleViolation):  # the researcher may not validate anything
        validate(
            staff[Role.QUANT_RESEARCHER], constitution, results[0], hypotheses[0], market, results
        )
    blocked = [e for e in constitution.audit if e.action == "VIOLATION BLOCKED"]
    assert [e.actor_id for e in blocked] == ["validator-1", "quant_researcher-1"]
