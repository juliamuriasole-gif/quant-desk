"""Roles, tasks and agents: every agent gets a role before it gets a task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quant_desk.errors import RoleViolation


class Role(Enum):
    DESK_LEAD = "desk lead"
    MARKET_INTELLIGENCE = "market intelligence"
    QUANT_RESEARCHER = "quant researcher"
    VALIDATOR = "validator"
    RISK_CONTROLLER = "risk controller"
    PORTFOLIO_ADVISER = "portfolio adviser"
    EXECUTION_OPERATOR = "execution operator"


class Task(Enum):
    ASSIGN_ROLES = "assign roles"
    WRITE_REPORT = "write desk report"
    GATHER_MARKET_DATA = "gather market data"
    PROPOSE_HYPOTHESES = "propose hypotheses"
    RUN_BACKTEST = "run backtest"
    VALIDATE = "validate backtest"
    ASSESS_RISK = "assess risk"
    ALLOCATE = "allocate paper portfolio"
    EXECUTE_PAPER = "execute paper fills"


ALLOWED_TASKS: dict[Role, frozenset[Task]] = {
    Role.DESK_LEAD: frozenset({Task.ASSIGN_ROLES, Task.WRITE_REPORT}),
    Role.MARKET_INTELLIGENCE: frozenset({Task.GATHER_MARKET_DATA}),
    Role.QUANT_RESEARCHER: frozenset({Task.PROPOSE_HYPOTHESES, Task.RUN_BACKTEST}),
    Role.VALIDATOR: frozenset({Task.VALIDATE}),
    Role.RISK_CONTROLLER: frozenset({Task.ASSESS_RISK}),
    Role.PORTFOLIO_ADVISER: frozenset({Task.ALLOCATE}),
    Role.EXECUTION_OPERATOR: frozenset({Task.EXECUTE_PAPER}),
}


@dataclass(frozen=True)
class Agent:
    """A desk member. The role is fixed at construction and bounds what it may do."""

    agent_id: str
    role: Role

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise TypeError(f"agent {self.agent_id!r} needs a Role, got {self.role!r}")

    def can(self, task: Task) -> bool:
        return task in ALLOWED_TASKS[self.role]

    def require(self, task: Task) -> None:
        if not self.can(task):
            raise RoleViolation(
                f"{self.agent_id} ({self.role.value}) may not '{task.value}'; "
                f"allowed: {sorted(t.value for t in ALLOWED_TASKS[self.role])}"
            )
