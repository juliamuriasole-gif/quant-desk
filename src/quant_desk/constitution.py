"""The desk constitution, enforced in code.

1. Give every agent a role before you give it a task (see `require_role`).
2. No agent grades its own work (see `assert_independent`).
3. No lead overrides the risk limits: limits are frozen, vetoes are final, and
   execution only accepts allocations carrying an approving RiskDecision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, TypeVar

from quant_desk.artifacts import Allocation, Artifact, RiskDecision
from quant_desk.errors import ConstitutionViolation, RoleViolation
from quant_desk.roles import Agent, Role, Task

A = TypeVar("A", bound=Artifact)


@dataclass(frozen=True)
class RiskLimits:
    max_gross_exposure: float = 1.0
    max_asset_weight: float = 0.30
    max_oos_drawdown: float = 0.20
    min_oos_sharpe: float = 0.30

    def __post_init__(self) -> None:
        if not 0 < self.max_asset_weight <= self.max_gross_exposure:
            raise ValueError("need 0 < max_asset_weight <= max_gross_exposure")
        if not 0 < self.max_oos_drawdown < 1:
            raise ValueError("need 0 < max_oos_drawdown < 1")


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    actor_id: str
    role: Role
    action: str
    artifact: str  # "<Kind>:<hash12>" or "-"
    note: str


class Constitution:
    """Holds the frozen risk limits and the audit log; every handoff goes through it."""

    def __init__(self, limits: RiskLimits) -> None:
        object.__setattr__(self, "_limits", limits)
        object.__setattr__(self, "audit", [])

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def __setattr__(self, name: str, value: Any) -> None:
        raise ConstitutionViolation(f"the constitution is fixed once the desk starts ({name!r})")

    def record(
        self, actor: Agent, action: str, artifact: Artifact | None = None, note: str = ""
    ) -> None:
        label = f"{artifact.kind}:{artifact.content_hash[:12]}" if artifact else "-"
        self.audit.append(
            AuditEntry(len(self.audit) + 1, actor.agent_id, actor.role, action, label, note)
        )

    def publish(self, artifact: A, actor: Agent, note: str = "") -> A:
        self.record(actor, "produced", artifact, note)
        return artifact

    def violation(self, actor: Agent, message: str) -> ConstitutionViolation:
        self.record(actor, "VIOLATION BLOCKED", note=message)
        return ConstitutionViolation(message)

    def require_role(self, actor: Agent, task: Task) -> None:
        try:
            actor.require(task)
        except RoleViolation as exc:
            self.record(actor, "VIOLATION BLOCKED", note=str(exc))
            raise

    def assert_independent(self, checker: Agent, artifact: Artifact) -> None:
        if checker.agent_id in artifact.provenance.chain:
            raise self.violation(
                checker,
                f"{checker.agent_id} cannot check {artifact.kind}:{artifact.content_hash[:12]}: "
                "its own work is in the provenance chain (no agent grades its own work)",
            )

    def override_veto(self, actor: Agent, decision: RiskDecision) -> NoReturn:
        """Any attempt to overturn a risk decision is logged and refused, whoever asks."""
        raise self.violation(
            actor,
            f"{actor.agent_id} ({actor.role.value}) tried to override risk decision "
            f"{decision.content_hash[:12]}; risk vetoes cannot be overridden",
        )

    def authorise_execution(
        self, actor: Agent, allocation: Allocation, decision: RiskDecision | None
    ) -> None:
        if decision is None:
            raise self.violation(actor, "allocation has no risk decision attached")
        if decision.provenance.role is not Role.RISK_CONTROLLER:
            raise self.violation(actor, "risk decision was not issued by a risk controller")
        if decision.subject_hash != allocation.content_hash:
            raise self.violation(actor, "risk decision is for a different allocation")
        if not decision.approved:
            raise self.violation(actor, "allocation was vetoed by the risk controller")
