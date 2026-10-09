"""The desk constitution, enforced in code.

1. Give every agent a role before you give it a task (see `enrol` and `require_role`).
2. No agent grades its own work (see `assert_independent`).
3. No lead overrides the risk limits: limits are frozen, vetoes are final, and
   execution only accepts allocations carrying an approving RiskDecision.

Only agents on the roster may act, an agent may only publish artifacts under its own name, and
checks only accept artifacts that were published to this desk unchanged (see `require_published`),
so provenance cannot be forged by constructing or editing an artifact by hand. This is a
discipline for in-process code, not a security boundary: Python code can always reach past it.
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
        object.__setattr__(self, "_audit", [])
        object.__setattr__(self, "_roster", {})  # agent id -> role
        object.__setattr__(self, "_published", {})  # content hash -> artifact

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    @property
    def audit(self) -> tuple[AuditEntry, ...]:
        """Read-only view of the audit log; entries can only be added through the constitution."""
        return tuple(self._audit)

    def __setattr__(self, name: str, value: Any) -> None:
        raise ConstitutionViolation(f"the constitution is fixed once the desk starts ({name!r})")

    def record(
        self, actor: Agent, action: str, artifact: Artifact | None = None, note: str = ""
    ) -> None:
        label = f"{artifact.kind}:{artifact.content_hash[:12]}" if artifact else "-"
        self._audit.append(
            AuditEntry(len(self._audit) + 1, actor.agent_id, actor.role, action, label, note)
        )

    def open_desk(self, lead: Agent) -> None:
        """Put the desk lead on the roster and freeze the limits. Happens once, first."""
        if self._roster:
            raise self.violation(lead, "the desk is already open")
        lead.require(Task.ASSIGN_ROLES)
        self._roster[lead.agent_id] = lead.role
        self.record(lead, "desk opened", note=f"risk limits frozen: {self.limits}")

    def enrol(self, lead: Agent, agent: Agent) -> None:
        """The desk lead gives `agent` its role. Agents not on the roster cannot act."""
        self.require_role(lead, Task.ASSIGN_ROLES)
        if agent.agent_id in self._roster:
            raise self.violation(lead, f"{agent.agent_id} already has a role")
        self._roster[agent.agent_id] = agent.role
        self.record(lead, "assigned role", note=f"{agent.agent_id} -> {agent.role.value}")

    def publish(self, artifact: A, actor: Agent, note: str = "") -> A:
        self._require_rostered(actor)
        prov = artifact.provenance
        if (prov.author_id, prov.role) != (actor.agent_id, actor.role):
            raise self.violation(
                actor,
                f"{actor.agent_id} cannot publish {artifact.kind} authored by "
                f"{prov.author_id} ({prov.role.value})",
            )
        self._published[artifact.content_hash] = artifact
        self.record(actor, "produced", artifact, note)
        return artifact

    def require_published(self, actor: Agent, *artifacts: Artifact) -> None:
        """Refuse artifacts that were never published here or were edited after publishing."""
        for artifact in artifacts:
            if artifact.content_hash not in self._published:
                raise self.violation(
                    actor,
                    f"{artifact.kind}:{artifact.content_hash[:12]} was never published to this "
                    "desk (forged or modified after publishing)",
                )

    def published(self, content_hash: str) -> Artifact | None:
        return self._published.get(content_hash)

    def _require_rostered(self, actor: Agent) -> None:
        if self._roster.get(actor.agent_id) is not actor.role:
            raise self.violation(
                actor, f"{actor.agent_id} ({actor.role.value}) is not on the desk roster"
            )

    def violation(self, actor: Agent, message: str) -> ConstitutionViolation:
        self.record(actor, "VIOLATION BLOCKED", note=message)
        return ConstitutionViolation(message)

    def require_role(self, actor: Agent, task: Task) -> None:
        self._require_rostered(actor)
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
        self.require_published(actor, allocation, decision)
