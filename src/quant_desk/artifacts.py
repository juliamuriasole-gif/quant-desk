"""Artifacts passed between agents. Each carries provenance and a content hash."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from quant_desk.roles import Agent, Role


@dataclass(frozen=True)
class Provenance:
    """Who made an artifact, from which inputs, and every agent upstream of it."""

    author_id: str
    role: Role
    parents: tuple[str, ...]  # content hashes of input artifacts
    chain: frozenset[str]  # ids of every agent that contributed, including the author

    @classmethod
    def of(cls, author: Agent, *parents: Artifact) -> Provenance:
        chain = frozenset({author.agent_id}).union(*(p.provenance.chain for p in parents))
        return cls(author.agent_id, author.role, tuple(p.content_hash for p in parents), chain)


def _canon(value: Any) -> Any:
    if isinstance(value, pd.DataFrame | pd.Series):
        return hashlib.sha256(value.to_csv().encode()).hexdigest()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, frozenset):
        return sorted(_canon(v) for v in value)
    if isinstance(value, tuple | list):
        return [_canon(v) for v in value]
    if dataclasses.is_dataclass(value):
        return {f.name: _canon(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return value


@dataclass(frozen=True, eq=False)
class Artifact:
    provenance: Provenance

    @property
    def kind(self) -> str:
        return type(self).__name__

    @property
    def content_hash(self) -> str:
        payload = json.dumps({"kind": self.kind, **_canon(self)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class Verdict(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, eq=False)
class MarketData(Artifact):
    prices: pd.DataFrame  # index: dates, columns: assets, values: close
    source: str


@dataclass(frozen=True, eq=False)
class Hypothesis(Artifact):
    name: str
    family: str
    params: tuple[tuple[str, int], ...]
    rationale: str


@dataclass(frozen=True, eq=False)
class BacktestResult(Artifact):
    hypothesis: str
    positions: pd.DataFrame  # target position per asset, decided at that bar's close
    returns: pd.Series  # net daily strategy returns
    cost_bps: float


@dataclass(frozen=True, eq=False)
class ValidationReport(Artifact):
    subject_hash: str
    verdict: Verdict
    oos_start: str
    is_sharpe: float
    oos_sharpe: float
    deflated_sharpe: float
    n_trials: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class RiskDecision(Artifact):
    subject_hash: str
    approved: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class Allocation(Artifact):
    as_of: str
    approvals: tuple[str, ...]  # hashes of the strategy RiskDecisions relied on
    strategy_weights: tuple[tuple[str, float], ...]
    asset_weights: tuple[tuple[str, float], ...]

    @property
    def gross(self) -> float:
        return sum(abs(w) for _, w in self.asset_weights)


@dataclass(frozen=True, eq=False)
class PaperFill(Artifact):
    allocation_hash: str
    as_of: str
    asset: str
    side: str
    quantity: float
    price: float
    notional: float
    weight: float
