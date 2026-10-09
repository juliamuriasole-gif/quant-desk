"""Execution operator: PAPER ONLY. Turns an approved allocation into fills in a local JSONL
ledger. There is no broker or exchange connection anywhere in this project."""

from __future__ import annotations

import json
from pathlib import Path

from quant_desk.artifacts import Allocation, MarketData, PaperFill, Provenance, RiskDecision
from quant_desk.constitution import Constitution
from quant_desk.roles import Agent, Task


def execute_paper(
    agent: Agent,
    constitution: Constitution,
    allocation: Allocation,
    decision: RiskDecision | None,
    market: MarketData,
    ledger: Path,
    *,
    capital: float = 100_000.0,
) -> tuple[PaperFill, ...]:
    """Rebalance a flat paper book of `capital` to the allocation at the last close.

    The ledger is append-only and idempotent: a fill whose hash is already recorded is skipped.
    """
    constitution.require_role(agent, Task.EXECUTE_PAPER)
    constitution.authorise_execution(agent, allocation, decision)
    constitution.require_published(agent, market)
    last = market.prices.iloc[-1]
    fills = tuple(
        PaperFill(
            Provenance.of(agent, allocation, decision),
            allocation.content_hash,
            allocation.as_of,
            asset,
            "BUY" if w > 0 else "SELL",
            round(abs(w) * capital / float(last[asset]), 4),
            float(last[asset]),
            round(abs(w) * capital, 2),
            w,
        )
        for asset, w in allocation.asset_weights
        if w != 0
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    seen = (
        {json.loads(line)["fill_hash"] for line in ledger.read_text().splitlines()}
        if ledger.exists()
        else set()
    )
    with ledger.open("a") as fh:
        for fill in fills:
            if fill.content_hash not in seen:
                row = {
                    "fill_hash": fill.content_hash,
                    "paper": True,
                    "author": fill.provenance.author_id,
                    "allocation_hash": fill.allocation_hash,
                    "risk_decision_hash": decision.content_hash,
                    "as_of": fill.as_of,
                    "asset": fill.asset,
                    "side": fill.side,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "notional": fill.notional,
                    "weight": fill.weight,
                }
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    for fill in fills:
        constitution.publish(fill, agent, f"PAPER {fill.side} {fill.quantity} {fill.asset}")
    return fills
