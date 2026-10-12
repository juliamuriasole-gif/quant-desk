"""Markdown desk report written by the desk lead."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_desk.artifacts import Verdict
from quant_desk.research import max_drawdown

if TYPE_CHECKING:
    from quant_desk.desk import DeskRun

DISCLAIMER = (
    "> **Research / paper only.** Nothing here is investment advice. No broker or exchange is "
    "connected; fills below exist only in a local paper ledger."
)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def render(run: DeskRun) -> str:
    c = run.constitution
    prices = run.market.prices
    survivors = [d for d in run.strategy_decisions if d.approved]
    passed = [r for r in run.reports if r.verdict is Verdict.PASS]
    out = ["# Quant desk report", "", DISCLAIMER, ""]

    out += ["## Desk", ""]
    out += _table(["agent", "role"], [[a.agent_id, a.role.value] for a in run.staff])
    out += ["", f"Risk limits (frozen at desk start): `{c.limits}`", ""]
    out += [
        f"Data: {run.market.source}; {prices.shape[1]} assets "
        f"({', '.join(prices.columns)}), {len(prices)} bars "
        f"{prices.index[0].date()} to {prices.index[-1].date()}.",
        "",
    ]

    out += ["## Hypotheses tested", ""]
    rows = []
    for res, rep, dec in zip(run.results, run.reports, run.strategy_decisions, strict=True):
        oos = res.returns.loc[rep.oos_start :]
        rows.append(
            [
                res.hypothesis,
                f"{rep.is_sharpe:.2f}",
                f"{rep.oos_sharpe:.2f}",
                f"{max_drawdown(oos):.1%}",
                f"{rep.deflated_sharpe:.3f}",
                rep.verdict.value,
                "approved" if dec.approved else "VETO",
            ]
        )
    out += _table(
        ["hypothesis", "IS Sharpe", "OOS Sharpe", "OOS max DD", "DSR", "validator", "risk"], rows
    )
    out += [
        "",
        f"In-sample / out-of-sample split at {run.reports[0].oos_start}; "
        f"costs {run.results[0].cost_bps:g} bps per unit turnover.",
        "",
    ]

    out += ["## Validation verdicts", ""]
    for res, rep in zip(run.results, run.reports, strict=True):
        out += [f"**{res.hypothesis}**: {rep.verdict.value}", ""]
        out += [f"- {reason}" for reason in rep.reasons] + [""]

    out += ["## Risk decisions", ""]
    for res, dec in zip(run.results, run.strategy_decisions, strict=True):
        status = "approved" if dec.approved else "VETOED"
        out += [f"- **{res.hypothesis}**: {status}: " + "; ".join(dec.reasons)]
    status = "approved" if run.allocation_decision.approved else "VETOED"
    out += [f"- **final allocation**: {status}: " + "; ".join(run.allocation_decision.reasons), ""]

    out += ["## Constitution drill", ""]
    out += [
        f"The desk lead tried to override a risk veto. Result: blocked. `{run.drill}`"
        if run.drill
        else "No veto was issued, so no override drill ran.",
        "",
    ]

    out += ["## Final paper allocation", ""]
    if run.allocation.asset_weights:
        out += [f"As of {run.allocation.as_of}; gross exposure {run.allocation.gross:.2%}.", ""]
        strategies = [[n, f"{w:.2%}"] for n, w in run.allocation.strategy_weights]
        assets = [[a, f"{w:+.2%}"] for a, w in run.allocation.asset_weights]
        out += [*_table(["strategy", "weight"], strategies), ""]
        out += [*_table(["asset", "weight"], assets), ""]
    else:
        out += ["All cash: no strategy survived validation and risk.", ""]
    out += ["### Paper fills (local ledger only)", ""]
    out += (
        _table(
            ["asset", "side", "quantity", "price", "notional"],
            [
                [f.asset, f.side, f"{f.quantity:,.4f}", f"{f.price:,.4f}", f"{f.notional:,.2f}"]
                for f in run.fills
            ],
        )
        if run.fills
        else ["None."]
    )
    out += [""]

    out += ["## Honesty", ""]
    out += [
        f"- Hypotheses tried: **{len(run.hypotheses)}**. Passed validation: **{len(passed)}**. "
        f"Survived risk: **{len(survivors)}**.",
        "- The deflated Sharpe ratio already penalises for the number of trials, but every "
        "variant ever tried should count; this run only knows about its own grid.",
        "- A surviving backtest is **not** evidence of future returns. "
        + (
            "The data is synthetic and was generated with persistent trends, so trend "
            "strategies are expected to look good here."
            if run.market.source.startswith("synth")
            else "Results depend entirely on the supplied CSV and its quality."
        ),
        "",
    ]

    out += ["## Audit log", "", "Every handoff, in order: who produced what, who checked it.", ""]
    out += _table(
        ["#", "actor", "role", "action", "artifact", "note"],
        [
            [
                str(e.seq),
                e.actor_id,
                e.role.value,
                e.action,
                f"`{e.artifact}`",
                e.note.replace("|", "/"),
            ]
            for e in c.audit
        ],
    )
    return "\n".join(out) + "\n"
