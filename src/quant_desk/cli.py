"""Command line: `quant-desk run --seed 7 --out output/` (or `quant-desk demo`)."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_desk.desk import run_desk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quant-desk",
        description="Offline, role-separated quant research desk. Research and paper only.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the desk once")
    run.add_argument("--seed", type=int, default=7, help="seed for synthetic data (default 7)")
    run.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    run.add_argument("--csv", type=Path, help="long-format CSV with columns date, asset, close")
    run.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit turnover, bps")
    sub.add_parser("demo", help="same as: run --seed 7 --out output")
    args = parser.parse_args(argv)

    if args.command == "demo":
        args = run.parse_args([])
    result = run_desk(seed=args.seed, out_dir=args.out, csv_path=args.csv, cost_bps=args.cost_bps)
    survivors = sum(d.approved for d in result.strategy_decisions)
    print(f"hypotheses tested: {len(result.hypotheses)}, survived validation + risk: {survivors}")
    print(f"paper fills: {len(result.fills)} (local ledger only, no broker)")
    print(f"report: {args.out / 'report.md'}")
    print(f"ledger: {args.out / 'ledger.jsonl'}")
    return 0
