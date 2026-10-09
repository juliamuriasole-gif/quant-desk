from quant_desk.cli import main
from quant_desk.desk import run_desk
from quant_desk.market import synthetic_prices


def test_full_run_is_deterministic_for_a_fixed_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_desk(seed=7, out_dir=a)
    run_desk(seed=7, out_dir=b)
    for name in ("report.md", "ledger.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    run_desk(seed=7, out_dir=a)  # re-running into the same dir must not duplicate fills
    assert (a / "ledger.jsonl").read_bytes() == (b / "ledger.jsonl").read_bytes()


def test_roles_are_assigned_before_any_task(tmp_path):
    run = run_desk(seed=7, out_dir=tmp_path)
    actions = [e.action for e in run.constitution.audit]
    first_task = actions.index("produced")
    assert actions[:first_task].count("assigned role") == 6
    assert len({a.role for a in run.staff}) == 7


def test_report_has_required_sections_and_paper_ledger(tmp_path):
    run = run_desk(seed=7, out_dir=tmp_path)
    report = (tmp_path / "report.md").read_text()
    for section in (
        "## Hypotheses tested",
        "## Validation verdicts",
        "## Risk decisions",
        "## Final paper allocation",
        "## Honesty",
        "## Audit log",
        "paper only",
    ):
        assert section in report
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == len(run.fills)
    assert all('"paper": true' in line for line in lines)


def test_cli_run_with_csv(tmp_path, capsys):
    prices = synthetic_prices(1, days=400)
    long = prices.stack().rename("close").reset_index().rename(columns={"level_1": "asset"})
    csv = tmp_path / "prices.csv"
    long.to_csv(csv, index=False)
    assert main(["run", "--csv", str(csv), "--out", str(tmp_path / "out")]) == 0
    assert "csv:prices.csv" in (tmp_path / "out" / "report.md").read_text()
    assert "hypotheses tested: 8" in capsys.readouterr().out
