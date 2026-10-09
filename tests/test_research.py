import numpy as np
import pandas as pd
import pytest

from quant_desk import research
from quant_desk.artifacts import Verdict
from quant_desk.research import backtest, max_drawdown, sharpe
from quant_desk.roles import Role
from quant_desk.validation import deflated_sharpe, lookahead_free, validate


def test_backtest_matches_hand_computed_example(tiny_market):
    prices = tiny_market.prices  # X: 100, 110, 99, 99
    positions = pd.DataFrame({"X": [1.0, 1.0, -1.0, 0.0]}, index=prices.index)
    # bar 0: enter long from flat, pay 10 bps on 1 unit       -> -0.001
    # bar 1: long held from bar 0 earns +10%, no trade         -> +0.100
    # bar 2: long earns -10%, flip to short costs 2 units      -> -0.100 - 0.002
    # bar 3: short earns 0%, close the short costs 1 unit      -> -0.001
    expected = [-0.001, 0.100, -0.102, -0.001]
    np.testing.assert_allclose(backtest(positions, prices, cost_bps=10).to_numpy(), expected)


def test_backtest_averages_across_assets(tiny_market):
    prices = tiny_market.prices.assign(Y=100.0)
    positions = pd.DataFrame({"X": [1.0, 1.0, 1.0, 1.0], "Y": 0.0}, index=prices.index)
    np.testing.assert_allclose(backtest(positions, prices, 0).to_numpy(), [0, 0.05, -0.05, 0])


def test_signals_are_applied_one_bar_late(tiny_market):
    """A position taken at bar t must not earn bar t's own return."""
    prices = tiny_market.prices
    positions = pd.DataFrame({"X": [0.0, 1.0, 0.0, 0.0]}, index=prices.index)  # long at bar 1 only
    returns = backtest(positions, prices, 0)
    assert returns.iloc[1] == 0.0  # did not capture the +10% into bar 1
    assert returns.iloc[2] == pytest.approx(-0.1)  # earned bar 2's move instead


def test_metrics():
    flat = pd.Series([0.0] * 5)
    assert sharpe(flat) == 0.0 and max_drawdown(flat) == 0.0
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    # losses from the first bar count: the starting capital is a peak
    assert max_drawdown(pd.Series([-0.5, 0.0])) == pytest.approx(0.5)
    assert max_drawdown(pd.Series([-0.1, -0.1, 0.3])) == pytest.approx(0.19)


def test_grid_strategies_are_causal(market, researched):
    hypotheses, _ = researched
    assert all(lookahead_free(h, market.prices) for h in hypotheses)


def test_validator_catches_a_strategy_that_peeks_at_the_future(
    monkeypatch, constitution, staff, market, researched
):
    """A 'perfect' strategy using tomorrow's close must FAIL validation on lookahead."""
    monkeypatch.setitem(
        research.STRATEGIES, "peek", lambda prices: np.sign(prices.shift(-1) / prices - 1)
    )
    researcher = staff[Role.QUANT_RESEARCHER]
    hypotheses, results = researched
    cheat = research.Hypothesis(hypotheses[0].provenance, "peek()", "peek", (), "clairvoyance")
    constitution.publish(cheat, researcher)
    result = research.run_backtest(researcher, constitution, cheat, market, 5.0)
    assert sharpe(result.returns) > 5  # looks spectacular...
    report = validate(
        staff[Role.VALIDATOR], constitution, result, cheat, market, (*results, result)
    )
    assert report.verdict is Verdict.FAIL  # ...and is rejected
    assert "FAIL: signals are causal (truncation test)" in report.reasons


def test_validator_catches_misreported_returns(constitution, staff, market, researched):
    hypotheses, results = researched
    import dataclasses

    inflated = dataclasses.replace(results[0], returns=results[0].returns + 0.001)
    constitution.publish(inflated, staff[Role.QUANT_RESEARCHER])  # the researcher misreports
    report = validate(staff[Role.VALIDATOR], constitution, inflated, hypotheses[0], market, results)
    assert "FAIL: backtest reproduces from the hypothesis and data" in report.reasons


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 1000))
    one, many = deflated_sharpe(returns, 1), deflated_sharpe(returns, 100)
    assert 0 < many < one < 1
    assert deflated_sharpe(pd.Series([0.01, 0.02]), 8) == 0.0  # too short: never passes
