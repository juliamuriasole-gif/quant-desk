import pandas as pd
import pytest

from quant_desk.artifacts import Hypothesis, MarketData, Provenance
from quant_desk.constitution import Constitution, RiskLimits
from quant_desk.market import gather_market_data
from quant_desk.research import propose_hypotheses, run_backtest
from quant_desk.roles import Agent, Role


@pytest.fixture
def constitution() -> Constitution:
    return Constitution(RiskLimits())


@pytest.fixture
def staff() -> dict[Role, Agent]:
    return {role: Agent(f"{role.name.lower()}-1", role) for role in Role}


@pytest.fixture
def market(constitution: Constitution, staff: dict[Role, Agent]) -> MarketData:
    return gather_market_data(staff[Role.MARKET_INTELLIGENCE], constitution, seed=3)


@pytest.fixture
def researched(constitution, staff, market) -> tuple[tuple[Hypothesis, ...], ...]:
    researcher = staff[Role.QUANT_RESEARCHER]
    hypotheses = propose_hypotheses(researcher, constitution)
    results = tuple(run_backtest(researcher, constitution, h, market, 5.0) for h in hypotheses)
    return hypotheses, results


@pytest.fixture
def tiny_market(staff: dict[Role, Agent]) -> MarketData:
    prices = pd.DataFrame(
        {"X": [100.0, 110.0, 99.0, 99.0]},
        index=pd.bdate_range("2024-01-01", periods=4, name="date"),
    )
    return MarketData(Provenance.of(staff[Role.MARKET_INTELLIGENCE]), prices, "test")
