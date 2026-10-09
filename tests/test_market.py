import pandas as pd
import pytest

from quant_desk.market import ASSETS, load_csv, synthetic_prices


def test_synthetic_prices_are_deterministic_per_seed():
    a, b, c = synthetic_prices(7, days=200), synthetic_prices(7, days=200), synthetic_prices(8, 200)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert list(a.columns) == list(ASSETS) and (a > 0).all().all() and len(a) == 200


def test_csv_loader_pivots_long_format_and_drops_incomplete_dates(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "date,asset,close\n"
        "2024-01-03,AAA,11\n2024-01-02,AAA,10\n2024-01-02,BBB,20\n"
        "2024-01-03,BBB,21\n2024-01-04,AAA,12\n"  # BBB missing on 01-04
    )
    prices = load_csv(path)
    assert list(prices.columns) == ["AAA", "BBB"]
    assert [str(d.date()) for d in prices.index] == ["2024-01-02", "2024-01-03"]
    assert prices.loc["2024-01-03", "BBB"] == 21.0


@pytest.mark.parametrize(
    "body, message",
    [
        ("date,ticker,close\n2024-01-02,A,1\n", "missing columns"),
        ("date,asset,close\n2024-01-02,A,1\n2024-01-02,A,2\n", "duplicate"),
        ("date,asset,close\n2024-01-02,A,0\n2024-01-03,A,1\n", "positive"),
        ("date,asset,close\n2024-01-02,A,1\n", "two complete dates"),
    ],
)
def test_csv_loader_rejects_bad_files(tmp_path, body, message):
    path = tmp_path / "bad.csv"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_csv(path)
