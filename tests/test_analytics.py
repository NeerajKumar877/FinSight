"""
FinSight - Independent Verification and Calculation Unit Test Suite
Verifies all analytical metrics using independently computed mathematical examples.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.financial_analytics import (
    load_financial_data,
    calculate_financial_summary
)
from src.portfolio_analytics import (
    load_price_data,
    calculate_daily_returns,
    normalize_weights,
    calculate_portfolio_daily_returns,
    calculate_cumulative_returns,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_correlation_matrix
)
from src.risk_analytics import (
    calculate_historical_var,
    calculate_expected_shortfall,
    calculate_stress_test
)
from src.recommendations import generate_business_recommendations


# =====================================================================
# 1. Financial Analytics Unit Tests
# =====================================================================

def test_financial_metrics_manual():
    """Verifies revenue, expense, profit, and margin calculations on a known toy dataset."""
    sample_data = {
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "month_name": ["Jan 2024", "Feb 2024", "Mar 2024"],
        "rev_wealth_management": [40.0, 50.0, 60.0],
        "rev_asset_management": [30.0, 30.0, 40.0],
        "rev_investment_advisory": [20.0, 15.0, 20.0],
        "rev_treasury_services": [10.0, 5.0, 10.0],
        "total_revenue": [100.0, 100.0, 130.0],
        "exp_personnel": [35.0, 35.0, 40.0],
        "exp_technology": [15.0, 15.0, 15.0],
        "exp_operations": [10.0, 10.0, 10.0],
        "exp_compliance_risk": [5.0, 5.0, 5.0],
        "exp_marketing": [5.0, 5.0, 5.0],
        "total_expenses": [70.0, 70.0, 75.0],
        "net_profit": [30.0, 30.0, 55.0],
        "profit_margin_pct": [30.0, 30.0, 42.31],
        "mom_revenue_growth_pct": [0.0, 0.0, 30.0]
    }
    df = pd.DataFrame(sample_data)
    summary = calculate_financial_summary(df)

    # Total revenue = 100 + 100 + 130 = 330
    assert summary["total_revenue_cumulative"] == 330.0
    # Total expenses = 70 + 70 + 75 = 215
    assert summary["total_expenses_cumulative"] == 215.0
    # Total net profit = 330 - 215 = 115
    assert summary["total_net_profit_cumulative"] == 115.0
    # Overall profit margin = (115 / 330) * 100 = 34.848... -> 34.85%
    assert round(summary["overall_profit_margin_pct"], 2) == 34.85
    # Wealth Management share = (150 / 330) * 100 = 45.45%
    assert summary["segment_shares_pct"]["Wealth Management"] == 45.45


# =====================================================================
# 2. Portfolio Analytics Unit Tests
# =====================================================================

def test_weight_normalization():
    """Verifies that weights properly normalize to 1.0 even if input does not sum to 1."""
    raw_weights = {"equities": 40, "bonds": 30, "gold": 15, "fx": 15}
    norm = normalize_weights(raw_weights)
    assert sum(norm.values()) == pytest.approx(1.0, 1e-6)
    assert norm["equities"] == 0.40
    assert norm["bonds"] == 0.30

    # Test all-zero fallback
    zero_weights = {"equities": 0, "bonds": 0, "gold": 0, "fx": 0}
    fallback = normalize_weights(zero_weights)
    assert fallback["equities"] == 0.25
    assert sum(fallback.values()) == pytest.approx(1.0, 1e-6)


def test_weighted_portfolio_returns_manual():
    """
    Verifies that weighted daily portfolio return matches exact pencil-and-paper calculation.
    Asset returns: Equities +2%, Bonds -1%, Gold +1%, FX 0%
    Weights: 40%, 30%, 15%, 15%
    Expected: (0.40 * 0.02) + (0.30 * -0.01) + (0.15 * 0.01) + (0.15 * 0.00)
            = 0.008 - 0.003 + 0.0015 + 0.000 = 0.0065 (0.65%)
    """
    returns_df = pd.DataFrame({
        "equities": [0.02],
        "bonds": [-0.01],
        "gold": [0.01],
        "fx": [0.00]
    })
    weights = {"equities": 0.40, "bonds": 0.30, "gold": 0.15, "fx": 0.15}
    port_ret = calculate_portfolio_daily_returns(returns_df, weights)
    assert port_ret.iloc[0] == pytest.approx(0.0065, 1e-6)


def test_cumulative_compounding_manual():
    """
    Day 1: +10% (0.10), Day 2: -10% (-0.10)
    Cumulative = (1 + 0.10) * (1 - 0.10) - 1 = 1.10 * 0.90 - 1 = 0.99 - 1 = -0.01 (-1.0%)
    """
    series = pd.Series([0.10, -0.10])
    cum = calculate_cumulative_returns(series)
    assert cum.iloc[0] == pytest.approx(0.10, 1e-6)
    assert cum.iloc[1] == pytest.approx(-0.01, 1e-6)


def test_annualized_volatility_manual():
    """
    Standard deviation of [0.01, -0.01, 0.01, -0.01]:
    Mean = 0.0
    Variance (ddof=1) = [0.01^2 + (-0.01)^2 + 0.01^2 + (-0.01)^2] / (4 - 1) = 0.0004 / 3
    Std = sqrt(0.0004 / 3) = 0.01154700538
    Annualized Vol = Std * sqrt(252) = 0.01154700538 * 15.874507866 = 0.183303028
    """
    series = pd.Series([0.01, -0.01, 0.01, -0.01])
    vol = calculate_annualized_volatility(series, trading_days=252)
    expected_vol = math.sqrt(0.0004 / 3.0) * math.sqrt(252)
    assert vol == pytest.approx(expected_vol, 1e-5)


def test_sharpe_ratio_manual():
    """
    Annualized return = 12% (0.12), Volatility = 18% (0.18), Risk-free rate = 3% (0.03)
    Sharpe = (0.12 - 0.03) / 0.18 = 0.09 / 0.18 = 0.50
    """
    sharpe = calculate_sharpe_ratio(0.12, 0.18, risk_free_rate=0.03)
    assert sharpe == pytest.approx(0.50, 1e-6)


# =====================================================================
# 3. Market Risk Unit Tests (VaR, Expected Shortfall, Stress Testing)
# =====================================================================

def test_historical_var_manual():
    """
    Construct 100 uniformly spaced daily returns from -0.050 to +0.049.
    For 100 observations at 95% confidence (5th percentile):
    Cutoff return should be close to -0.045 to -0.046.
    VaR reported as a positive loss.
    """
    # 100 returns sorted ascending
    returns = pd.Series(np.linspace(-0.05, 0.049, 100))
    portfolio_val = 10_000_000.0

    var_result = calculate_historical_var(returns, confidence_level=0.95, portfolio_value=portfolio_val)

    # 5th percentile of np.linspace(-0.05, 0.049, 100)
    expected_5th = float(np.percentile(returns, 5.0))
    expected_var_pct = -expected_5th * 100.0
    expected_var_dollar = -expected_5th * portfolio_val

    assert var_result["var_pct"] == pytest.approx(expected_var_pct, 1e-2)
    assert var_result["var_dollar"] == pytest.approx(expected_var_dollar, 1.0)
    assert var_result["var_pct"] > 0.0  # Positive loss convention


def test_expected_shortfall_manual():
    """
    Verifies that Expected Shortfall is the exact arithmetic mean of tail losses
    and strictly greater than or equal to VaR.
    """
    # Known 20 returns: 5 negative tail losses + 15 positive/neutral returns
    # Tail returns (worst 5% of 100 observations or worst of 20 with 5% threshold):
    # Let's test with 20 values where 5% corresponds to 1 worst observation, or 100 values.
    # With 100 values:
    returns = pd.Series(np.linspace(-0.05, 0.049, 100))
    portfolio_val = 10_000_000.0

    var_res = calculate_historical_var(returns, confidence_level=0.95, portfolio_value=portfolio_val)
    es_res = calculate_expected_shortfall(returns, confidence_level=0.95, portfolio_value=portfolio_val)

    # Mathematical property: Expected Shortfall is strictly >= VaR
    assert es_res["es_pct"] >= var_res["var_pct"]
    assert es_res["es_dollar"] >= var_res["var_dollar"]

    # Verify tail mean manually
    tail_returns = returns[returns <= var_res["cutoff_return"]]
    expected_tail_mean = float(np.mean(tail_returns))
    expected_es_pct = -expected_tail_mean * 100.0
    assert es_res["es_pct"] == pytest.approx(expected_es_pct, 1e-2)


def test_stress_testing_manual():
    """
    Prompt requirement shocks:
    - Equities: -10% (-0.10)
    - Bonds: -5% (-0.05)
    - Gold: -8% (-0.08)
    - FX: -5% (-0.05)

    Weights: 40% Equities, 30% Bonds, 15% Gold, 15% FX
    Portfolio value: $10,000,000

    Expected return:
    (0.40 * -0.10) + (0.30 * -0.05) + (0.15 * -0.08) + (0.15 * -0.05)
    = -0.0400 - 0.0150 - 0.0120 - 0.0075 = -0.0745 (-7.45%)

    Expected dollar loss = $10,000,000 * 0.0745 = $745,000.
    """
    weights = {"equities": 0.40, "bonds": 0.30, "gold": 0.15, "fx": 0.15}
    shocks = {"equities": -0.10, "bonds": -0.05, "gold": -0.08, "fx": -0.05}
    portfolio_val = 10_000_000.0

    stress = calculate_stress_test(weights, shocks=shocks, portfolio_value=portfolio_val)

    assert stress["portfolio_loss_pct"] == 7.45
    assert stress["portfolio_loss_dollar"] == 745_000.0
    assert stress["portfolio_value_post_shock"] == 9_255_000.0

    # Verify asset dollar contributions:
    breakdown = {item["asset_key"]: item for item in stress["asset_breakdown"]}
    assert breakdown["equities"]["dollar_loss"] == 400_000.0
    assert breakdown["bonds"]["dollar_loss"] == 150_000.0
    assert breakdown["gold"]["dollar_loss"] == 120_000.0
    assert breakdown["fx"]["dollar_loss"] == 75_000.0

    # Total contribution sum must equal total loss
    total_asset_loss = sum(item["dollar_loss"] for item in stress["asset_breakdown"])
    assert total_asset_loss == 745_000.0


# =====================================================================
# 4. Integration Tests with Synthetic Datasets
# =====================================================================

def test_synthetic_datasets_integration():
    """Validates real load, calculations, and consistency on the generated synthetic CSV files."""
    fin_df = load_financial_data("data/synthetic_financial_performance.csv")
    assert len(fin_df) == 24
    assert fin_df["data_label"].iloc[0] == "SYNTHETIC"

    fin_summary = calculate_financial_summary(fin_df)
    assert fin_summary["total_revenue_cumulative"] > 0
    assert fin_summary["overall_profit_margin_pct"] > 0

    price_df = load_price_data("data/synthetic_portfolio_prices.csv")
    assert len(price_df) == 504
    assert price_df["data_label"].iloc[0] == "SYNTHETIC"

    returns_df = calculate_daily_returns(price_df)
    assert len(returns_df) == 503

    weights = {"equities": 0.40, "bonds": 0.30, "gold": 0.15, "fx": 0.15}
    port_ret = calculate_portfolio_daily_returns(returns_df, weights)
    assert len(port_ret) == 503

    var_res = calculate_historical_var(port_ret, confidence_level=0.95, portfolio_value=10_000_000)
    es_res = calculate_expected_shortfall(port_ret, confidence_level=0.95, portfolio_value=10_000_000)

    assert var_res["var_pct"] > 0
    assert es_res["es_pct"] >= var_res["var_pct"]

    corr = calculate_correlation_matrix(returns_df)
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr), 1.0)


def test_recommendation_engine_rule_triggers():
    """Verifies that the recommendation engine deterministically triggers alerts on thresholds."""
    # Mock portfolio with extreme 60% equity concentration and high volatility
    port_summary = {
        "weights": {"equities": 0.60, "bonds": 0.15, "gold": 0.15, "fx": 0.10},
        "annualized_volatility_pct": 16.5,
        "annualized_return_pct": 10.2,
        "sharpe_ratio": 0.44
    }
    risk_summary = {
        "var_pct": 2.1,
        "es_pct": 2.9,
        "es_dollar": 290_000
    }
    stress_summary = {
        "portfolio_loss_pct": 8.5,
        "portfolio_loss_dollar": 850_000,
        "asset_breakdown": [
            {"asset_name": "Equities", "dollar_loss": 600_000, "contribution_to_total_loss_pct": 70.6}
        ]
    }
    financial_summary = {
        "latest_profit_margin_pct": 28.0,
        "overall_profit_margin_pct": 33.0,
        "expense_shares_pct": {"Personnel & Talent": 50.0}
    }
    asset_table = [
        {"Short Name": "FX Basket", "Sharpe Ratio": 0.15, "Annualized Return (%)": 2.1, "Annualized Volatility (%)": 7.8},
        {"Short Name": "Equities", "Sharpe Ratio": 0.60, "Annualized Return (%)": 11.0, "Annualized Volatility (%)": 16.5}
    ]

    recs = generate_business_recommendations(port_summary, risk_summary, stress_summary, financial_summary, asset_table)

    assert len(recs) >= 3
    # Check that high concentration is flagged
    rec_titles = [r["title"] for r in recs]
    assert any("Concentration in Equities" in t for t in rec_titles)
    # Check that volatility warning is triggered
    assert any("Volatility" in t for t in rec_titles)
    # Check that metric evidence cites actual numbers
    assert any("60.0%" in r["metric_evidence"] for r in recs)
