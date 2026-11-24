"""
FinSight - Market Risk Analytics Engine (Module 3)
Calculates Historical Value at Risk (VaR 95%), Expected Shortfall (CVaR),
and Multi-Asset Scenario Stress Testing.
"""

import numpy as np
import pandas as pd


def calculate_historical_var(portfolio_returns, confidence_level=0.95, portfolio_value=10_000_000):
    """
    Calculates 1-day Historical Value at Risk (VaR) at the specified confidence level.

    Methodology:
    1. Sort historical daily portfolio returns in ascending order.
    2. Determine the percentile cutoff: alpha = (1 - confidence_level). E.g., for 95%, alpha = 0.05.
    3. The return threshold is the 5th percentile of historical daily returns.
    4. Loss convention: Reported as a POSITIVE dollar loss and POSITIVE loss percentage.

    Parameters:
    - portfolio_returns: pandas Series or 1D numpy array of historical daily portfolio returns.
    - confidence_level: float, default 0.95 (95% confidence).
    - portfolio_value: float, nominal portfolio capital in dollars.

    Returns:
    - dict containing:
        - var_pct: VaR as positive percentage (e.g., 1.25 for 1.25% loss)
        - var_dollar: VaR in currency units
        - cutoff_return: raw return at the 5th percentile (negative number, e.g. -0.0125)
        - confidence_level: 0.95
        - explanation: plain-English explanation of the metric
    """
    clean_returns = np.asarray(portfolio_returns.dropna())
    if len(clean_returns) == 0:
        return {
            "var_pct": 0.0,
            "var_dollar": 0.0,
            "cutoff_return": 0.0,
            "confidence_level": confidence_level,
            "explanation": "No return data available."
        }

    alpha = (1.0 - confidence_level) * 100.0  # 5.0 for 95%
    cutoff_return = float(np.percentile(clean_returns, alpha))

    # Loss is the negative of the negative return cutoff
    var_pct = float(max(0.0, -cutoff_return * 100.0))
    var_dollar = float(max(0.0, -cutoff_return * portfolio_value))

    explanation = (
        f"At a {int(confidence_level * 100)}% confidence level over a 1-day holding period, "
        f"the maximum expected loss is {var_pct:.2f}% (${var_dollar:,.0f} on a ${portfolio_value:,.0f} portfolio). "
        f"In other words, on 95% of trading days, daily losses are expected to be smaller than this amount. "
        f"Conversely, on 1 in 20 trading days (5%), losses are expected to equal or exceed ${var_dollar:,.0f}."
    )

    return {
        "var_pct": round(var_pct, 2),
        "var_dollar": round(var_dollar, 2),
        "cutoff_return": cutoff_return,
        "confidence_level": confidence_level,
        "portfolio_value": portfolio_value,
        "explanation": explanation
    }


def calculate_expected_shortfall(portfolio_returns, confidence_level=0.95, portfolio_value=10_000_000):
    """
    Calculates 1-day Expected Shortfall (ES), also known as Conditional VaR (CVaR).

    Methodology:
    1. Identifies the 5th percentile VaR cutoff return.
    2. Isolates all historical returns strictly equal to or worse than this cutoff (the tail).
    3. Calculates the arithmetic mean of these tail returns.
    4. Loss convention: Reported as a POSITIVE dollar loss and POSITIVE loss percentage.
       By definition, ES >= VaR.

    Returns:
    - dict containing:
        - es_pct: Expected Shortfall as positive percentage
        - es_dollar: Expected Shortfall in dollars
        - tail_mean_return: raw average return of tail events (negative number)
        - tail_observations_count: count of historical trading days in the tail
        - explanation: plain-English explanation of the metric
    """
    clean_returns = np.asarray(portfolio_returns.dropna())
    if len(clean_returns) == 0:
        return {
            "es_pct": 0.0,
            "es_dollar": 0.0,
            "tail_mean_return": 0.0,
            "tail_observations_count": 0,
            "explanation": "No return data available."
        }

    alpha = (1.0 - confidence_level) * 100.0
    cutoff_return = float(np.percentile(clean_returns, alpha))

    tail_returns = clean_returns[clean_returns <= cutoff_return]
    tail_count = len(tail_returns)

    if tail_count == 0:
        tail_mean = cutoff_return
    else:
        tail_mean = float(np.mean(tail_returns))

    es_pct = float(max(0.0, -tail_mean * 100.0))
    es_dollar = float(max(0.0, -tail_mean * portfolio_value))

    explanation = (
        f"If the portfolio experiences a tail event beyond the {int(confidence_level * 100)}% VaR threshold "
        f"(i.e., within the worst 5% of historical market days), the average expected loss is "
        f"{es_pct:.2f}% (${es_dollar:,.0f} on a ${portfolio_value:,.0f} portfolio). "
        f"Unlike VaR (which only marks the threshold where tail losses begin), Expected Shortfall answers: "
        f"'When things go wrong, how bad is the average damage in the tail?'"
    )

    return {
        "es_pct": round(es_pct, 2),
        "es_dollar": round(es_dollar, 2),
        "tail_mean_return": tail_mean,
        "tail_observations_count": tail_count,
        "cutoff_return": cutoff_return,
        "portfolio_value": portfolio_value,
        "explanation": explanation
    }


def calculate_comprehensive_risk_summary(portfolio_returns, confidence_level=0.95, portfolio_value=10_000_000):
    """
    Computes both 95% Historical VaR and 95% Expected Shortfall in a single consolidated dictionary.
    """
    var_dict = calculate_historical_var(portfolio_returns, confidence_level, portfolio_value)
    es_dict = calculate_expected_shortfall(portfolio_returns, confidence_level, portfolio_value)
    merged = {**var_dict, **es_dict}
    merged["var_explanation"] = var_dict["explanation"]
    merged["es_explanation"] = es_dict["explanation"]
    return merged


def calculate_stress_test(weights, shocks=None, portfolio_value=10_000_000):
    """
    Performs deterministic scenario stress testing on the multi-asset portfolio.

    Default Shocks (as specified in business requirements):
    - Equities: -10% (-0.10)
    - Government Bonds: -5% (-0.05)
    - Gold: -8% (-0.08)
    - Foreign Exchange: -5% (-0.05)

    Formula:
    Portfolio Return = sum(w_i * Shock_i)
    Portfolio Loss Dollar = Portfolio Value * abs(Portfolio Return)
    """
    assets = ["equities", "bonds", "gold", "fx"]
    asset_display = {
        "equities": "Equities",
        "bonds": "Government Bonds",
        "gold": "Gold",
        "fx": "Foreign Exchange"
    }

    if shocks is None:
        shocks = {
            "equities": -0.10,
            "bonds": -0.05,
            "gold": -0.08,
            "fx": -0.05
        }

    # Normalize weights
    raw_sum = sum(weights.get(a, 0.0) for a in assets)
    norm_w = {a: (weights.get(a, 0.0) / raw_sum if raw_sum > 0 else 0.25) for a in assets}

    breakdown = []
    total_shock_pct = 0.0

    for a in assets:
        w = norm_w[a]
        shk = shocks.get(a, 0.0)
        weighted_shock = w * shk
        total_shock_pct += weighted_shock

        dollar_allocation = w * portfolio_value
        dollar_impact = dollar_allocation * shk

        breakdown.append({
            "asset_key": a,
            "asset_name": asset_display[a],
            "weight_pct": round(w * 100, 2),
            "allocation_dollar": round(dollar_allocation, 2),
            "shock_pct": round(shk * 100, 2),
            "dollar_loss": round(abs(dollar_impact) if dollar_impact < 0 else -dollar_impact, 2),
            "impact_direction": "Loss" if dollar_impact < 0 else "Gain",
            "contribution_to_total_loss_pct": 0.0  # calculated below
        })

    total_loss_dollar = abs(total_shock_pct * portfolio_value) if total_shock_pct < 0 else 0.0
    total_impact_pct = round(total_shock_pct * 100, 2)

    # Compute percentage contribution of each asset to the total loss
    for item in breakdown:
        if total_loss_dollar > 0 and item["impact_direction"] == "Loss":
            item["contribution_to_total_loss_pct"] = round((item["dollar_loss"] / total_loss_dollar) * 100, 2)
        else:
            item["contribution_to_total_loss_pct"] = 0.0

    return {
        "portfolio_loss_pct": round(abs(total_impact_pct), 2),
        "portfolio_impact_raw_pct": total_impact_pct,
        "portfolio_loss_dollar": round(total_loss_dollar, 2),
        "portfolio_value_post_shock": round(portfolio_value + (total_shock_pct * portfolio_value), 2),
        "shocks_applied": {k: round(v * 100, 2) for k, v in shocks.items()},
        "asset_breakdown": breakdown
    }


def get_predefined_stress_scenarios():
    """
    Returns macroeconomic stress scenarios for scenario analysis.
    """
    return {
        "Standard Market Shock (Prompt Baseline)": {
            "equities": -0.10,
            "bonds": -0.05,
            "gold": -0.08,
            "fx": -0.05,
            "description": "Baseline adverse scenario across all four asset classes."
        },
        "Equity Market Crash & Flight-to-Quality": {
            "equities": -0.20,
            "bonds": 0.04,
            "gold": 0.10,
            "fx": -0.03,
            "description": "Severe equity sell-off (-20%) where sovereign bonds and gold act as safe-haven buffers."
        },
        "Stagflation & Rate Hike Pressure": {
            "equities": -0.12,
            "bonds": -0.10,
            "gold": 0.05,
            "fx": -0.04,
            "description": "Simultaneous equity and bond decline triggered by sticky inflation and aggressive central bank tightening."
        },
        "Currency Devaluation & FX Crisis": {
            "equities": -0.15,
            "bonds": -0.06,
            "gold": 0.08,
            "fx": -0.12,
            "description": "Sharp foreign exchange depreciation accompanied by imported inflation and domestic equity pullback."
        }
    }
