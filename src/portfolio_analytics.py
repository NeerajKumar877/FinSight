"""
FinSight - Portfolio Analytics Engine (Module 2)
Computes asset returns, weighted portfolio performance, annualized metrics,
and multi-asset correlation matrices.
"""

import pandas as pd
import numpy as np


ASSET_NAMES = {
    "equities": "Equities (Global Stock Index)",
    "bonds": "Government Bonds (10Y Sovereign)",
    "gold": "Gold (Precious Metals)",
    "fx": "Foreign Exchange (FX Basket)"
}

ASSET_SHORT_LABELS = {
    "equities": "Equities",
    "bonds": "Govt Bonds",
    "gold": "Gold",
    "fx": "FX Basket"
}


def load_price_data(filepath="data/synthetic_portfolio_prices.csv"):
    """Loads and parses synthetic daily asset price data."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_daily_returns(prices_df):
    """
    Computes daily simple returns for each asset:
    R_{i,t} = (P_{i,t} - P_{i,t-1}) / P_{i,t-1}
    """
    assets = ["equities", "bonds", "gold", "fx"]
    returns_df = pd.DataFrame()
    returns_df["date"] = prices_df["date"].iloc[1:].reset_index(drop=True)

    for asset in assets:
        returns_df[asset] = prices_df[asset].pct_change().dropna().reset_index(drop=True)

    return returns_df


def normalize_weights(weights):
    """
    Ensures weights are a valid probability distribution summing to 1.0.
    weights: dict of {asset: weight_value}
    """
    assets = ["equities", "bonds", "gold", "fx"]
    raw_sum = sum(weights.get(a, 0.0) for a in assets)

    if raw_sum <= 0:
        # Equal weights fallback
        return {a: 0.25 for a in assets}

    return {a: float(weights.get(a, 0.0)) / raw_sum for a in assets}


def calculate_portfolio_daily_returns(returns_df, weights):
    """
    Computes the daily portfolio return as the weighted sum of individual asset returns:
    R_{p,t} = w_equities * R_{eq,t} + w_bonds * R_{bd,t} + w_gold * R_{gd,t} + w_fx * R_{fx,t}
    """
    norm_w = normalize_weights(weights)
    portfolio_ret = (
        norm_w["equities"] * returns_df["equities"] +
        norm_w["bonds"] * returns_df["bonds"] +
        norm_w["gold"] * returns_df["gold"] +
        norm_w["fx"] * returns_df["fx"]
    )
    return portfolio_ret


def calculate_cumulative_returns(daily_returns_series):
    """
    Calculates the compound growth of 1 dollar invested over time:
    Cumulative Return_t = prod_{s=1}^t (1 + R_s) - 1
    """
    return (1.0 + daily_returns_series).cumprod() - 1.0


def calculate_annualized_return(daily_returns_series, trading_days=252):
    """
    Annualized return calculated using compound annual growth rate (CAGR):
    CAGR = (1 + Total_Return) ** (trading_days / N) - 1
    """
    n = len(daily_returns_series)
    if n == 0:
        return 0.0
    total_cum_ret = (1.0 + daily_returns_series).prod() - 1.0
    # Avoid complex numbers if total return <= -1.0
    if total_cum_ret <= -1.0:
        return -1.0
    cagr = ((1.0 + total_cum_ret) ** (trading_days / n)) - 1.0
    return float(cagr)


def calculate_annualized_volatility(daily_returns_series, trading_days=252):
    """
    Annualized volatility calculated from sample standard deviation:
    sigma_annual = std(daily_returns, ddof=1) * sqrt(trading_days)
    """
    if len(daily_returns_series) < 2:
        return 0.0
    daily_std = float(daily_returns_series.std(ddof=1))
    return float(daily_std * np.sqrt(trading_days))


def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate=0.03):
    """
    Sharpe Ratio = (Annualized Return - Risk Free Rate) / Annualized Volatility
    Uses standard benchmark risk-free rate of 3.0% (0.03).
    """
    if annualized_volatility <= 1e-8:
        return 0.0
    return float((annualized_return - risk_free_rate) / annualized_volatility)


def calculate_correlation_matrix(returns_df):
    """
    Computes the 4x4 Pearson correlation matrix between asset return series.
    """
    assets = ["equities", "bonds", "gold", "fx"]
    corr = returns_df[assets].corr(method="pearson")
    corr.rename(index=ASSET_SHORT_LABELS, columns=ASSET_SHORT_LABELS, inplace=True)
    return corr


def calculate_asset_summary_table(returns_df, risk_free_rate=0.03):
    """
    Computes standalone metrics for each asset class:
    - Cumulative Return (%)
    - Annualized Return (%)
    - Annualized Volatility (%)
    - Sharpe Ratio
    """
    assets = ["equities", "bonds", "gold", "fx"]
    rows = []

    for a in assets:
        series = returns_df[a]
        cum_ret = (1.0 + series).prod() - 1.0
        ann_ret = calculate_annualized_return(series)
        ann_vol = calculate_annualized_volatility(series)
        sharpe = calculate_sharpe_ratio(ann_ret, ann_vol, risk_free_rate)

        rows.append({
            "Asset": ASSET_NAMES[a],
            "Short Name": ASSET_SHORT_LABELS[a],
            "Cumulative Return (%)": round(cum_ret * 100, 2),
            "Annualized Return (%)": round(ann_ret * 100, 2),
            "Annualized Volatility (%)": round(ann_vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 2)
        })

    return pd.DataFrame(rows)


def calculate_portfolio_overview(returns_df, weights, risk_free_rate=0.03):
    """
    Computes full performance metrics for the weighted portfolio.
    """
    norm_w = normalize_weights(weights)
    port_ret = calculate_portfolio_daily_returns(returns_df, norm_w)

    cum_ret = (1.0 + port_ret).prod() - 1.0
    ann_ret = calculate_annualized_return(port_ret)
    ann_vol = calculate_annualized_volatility(port_ret)
    sharpe = calculate_sharpe_ratio(ann_ret, ann_vol, risk_free_rate)

    return {
        "weights": norm_w,
        "daily_returns": port_ret,
        "cumulative_return_pct": round(cum_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "annualized_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trading_days": len(port_ret)
    }
