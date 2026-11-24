"""
FinSight - Financial Performance Analytics Engine (Module 1)
Calculates corporate revenue, operating expenses, profit margins, growth rates,
and segment-wise breakdowns.
"""

import pandas as pd
import numpy as np


def load_financial_data(filepath="data/synthetic_financial_performance.csv"):
    """Loads and formats synthetic financial performance data."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_financial_summary(df):
    """
    Computes key executive financial metrics across the entire 24-month period.
    """
    total_rev = float(df["total_revenue"].sum())
    total_exp = float(df["total_expenses"].sum())
    total_net_profit = float(df["net_profit"].sum())
    avg_margin = float(df["profit_margin_pct"].mean())
    latest_rev = float(df["total_revenue"].iloc[-1])
    latest_profit = float(df["net_profit"].iloc[-1])
    latest_margin = float(df["profit_margin_pct"].iloc[-1])
    latest_mom_growth = float(df["mom_revenue_growth_pct"].iloc[-1])
    avg_mom_growth = float(df["mom_revenue_growth_pct"].iloc[1:].mean())

    # Segment revenue aggregates
    segment_cols = [
        "rev_wealth_management",
        "rev_asset_management",
        "rev_investment_advisory",
        "rev_treasury_services"
    ]
    segment_names = {
        "rev_wealth_management": "Wealth Management",
        "rev_asset_management": "Asset Management",
        "rev_investment_advisory": "Investment Advisory",
        "rev_treasury_services": "Treasury Services"
    }
    segment_totals = {segment_names[c]: float(df[c].sum()) for c in segment_cols}
    segment_shares = {k: round((v / total_rev) * 100, 2) for k, v in segment_totals.items()}

    # Expense category aggregates
    expense_cols = [
        "exp_personnel",
        "exp_technology",
        "exp_operations",
        "exp_compliance_risk",
        "exp_marketing"
    ]
    expense_names = {
        "exp_personnel": "Personnel & Talent",
        "exp_technology": "Technology & Feeds",
        "exp_operations": "Operations & Admin",
        "exp_compliance_risk": "Compliance & Risk",
        "exp_marketing": "Client Acquisition & Mktg"
    }
    expense_totals = {expense_names[c]: float(df[c].sum()) for c in expense_cols}
    expense_shares = {k: round((v / total_exp) * 100, 2) for k, v in expense_totals.items()}

    return {
        "total_revenue_cumulative": round(total_rev, 2),
        "total_expenses_cumulative": round(total_exp, 2),
        "total_net_profit_cumulative": round(total_net_profit, 2),
        "overall_profit_margin_pct": round((total_net_profit / total_rev) * 100, 2),
        "avg_monthly_profit_margin_pct": round(avg_margin, 2),
        "latest_monthly_revenue": round(latest_rev, 2),
        "latest_monthly_profit": round(latest_profit, 2),
        "latest_profit_margin_pct": round(latest_margin, 2),
        "latest_mom_growth_pct": round(latest_mom_growth, 2),
        "avg_mom_growth_pct": round(avg_mom_growth, 2),
        "segment_totals": segment_totals,
        "segment_shares_pct": segment_shares,
        "expense_totals": expense_totals,
        "expense_shares_pct": expense_shares
    }


def get_segment_breakdown_df(df):
    """Reshapes segment revenue into long format for multi-bar / stacked charts."""
    segment_map = {
        "rev_wealth_management": "Wealth Management",
        "rev_asset_management": "Asset Management",
        "rev_investment_advisory": "Investment Advisory",
        "rev_treasury_services": "Treasury Services"
    }
    id_vars = ["date", "month_name"]
    val_vars = list(segment_map.keys())
    melted = df.melt(id_vars=id_vars, value_vars=val_vars, var_name="segment_key", value_name="revenue")
    melted["segment"] = melted["segment_key"].map(segment_map)
    return melted


def get_expense_breakdown_df(df):
    """Reshapes expense categories into long format for visualization."""
    expense_map = {
        "exp_personnel": "Personnel & Talent",
        "exp_technology": "Technology & Feeds",
        "exp_operations": "Operations & Admin",
        "exp_compliance_risk": "Compliance & Risk",
        "exp_marketing": "Marketing"
    }
    id_vars = ["date", "month_name"]
    val_vars = list(expense_map.keys())
    melted = df.melt(id_vars=id_vars, value_vars=val_vars, var_name="expense_key", value_name="expense")
    melted["category"] = melted["expense_key"].map(expense_map)
    return melted


def generate_financial_narrative(summary):
    """Generates concise financial insights for business executives."""
    top_segment = max(summary["segment_shares_pct"].items(), key=lambda x: x[1])
    top_expense = max(summary["expense_shares_pct"].items(), key=lambda x: x[1])

    insights = [
        f"**Core Revenue Driver:** {top_segment[0]} is the largest business unit, contributing {top_segment[1]}% (${summary['segment_totals'][top_segment[0]]:.1f}M) of 24-month cumulative revenue.",
        f"**Cost Structure Concentration:** {top_expense[0]} represents {top_expense[1]}% (${summary['expense_totals'][top_expense[0]]:.1f}M) of operating expenditures, typical for knowledge-intensive institutional financial services.",
        f"**Operating Health:** The firm maintains an overall net profit margin of {summary['overall_profit_margin_pct']}%, showing healthy operating leverage as revenue scaled.",
        f"**Top-Line Momentum:** Average monthly revenue growth stood at {summary['avg_mom_growth_pct']}% with positive seasonal acceleration during quarter-end advisory closings."
    ]
    return insights
