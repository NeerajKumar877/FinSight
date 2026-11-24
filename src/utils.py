"""
FinSight - Core Utilities, Formatting, and Plotly Visualization Builders
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Professional Color Palette
COLORS = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#2563EB",    # Royal Blue
    "accent_teal": "#0D9488",  # Teal / Growth
    "accent_gold": "#D97706",  # Gold / Amber
    "accent_red": "#DC2626",   # Crimson / Danger / Loss
    "accent_green": "#16A34A", # Emerald / Profit
    "slate_dark": "#1E293B",   # Dark Slate
    "slate_light": "#F8FAFC",  # Light Slate Background
    "border": "#E2E8F0",
    "equities": "#2563EB",     # Blue
    "bonds": "#0D9488",        # Teal
    "gold": "#D97706",         # Gold
    "fx": "#7C3AED"            # Purple
}


def format_currency(val, decimals=2):
    """Formats numeric values into clean currency strings ($M or standard)."""
    if val is None or np.isnan(val):
        return "$0.00"
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.{decimals}f}M"
    elif abs(val) >= 1_000:
        return f"${val:,.{decimals}f}"
    else:
        return f"${val:.{decimals}f}"


def format_percent(val, decimals=2):
    """Formats numeric values into percentage strings."""
    if val is None or np.isnan(val):
        return "0.00%"
    return f"{val:.{decimals}f}%"


def apply_chart_style(fig, title="", height=400):
    """Applies a clean, institutional styling to Plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter, sans-serif", color="#1E293B", weight="bold")),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#334155"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            linecolor="#CBD5E1"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            linecolor="#CBD5E1"
        )
    )
    return fig


def plot_financial_trends(fin_df):
    """Line and bar combo chart for monthly revenue, expenses, and net profit margin."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=fin_df["month_name"],
        y=fin_df["total_revenue"],
        name="Revenue ($M)",
        marker_color=COLORS["secondary"],
        opacity=0.85
    ))

    fig.add_trace(go.Bar(
        x=fin_df["month_name"],
        y=fin_df["total_expenses"],
        name="Expenses ($M)",
        marker_color="#94A3B8",
        opacity=0.85
    ))

    fig.add_trace(go.Scatter(
        x=fin_df["month_name"],
        y=fin_df["profit_margin_pct"],
        name="Profit Margin (%)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=COLORS["accent_teal"], width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Amount ($M)", showgrid=True, gridcolor="#F1F5F9"),
        yaxis2=dict(
            title="Margin (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, max(fin_df["profit_margin_pct"]) * 1.3]
        )
    )
    return apply_chart_style(fig, "Monthly Corporate Financial Performance (Revenue, Expenses & Margin)")


def plot_segment_revenue_stacked(fin_df):
    """Stacked bar chart showing revenue contribution by segment over time."""
    fig = go.Figure()
    segments = [
        ("rev_wealth_management", "Wealth Management", "#2563EB"),
        ("rev_asset_management", "Asset Management", "#0D9488"),
        ("rev_investment_advisory", "Investment Advisory", "#D97706"),
        ("rev_treasury_services", "Treasury Services", "#7C3AED")
    ]
    for col, name, color in segments:
        fig.add_trace(go.Bar(
            x=fin_df["month_name"],
            y=fin_df[col],
            name=name,
            marker_color=color
        ))

    fig.update_layout(barmode="stack", yaxis=dict(title="Revenue ($M)"))
    return apply_chart_style(fig, "Segment Revenue Contribution ($M)")


def plot_asset_allocation_donut(weights):
    """Donut chart of current portfolio asset allocations."""
    labels = ["Equities", "Govt Bonds", "Gold", "Foreign Exchange"]
    values = [weights["equities"], weights["bonds"], weights["gold"], weights["fx"]]
    colors = [COLORS["equities"], COLORS["bonds"], COLORS["gold"], COLORS["fx"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hoverinfo="label+percent+value"
    )])

    fig.update_layout(
        showlegend=True,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig


def plot_cumulative_performance(returns_df, port_cum_series):
    """Line chart comparing cumulative return of the portfolio vs. each asset class."""
    fig = go.Figure()

    # Portfolio trace (bold Navy)
    fig.add_trace(go.Scatter(
        x=returns_df["date"],
        y=port_cum_series * 100,
        mode="lines",
        name="Combined Portfolio",
        line=dict(color=COLORS["primary"], width=3.5)
    ))

    # Asset traces
    asset_meta = [
        ("equities", "Equities", COLORS["equities"]),
        ("bonds", "Govt Bonds", COLORS["bonds"]),
        ("gold", "Gold", COLORS["gold"]),
        ("fx", "Foreign Exchange", COLORS["fx"])
    ]

    for col, label, color in asset_meta:
        cum_ret = ((1.0 + returns_df[col]).cumprod() - 1.0) * 100
        fig.add_trace(go.Scatter(
            x=returns_df["date"],
            y=cum_ret,
            mode="lines",
            name=label,
            line=dict(color=color, width=1.5, dash="dot")
        ))

    fig.update_layout(
        yaxis=dict(title="Cumulative Return (%)", showgrid=True, gridcolor="#F1F5F9"),
        xaxis=dict(title="Date")
    )
    return apply_chart_style(fig, "Cumulative Historical Return: Portfolio vs. Individual Assets")


def plot_correlation_heatmap(corr_df):
    """Heatmap of the 4x4 asset correlation matrix."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        origin="upper",
        zmin=-0.5,
        zmax=1.0
    )
    fig.update_layout(
        title=dict(text="Asset Return Correlation Matrix (2-Year Historical)", font=dict(size=15, weight="bold")),
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
        font=dict(family="Inter, sans-serif")
    )
    return fig


def plot_var_distribution(portfolio_returns, var_cutoff, es_cutoff):
    """
    Histogram of daily portfolio returns with clear vertical threshold lines
    for 95% Historical VaR and 95% Expected Shortfall.
    """
    ret_pct = portfolio_returns * 100
    var_cutoff_pct = var_cutoff * 100
    es_cutoff_pct = es_cutoff * 100

    fig = go.Figure()

    # Base distribution
    fig.add_trace(go.Histogram(
        x=ret_pct,
        nbinsx=50,
        name="Daily Return Distribution",
        marker_color="#93C5FD",
        opacity=0.75
    ))

    # VaR vertical line
    fig.add_vline(
        x=var_cutoff_pct,
        line_width=3,
        line_dash="dash",
        line_color=COLORS["accent_gold"],
        annotation_text=f"95% VaR Cutoff: {var_cutoff_pct:.2f}%",
        annotation_position="top left",
        annotation_font=dict(color=COLORS["accent_gold"], size=12, family="Inter")
    )

    # Expected Shortfall vertical line
    fig.add_vline(
        x=es_cutoff_pct,
        line_width=3,
        line_dash="dot",
        line_color=COLORS["accent_red"],
        annotation_text=f"95% Expected Shortfall: {es_cutoff_pct:.2f}%",
        annotation_position="bottom left",
        annotation_font=dict(color=COLORS["accent_red"], size=12, family="Inter")
    )

    fig.update_layout(
        xaxis=dict(title="Daily Portfolio Return (%)", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="Observation Count", showgrid=True, gridcolor="#F1F5F9")
    )
    return apply_chart_style(fig, "Daily Portfolio Return Distribution with VaR & ES Tail Thresholds")


def plot_stress_test_waterfall(stress_summary):
    """Horizontal bar chart illustrating dollar loss contribution by asset class under stress."""
    breakdown = stress_summary["asset_breakdown"]
    df = pd.DataFrame(breakdown)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["asset_name"],
        x=df["dollar_loss"],
        orientation="h",
        marker_color=COLORS["accent_red"],
        text=[f"${x:,.0f} ({c}%)" for x, c in zip(df["dollar_loss"], df["contribution_to_total_loss_pct"])],
        textposition="auto"
    ))

    fig.update_layout(
        xaxis=dict(title="Estimated Loss ($ USD)", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(autorange="reversed")
    )
    return apply_chart_style(fig, "Estimated Stress Loss Contribution by Asset Class ($ USD)")
