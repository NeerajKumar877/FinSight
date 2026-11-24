"""
FinSight – Financial Performance & Market Risk Analysis Platform
Streamlit Interactive Application

Designed for placement interviews (Macquarie, KPMG, etc.)
Focusing on explainability, financial markets, risk management, and business decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.financial_analytics import (
    load_financial_data,
    calculate_financial_summary,
    get_segment_breakdown_df,
    get_expense_breakdown_df,
    generate_financial_narrative
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
    calculate_correlation_matrix,
    calculate_asset_summary_table,
    calculate_portfolio_overview,
    ASSET_NAMES,
    ASSET_SHORT_LABELS
)
from src.risk_analytics import (
    calculate_historical_var,
    calculate_expected_shortfall,
    calculate_stress_test,
    get_predefined_stress_scenarios
)
from src.recommendations import generate_business_recommendations
from src.utils import (
    format_currency,
    format_percent,
    plot_financial_trends,
    plot_segment_revenue_stacked,
    plot_asset_allocation_donut,
    plot_cumulative_performance,
    plot_correlation_heatmap,
    plot_var_distribution,
    plot_stress_test_waterfall,
    COLORS
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSight | Financial Performance & Market Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .disclaimer-banner {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 10px 16px;
        border-radius: 8px;
        border-left: 5px solid #F59E0B;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    
    .metric-title {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #0F172A;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 2px;
    }
    
    .metric-subtitle {
        color: #64748B;
        font-size: 12px;
    }
    
    .recommendation-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-opportunity {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-positive {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_financial_data():
    return load_financial_data("data/synthetic_financial_performance.csv")

@st.cache_data
def get_cached_portfolio_data():
    return load_price_data("data/synthetic_portfolio_prices.csv")

fin_df = get_cached_financial_data()
price_df = get_cached_portfolio_data()
returns_df = calculate_daily_returns(price_df)
fin_summary = calculate_financial_summary(fin_df)
asset_summary_table = calculate_asset_summary_table(returns_df)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("FinSight")
    st.caption("Financial Performance & Market Risk Platform")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Financial Performance",
            "Portfolio Analysis",
            "Market Risk (VaR & ES)",
            "Scenario Stress Testing",
            "Business Recommendations"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("Portfolio Parameters")

    portfolio_value = st.number_input(
        "Portfolio Capital ($)",
        min_value=1_000_000,
        max_value=100_000_000,
        value=10_000_000,
        step=500_000,
        format="%d"
    )

    risk_free_rate = st.slider(
        "Risk-Free Rate (Rf)",
        min_value=0.0,
        max_value=6.0,
        value=3.0,
        step=0.25,
        format="%.2f%%"
    ) / 100.0

    st.markdown("---")
    st.subheader("Asset Allocation Presets")
    preset_choice = st.selectbox(
        "Select Strategy Preset",
        [
            "Custom Allocation",
            "Balanced Institutional (40/30/15/15)",
            "Aggressive Growth (60/15/15/10)",
            "Conservative Capital Preservation (20/50/20/10)"
        ],
        index=1
    )

    if preset_choice == "Balanced Institutional (40/30/15/15)":
        init_eq, init_bd, init_gd, init_fx = 40, 30, 15, 15
    elif preset_choice == "Aggressive Growth (60/15/15/10)":
        init_eq, init_bd, init_gd, init_fx = 60, 15, 15, 10
    elif preset_choice == "Conservative Capital Preservation (20/50/20/10)":
        init_eq, init_bd, init_gd, init_fx = 20, 50, 20, 10
    else:
        init_eq, init_bd, init_gd, init_fx = 40, 30, 15, 15

    w_equities = st.slider("Equities (%)", 0, 100, init_eq, 5)
    w_bonds = st.slider("Govt Bonds (%)", 0, 100, init_bd, 5)
    w_gold = st.slider("Gold (%)", 0, 100, init_gd, 5)
    w_fx = st.slider("Foreign Exchange (%)", 0, 100, init_fx, 5)

    raw_weights = {
        "equities": w_equities,
        "bonds": w_bonds,
        "gold": w_gold,
        "fx": w_fx
    }
    norm_weights = normalize_weights(raw_weights)

    sum_raw = w_equities + w_bonds + w_gold + w_fx
    if sum_raw != 100:
        st.warning(f"Raw weights sum to {sum_raw}%. Automatically normalized to 100% for mathematical consistency.")

    st.markdown("---")
    st.caption("FinSight v1.0 | Academic & Placement Evaluation")


# -----------------------------------------------------------------------------
# Portfolio & Risk Calculations for Active Inputs
# -----------------------------------------------------------------------------
port_overview = calculate_portfolio_overview(returns_df, norm_weights, risk_free_rate)
port_cum_series = calculate_cumulative_returns(port_overview["daily_returns"])

risk_summary = calculate_historical_var(
    port_overview["daily_returns"],
    confidence_level=0.95,
    portfolio_value=portfolio_value
)

es_summary = calculate_expected_shortfall(
    port_overview["daily_returns"],
    confidence_level=0.95,
    portfolio_value=portfolio_value
)

default_stress_summary = calculate_stress_test(
    norm_weights,
    shocks=None,
    portfolio_value=portfolio_value
)


# -----------------------------------------------------------------------------
# Universal Top Disclaimer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    <strong>DISCLAIMER – SYNTHETIC DATASET:</strong> All market prices, corporate financials, and transaction histories in FinSight 
    are generated synthetically for academic and interview demonstration purposes. This project does not utilize live commercial data 
    or regulatory Basel capital frameworks.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
if page == "Executive Overview":
    st.title("Executive Overview")
    st.markdown(
        "High-level strategic briefing consolidating institutional financial performance, "
        "multi-asset portfolio returns, and key 1-day market risk metrics."
    )

    # Top KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">24M Total Revenue</div>
            <div class="metric-value">${fin_summary['total_revenue_cumulative']:.1f}M</div>
            <div class="metric-subtitle">Across 4 business units</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Operating Profit Margin</div>
            <div class="metric-value">{fin_summary['overall_profit_margin_pct']}%</div>
            <div class="metric-subtitle">Net Profit: ${fin_summary['total_net_profit_cumulative']:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Portfolio Annual Return</div>
            <div class="metric-value">{port_overview['annualized_return_pct']}%</div>
            <div class="metric-subtitle">Sharpe Ratio: {port_overview['sharpe_ratio']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Annualized Volatility</div>
            <div class="metric-value">{port_overview['annualized_volatility_pct']}%</div>
            <div class="metric-subtitle">sqrt(252) daily price dispersion</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">1-Day 95% Historical VaR</div>
            <div class="metric-value">{risk_summary['var_pct']}%</div>
            <div class="metric-subtitle">${risk_summary['var_dollar']:,.0f} on ${portfolio_value:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Charts (Financial Trajectory & Portfolio Mix)
    col_left, col_right = st.columns([6, 4])
    with col_left:
        st.plotly_chart(plot_financial_trends(fin_df), use_container_width=True)

    with col_right:
        st.subheader("Current Asset Allocation")
        st.plotly_chart(plot_asset_allocation_donut(norm_weights), use_container_width=True)

        st.markdown(f"""
        **Active Weights:**
        - **Equities:** {norm_weights['equities']*100:.1f}% (${portfolio_value*norm_weights['equities']:,.0f})
        - **Govt Bonds:** {norm_weights['bonds']*100:.1f}% (${portfolio_value*norm_weights['bonds']:,.0f})
        - **Gold:** {norm_weights['gold']*100:.1f}% (${portfolio_value*norm_weights['gold']:,.0f})
        - **FX Basket:** {norm_weights['fx']*100:.1f}% (${portfolio_value*norm_weights['fx']:,.0f})
        """)

    # Executive Narrative Summary
    st.markdown("### Executive Highlights")
    st.info(
        "**Strategic Synthesis:** The firm operates a profitable asset management and treasury enterprise with an "
        f"average net margin of {fin_summary['overall_profit_margin_pct']}%. On the investment side, the multi-asset portfolio "
        f"yields an annualized return of {port_overview['annualized_return_pct']}% with an annualized volatility of "
        f"{port_overview['annualized_volatility_pct']}%. Historical VaR indicates that daily losses exceed {risk_summary['var_pct']}% "
        "only on 5% of trading days (1 day in 20). Expected Shortfall underscores that tail events average a loss of "
        f"{es_summary['es_pct']}% (${es_summary['es_dollar']:,.0f})."
    )


# =============================================================================
# PAGE 2: FINANCIAL PERFORMANCE
# =============================================================================
elif page == "Financial Performance":
    st.title("Corporate Financial Performance (Module 1)")
    st.markdown(
        "Interactive analysis of corporate revenue streams, operational expenses, "
        "profit margins, and business unit trajectories over 24 historical months."
    )

    # Financial KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "Latest Month Revenue",
            f"${fin_summary['latest_monthly_revenue']:.2f}M",
            delta=f"{fin_summary['latest_mom_growth_pct']}% MoM"
        )
    with kpi2:
        st.metric(
            "Latest Month Net Profit",
            f"${fin_summary['latest_monthly_profit']:.2f}M"
        )
    with kpi3:
        st.metric(
            "Latest Profit Margin",
            f"{fin_summary['latest_profit_margin_pct']}%",
            delta=f"{(fin_summary['latest_profit_margin_pct'] - fin_summary['overall_profit_margin_pct']):.2f}% vs. Avg"
        )
    with kpi4:
        st.metric(
            "Avg MoM Revenue Growth",
            f"{fin_summary['avg_mom_growth_pct']}%",
            help="Average month-over-month growth across the 24-month horizon."
        )

    # Monthly Trends
    st.markdown("---")
    st.plotly_chart(plot_financial_trends(fin_df), use_container_width=True)

    # Segment Breakdown & Expense Category Breakdown
    col_seg, col_exp = st.columns(2)
    with col_seg:
        st.plotly_chart(plot_segment_revenue_stacked(fin_df), use_container_width=True)
        st.markdown("**Segment Revenue Shares (24M Cumulative):**")
        seg_df = pd.DataFrame([
            {"Segment": k, "Cumulative ($M)": round(v, 2), "Share (%)": fin_summary['segment_shares_pct'][k]}
            for k, v in fin_summary['segment_totals'].items()
        ])
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

    with col_exp:
        st.subheader("Operating Cost Structure")
        exp_df = pd.DataFrame([
            {"Expense Category": k, "Cumulative ($M)": round(v, 2), "Share (%)": fin_summary['expense_shares_pct'][k]}
            for k, v in fin_summary['expense_totals'].items()
        ])
        st.dataframe(exp_df, use_container_width=True, hide_index=True)

        st.markdown("""
        **Cost Structure Insights:**
        - **Personnel & Talent** constitutes the primary expenditure (~50%), reflecting portfolio manager and analyst compensation.
        - **Technology & Market Data Feeds** represent ~21% of costs, supporting quantitative analytics and execution infrastructure.
        - **Operations & Compliance** maintain regulatory adherence and operational resilience.
        """)

    # Financial Narrative Insights
    st.markdown("### Management Insights & Takeaways")
    narrative_points = generate_financial_narrative(fin_summary)
    for point in narrative_points:
        st.markdown(f"- {point}")


# =============================================================================
# PAGE 3: PORTFOLIO ANALYSIS
# =============================================================================
elif page == "Portfolio Analysis":
    st.title("Multi-Asset Portfolio Analysis (Module 2)")
    st.markdown(
        "Quantitative evaluation of asset returns, portfolio volatility, historical compounding, "
        "and correlation dynamics across Equities, Government Bonds, Gold, and Foreign Exchange."
    )

    # Top KPI Metrics for current allocation
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Portfolio Cumulative Return", f"{port_overview['cumulative_return_pct']}%")
    with c2:
        st.metric("Annualized Return (CAGR)", f"{port_overview['annualized_return_pct']}%")
    with c3:
        st.metric("Annualized Volatility (σ)", f"{port_overview['annualized_volatility_pct']}%")
    with c4:
        st.metric("Sharpe Ratio (Rf=3%)", f"{port_overview['sharpe_ratio']}")

    # Cumulative Return Chart
    st.markdown("---")
    st.plotly_chart(plot_cumulative_performance(returns_df, port_cum_series), use_container_width=True)

    # Asset Performance Summary Table & Correlation Heatmap
    col_tbl, col_heat = st.columns([6, 5])
    with col_tbl:
        st.subheader("Individual Asset Performance Summary (2-Year)")
        st.dataframe(asset_summary_table, use_container_width=True, hide_index=True)

        st.markdown("""
        **Financial Concepts & Observations:**
        - **Equities:** Offers high capital appreciation potential (~11% return) accompanied by higher price volatility (~17%).
        - **Government Bonds:** Yields lower return (~4.5%) with very low volatility (~5.5%), acting as a volatility dampener.
        - **Gold:** Traditional store of value with solid risk-adjusted return (~8% return, 13.5% vol) and inflation hedging properties.
        - **Foreign Exchange:** Currency reserve buffer exhibiting low directional drift (~2.5%) and moderate volatility (~8%).
        """)

    with col_heat:
        corr_matrix = calculate_correlation_matrix(returns_df)
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

        st.markdown("""
        **Correlation & Diversification Rationale:**
        - **Negative Equity-Bond Correlation:** When equities sell off, sovereign bonds typically experience flight-to-quality inflows.
        - **Gold Decorrelation:** Gold exhibits near-zero to slightly negative correlation with risk assets, reducing portfolio drawdown.
        """)


# =============================================================================
# PAGE 4: MARKET RISK (VaR & EXPECTED SHORTFALL)
# =============================================================================
elif page == "Market Risk (VaR & ES)":
    st.title("Market Risk Analytics (Module 3)")
    st.markdown(
        "Rigorous quantitative tail risk assessment implementing **1-Day 95% Historical Value at Risk (VaR)** "
        "and **1-Day 95% Expected Shortfall (Conditional VaR)** on historical daily portfolio returns."
    )

    # Key Risk Cards
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">1-Day 95% Historical VaR</div>
            <div class="metric-value">{risk_summary['var_pct']}%</div>
            <div class="metric-subtitle">${risk_summary['var_dollar']:,.0f} on ${portfolio_value:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">1-Day 95% Expected Shortfall</div>
            <div class="metric-value">{es_summary['es_pct']}%</div>
            <div class="metric-subtitle">${es_summary['es_dollar']:,.0f} average tail loss</div>
        </div>
        """, unsafe_allow_html=True)

    with rc3:
        tail_ratio = round(es_summary['es_pct'] / risk_summary['var_pct'], 2) if risk_summary['var_pct'] > 0 else 1.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tail Severity Ratio (ES / VaR)</div>
            <div class="metric-value">{tail_ratio}x</div>
            <div class="metric-subtitle">{es_summary['tail_observations_count']} days in the worst 5% tail</div>
        </div>
        """, unsafe_allow_html=True)

    # Return Distribution Histogram
    st.markdown("---")
    st.plotly_chart(
        plot_var_distribution(
            port_overview["daily_returns"],
            risk_summary["cutoff_return"],
            es_summary["tail_mean_return"]
        ),
        use_container_width=True
    )

    # Educational & Interview Interpretation Cards
    st.markdown("### Beginner-to-Intermediate Interpretation Guide")
    col_var, col_es = st.columns(2)
    with col_var:
        st.subheader("1. What does 95% Historical VaR mean?")
        st.info(risk_summary["explanation"])
        st.markdown("""
        **Methodology Summary:**
        1. Take all 503 historical daily returns of the current weighted portfolio.
        2. Sort them from worst to best.
        3. Identify the 5th percentile return (the cutoff point where 95% of returns are better).
        4. State the loss as a positive percentage and dollar figure.
        """)

    with col_es:
        st.subheader("2. What does Expected Shortfall (CVaR) mean?")
        st.warning(es_summary["explanation"])
        st.markdown("""
        **Why Expected Shortfall is Crucial:**
        - **Limitation of VaR:** VaR only tells you where the door to the disaster zone begins; it does **not** tell you how bad the fire is once inside.
        - **Advantage of ES:** Expected Shortfall calculates the *average loss* across all days worse than the VaR threshold.
        - **Mathematical Rule:** $ES \ge VaR$ always holds true.
        """)


# =============================================================================
# PAGE 5: SCENARIO STRESS TESTING
# =============================================================================
elif page == "Scenario Stress Testing":
    st.title("Scenario Stress Testing (Module 3)")
    st.markdown(
        "Deterministic stress testing evaluating the financial resilience of the portfolio "
        "under adverse macroeconomic shocks and severe market dislocations."
    )

    # Scenario Selection
    scenarios = get_predefined_stress_scenarios()
    selected_scenario_name = st.selectbox(
        "Select Stress Scenario",
        list(scenarios.keys()) + ["Custom Shocks (User Defined)"],
        index=0
    )

    if selected_scenario_name == "Custom Shocks (User Defined)":
        st.markdown("**Configure Custom Market Shocks:**")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            shock_eq = st.slider("Equities Shock (%)", -40, 20, -10, 1) / 100.0
        with col_s2:
            shock_bd = st.slider("Govt Bonds Shock (%)", -20, 20, -5, 1) / 100.0
        with col_s3:
            shock_gd = st.slider("Gold Shock (%)", -30, 30, -8, 1) / 100.0
        with col_s4:
            shock_fx = st.slider("FX Shock (%)", -20, 20, -5, 1) / 100.0

        active_shocks = {
            "equities": shock_eq,
            "bonds": shock_bd,
            "gold": shock_gd,
            "fx": shock_fx
        }
        st.caption("Custom user-defined shocks applied.")
    else:
        active_scenario = scenarios[selected_scenario_name]
        active_shocks = {
            "equities": active_scenario["equities"],
            "bonds": active_scenario["bonds"],
            "gold": active_scenario["gold"],
            "fx": active_scenario["fx"]
        }
        st.info(f"**Scenario Narrative:** {active_scenario['description']}")

    # Run stress test calculation
    active_stress = calculate_stress_test(norm_weights, shocks=active_shocks, portfolio_value=portfolio_value)

    # Stress Impact Metrics
    st.markdown("---")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Pre-Stress Portfolio Capital", f"${portfolio_value:,.0f}")
    with sc2:
        st.metric("Post-Stress Portfolio Value", f"${active_stress['portfolio_value_post_shock']:,.0f}")
    with sc3:
        st.metric("Estimated Portfolio Loss", f"${active_stress['portfolio_loss_dollar']:,.0f}")
    with sc4:
        st.metric("Drawdown Percentage", f"{active_stress['portfolio_loss_pct']}%")

    # Visualizations: Waterfall of dollar contributions
    st.plotly_chart(plot_stress_test_waterfall(active_stress), use_container_width=True)

    # Detailed Table Breakdown
    st.subheader("Asset-Wise Stress Impact Breakdown")
    breakdown_df = pd.DataFrame(active_stress["asset_breakdown"])[
        ["asset_name", "weight_pct", "allocation_dollar", "shock_pct", "dollar_loss", "contribution_to_total_loss_pct"]
    ]
    breakdown_df.columns = [
        "Asset Class",
        "Allocation (%)",
        "Pre-Shock Value ($)",
        "Assumed Shock (%)",
        "Estimated Loss ($)",
        "Share of Total Loss (%)"
    ]
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.markdown("""
    > [!NOTE]
    > **Methodological Assumptions:** This stress test employs a deterministic linear factor model. It applies instantaneous price 
    > shocks directly to asset holdings without incorporating secondary non-linear effects (such as bid-ask widening or forced liquidations). 
    > All assumptions remain transparent and fully editable.
    """)


# =============================================================================
# PAGE 6: BUSINESS RECOMMENDATIONS
# =============================================================================
elif page == "Business Recommendations":
    st.title("Strategic Business Recommendations (Module 4)")
    st.markdown(
        "Actionable, metric-driven recommendations for senior management and the investment committee. "
        "Every single insight is directly anchored to computed figures from prior modules."
    )

    # Generate dynamic recommendations
    combined_risk = {**risk_summary, **es_summary}
    recommendations = generate_business_recommendations(
        port_overview,
        combined_risk,
        default_stress_summary,
        fin_summary,
        asset_summary_table
    )

    # Render each recommendation card
    for rec in recommendations:
        severity_class = f"badge-{rec['severity'].lower()}"
        st.markdown(f"""
        <div class="recommendation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; font-size: 17px; color: #0F172A;">{rec['title']}</span>
                <span class="{severity_class}">{rec['severity'].upper()}</span>
            </div>
            <div style="color: #475569; font-size: 13px; font-weight: 600; margin-bottom: 6px;">
                Category: {rec['category']}
            </div>
            <div style="background-color: #F8FAFC; border-left: 3px solid #3B82F6; padding: 8px 12px; margin-bottom: 10px; font-size: 13px; font-family: monospace; color: #1E293B;">
                <strong>Metric Evidence:</strong> {rec['metric_evidence']}
            </div>
            <div style="color: #334155; font-size: 14px; margin-bottom: 8px;">
                <strong>Business Rationale:</strong> {rec['business_rationale']}
            </div>
            <div style="color: #0F172A; font-size: 14px;">
                <strong>Actionable Next Step:</strong> {rec['actionable_recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Summary Table of Managerial Action Items")
    summary_action_df = pd.DataFrame([
        {
            "Category": r["category"],
            "Finding / Risk": r["title"],
            "Severity": r["severity"],
            "Recommended Managerial Action": r["actionable_recommendation"]
        }
        for r in recommendations
    ])
    st.dataframe(summary_action_df, use_container_width=True, hide_index=True)
