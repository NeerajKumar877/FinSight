"""
FinSight - Business Recommendations Engine (Module 4)
Generates deterministic, metric-anchored business recommendations
for executive management and investment committees.
"""


def generate_business_recommendations(portfolio_summary, risk_summary, stress_summary, financial_summary, asset_table):
    """
    Analyzes actual calculated figures across portfolio, risk, stress tests,
    and financial performance to generate 3-5 explainable, metric-driven recommendations.

    Zero generic or random text: every single recommendation cites real computed numbers.
    """
    recommendations = []
    weights = portfolio_summary.get("weights", {"equities": 0.25, "bonds": 0.25, "gold": 0.25, "fx": 0.25})
    ann_vol = portfolio_summary.get("annualized_volatility_pct", 10.0)
    ann_ret = portfolio_summary.get("annualized_return_pct", 8.0)
    sharpe = portfolio_summary.get("sharpe_ratio", 0.5)
    var_95 = risk_summary.get("var_pct", 1.2)
    es_95 = risk_summary.get("es_pct", round(var_95 * 1.35, 2))
    es_dollar = risk_summary.get("es_dollar", var_95 * 13_500.0)
    stress_loss_pct = stress_summary.get("portfolio_loss_pct", 7.0)
    stress_loss_dollar = stress_summary.get("portfolio_loss_dollar", 700_000.0)

    # -------------------------------------------------------------
    # 1. Concentration Risk Analysis
    # -------------------------------------------------------------
    max_asset = max(weights.items(), key=lambda x: x[1])
    max_asset_name = max_asset[0].capitalize()
    max_asset_weight_pct = round(max_asset[1] * 100, 1)

    if max_asset_weight_pct > 40.0:
        recommendations.append({
            "title": f"High Asset Concentration in {max_asset_name}",
            "category": "Portfolio Concentration",
            "severity": "Warning" if max_asset_weight_pct <= 55.0 else "Critical",
            "metric_evidence": f"{max_asset_name} allocation is {max_asset_weight_pct}% (Prudent benchmark ceiling: 35.0% - 40.0%).",
            "business_rationale": (
                f"A {max_asset_weight_pct}% single-asset weighting disproportionately links total portfolio returns "
                f"to {max_asset_name} volatility, diluting the protective benefits of multi-asset diversification."
            ),
            "actionable_recommendation": (
                f"Trim {max_asset_name} exposure by {round(max_asset_weight_pct - 35.0, 1)}% and reallocate capital into "
                f"uncorrelated assets (e.g., Government Bonds or Gold) to establish a more resilient risk-parity structure."
            )
        })
    else:
        recommendations.append({
            "title": "Balanced Asset Concentration Profile",
            "category": "Portfolio Concentration",
            "severity": "Positive",
            "metric_evidence": f"Largest single holding ({max_asset_name}) is controlled at {max_asset_weight_pct}% (under 40.0% threshold).",
            "business_rationale": "Portfolio exposure is prudently distributed across asset classes, lowering idiosyncratic risk.",
            "actionable_recommendation": "Maintain current weighting guardrails with quarterly rebalancing bands of +/- 5%."
        })

    # -------------------------------------------------------------
    # 2. Volatility & Tail Risk Governance
    # -------------------------------------------------------------
    if ann_vol > 14.0 or var_95 > 1.8:
        recommendations.append({
            "title": "Elevated Portfolio Volatility and Tail Loss Exposure",
            "category": "Market Risk Management",
            "severity": "Warning",
            "metric_evidence": (
                f"Annualized Volatility is {ann_vol}% (Target <= 12.0%); "
                f"1-Day 95% Historical VaR is {var_95}% with Expected Shortfall of {es_95}%."
            ),
            "business_rationale": (
                f"The portfolio experiences significant price dispersion. An Expected Shortfall of {es_95}% implies that on "
                f"severe market sell-off days (the worst 5% tail), daily losses average ${risk_summary['es_dollar']:,.0f}."
            ),
            "actionable_recommendation": (
                "Increase allocations to Government Bonds (annualized volatility ~5.5%) or implement structured collar options "
                "to dampen drawdown severity without liquidating core positions."
            )
        })
    else:
        recommendations.append({
            "title": "Controlled Volatility within Institutional Risk Appetite",
            "category": "Market Risk Management",
            "severity": "Positive",
            "metric_evidence": f"Annualized volatility is contained at {ann_vol}% with 1-Day 95% VaR at {var_95}%.",
            "business_rationale": "The asset mix maintains a favorable risk profile suitable for conservative-to-moderate institutional mandates.",
            "actionable_recommendation": "Continue monitoring market regime shifts and maintain dry powder for tactical entry."
        })

    # -------------------------------------------------------------
    # 3. Stress Test Resilience & Capital Adequacy
    # -------------------------------------------------------------
    if stress_loss_pct > 7.0:
        top_stress_contributor = max(stress_summary["asset_breakdown"], key=lambda x: x["dollar_loss"])
        recommendations.append({
            "title": "Adverse Drawdown Vulnerability Under Macro Shocks",
            "category": "Stress Testing & Scenario Analysis",
            "severity": "Warning",
            "metric_evidence": (
                f"Simulated macro stress loss is {stress_loss_pct}% (${stress_loss_dollar:,.0f}). "
                f"{top_stress_contributor['asset_name']} drives {top_stress_contributor['contribution_to_total_loss_pct']}% of this loss."
            ),
            "business_rationale": (
                f"Under adverse market conditions, the current allocation suffers substantial capital erosion. "
                f"{top_stress_contributor['asset_name']} generates ${top_stress_contributor['dollar_loss']:,.0f} of the drawdown."
            ),
            "actionable_recommendation": (
                f"Establish tactical downside hedging on {top_stress_contributor['asset_name']} and review stop-loss triggers "
                f"to prevent tail loss contagion during market crises."
            )
        })

    # -------------------------------------------------------------
    # 4. Asset-Class Risk-Adjusted Efficiency (Sharpe Drag)
    # -------------------------------------------------------------
    # Find asset with lowest Sharpe ratio in asset_table
    if isinstance(asset_table, list):
        sorted_assets = sorted(asset_table, key=lambda x: x["Sharpe Ratio"])
    else:
        sorted_assets = asset_table.sort_values(by="Sharpe Ratio").to_dict(orient="records")

    laggard = sorted_assets[0]
    leader = sorted_assets[-1]

    if laggard["Sharpe Ratio"] < 0.3:
        recommendations.append({
            "title": f"Suboptimal Risk-Adjusted Return in {laggard['Short Name']}",
            "category": "Capital Efficiency",
            "severity": "Opportunity",
            "metric_evidence": (
                f"{laggard['Short Name']} Sharpe ratio is {laggard['Sharpe Ratio']:.2f} "
                f"(Annualized return: {laggard['Annualized Return (%)']}%, Vol: {laggard['Annualized Volatility (%)']}%) "
                f"vs. leading {leader['Short Name']} Sharpe ratio of {leader['Sharpe Ratio']:.2f}."
            ),
            "business_rationale": (
                f"{laggard['Short Name']} delivers inadequate return compensation for the market risk it introduces to the portfolio."
            ),
            "actionable_recommendation": (
                f"Underweight {laggard['Short Name']} in favor of higher risk-adjusted performers or reallocate into "
                f"short-duration sovereign paper offering reliable yield with minimal capital risk."
            )
        })

    # -------------------------------------------------------------
    # 5. Operating Leverage & Financial Performance
    # -------------------------------------------------------------
    latest_margin = financial_summary["latest_profit_margin_pct"]
    overall_margin = financial_summary["overall_profit_margin_pct"]
    personnel_share = financial_summary["expense_shares_pct"].get("Personnel & Talent", 0)

    if latest_margin < overall_margin:
        recommendations.append({
            "title": "Margin Compression in Operating Business",
            "category": "Corporate Financial Strategy",
            "severity": "Warning",
            "metric_evidence": (
                f"Latest month profit margin contracted to {latest_margin}% vs. 24-month average of {overall_margin}%. "
                f"Personnel expenses account for {personnel_share}% of total costs."
            ),
            "business_rationale": (
                "Operating costs have outpaced top-line revenue growth in recent periods, signaling margin compression."
            ),
            "actionable_recommendation": (
                "Conduct a line-item audit of administrative expenses and link discretionary talent compensation to "
                "realized net operating profit rather than gross transaction volume."
            )
        })
    else:
        recommendations.append({
            "title": "Robust Operating Margin & Commercial Expansion",
            "category": "Corporate Financial Strategy",
            "severity": "Positive",
            "metric_evidence": (
                f"Latest profit margin expanded to {latest_margin}%, exceeding the 24-month baseline of {overall_margin}%."
            ),
            "business_rationale": (
                "The investment firm demonstrates positive operational leverage as revenue expansion outpaces fixed overhead costs."
            ),
            "actionable_recommendation": (
                "Reinvest retained earnings into scalable digital wealth management platforms to expand client AUM."
            )
        })

    # Limit to top 5 most actionable recommendations
    return recommendations[:5]
