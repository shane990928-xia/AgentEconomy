"""页面2: 宏观经济面板"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from demo_pages.data_utils import load_monthly_data


def render():
    st.title("Macroeconomic Dynamics")
    data = load_monthly_data()
    months = [d["econ_month"] for d in data]

    # ── GDP ──
    st.subheader("GDP")
    ngdp = [d.get("macro", {}).get("nominal_gdp", 0) for d in data]
    rgdp = [d.get("macro", {}).get("real_gdp", 0) for d in data]
    ngr = [d.get("macro", {}).get("gdp_growth_rate", 0) for d in data]
    rgr = [d.get("macro", {}).get("real_gdp_growth_rate", 0) for d in data]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("GDP Levels", "Growth Rates (%)"))
    fig.add_trace(go.Scatter(x=months, y=ngdp, name="Nominal GDP", line=dict(color="#636EFA")), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=rgdp, name="Real GDP", line=dict(color="#EF553B")), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=[g * 100 if abs(g) < 1 else g for g in ngr], name="Nominal Growth", line=dict(color="#636EFA")), row=1, col=2)
    fig.add_trace(go.Scatter(x=months, y=[g * 100 if abs(g) < 1 else g for g in rgr], name="Real Growth", line=dict(color="#EF553B")), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_layout(height=400, margin=dict(t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # ── 通胀 ──
    st.subheader("Inflation & CPI")
    cpi = []
    infl = []
    for d in data:
        pi = d.get("macro", {}).get("price_index", {})
        cpi.append(pi.get("index", 1.0) if isinstance(pi, dict) else 1.0)
        infl.append(d.get("macro", {}).get("inflation_rate", 0))

    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("CPI Index", "Monthly Inflation Rate (%)"))
    fig2.add_trace(go.Scatter(x=months, y=cpi, name="CPI", line=dict(color="#AB63FA"), fill="tozeroy"), row=1, col=1)
    fig2.add_trace(go.Bar(x=months, y=[i * 100 for i in infl], name="Inflation", marker_color=["#EF553B" if i > 0 else "#00CC96" for i in infl]), row=1, col=2)
    fig2.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="2% target", row=1, col=2)
    fig2.update_layout(height=350, margin=dict(t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    # ── 基尼系数 ──
    st.subheader("Inequality (Gini Coefficient)")
    gini_w = [d.get("household", {}).get("aggregate", {}).get("assets_distribution", {}).get("gini", 0) for d in data]
    gini_i = [d.get("household", {}).get("aggregate", {}).get("assets_distribution", {}).get("income_gini", 0) for d in data]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=months, y=gini_w, name="Wealth Gini", line=dict(color="#AB63FA", width=3)))
    fig3.add_trace(go.Scatter(x=months, y=gini_i, name="Income Gini", line=dict(color="#636EFA", width=3)))
    fig3.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Warning threshold (0.4)")
    fig3.update_layout(height=350, yaxis_title="Gini Coefficient", xaxis_title="Month", margin=dict(t=20, b=40))
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Wealth Gini", f"{gini_w[-1]:.3f}", f"{gini_w[-1] - gini_w[0]:+.3f}")
    c2.metric("Income Gini", f"{gini_i[-1]:.3f}", f"{gini_i[-1] - gini_i[0]:+.3f}")
    c3.metric("Gap (W - I)", f"{gini_w[-1] - gini_i[-1]:.3f}")

    # ── 政府财政 ──
    st.subheader("Government Finance")
    inc_tax = [d.get("government", {}).get("personal_income_tax", 0) for d in data]
    vat = [d.get("government", {}).get("consume_tax", 0) for d in data]
    corp = [d.get("government", {}).get("corporate_tax", 0) for d in data]
    redist = [d.get("government", {}).get("redistribution_total", 0) for d in data]
    total_tax = [a + b + c for a, b, c in zip(inc_tax, vat, corp)]

    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("Tax Composition", "Revenue vs Redistribution"))
    fig4.add_trace(go.Bar(x=months, y=inc_tax, name="Income Tax", marker_color="#636EFA"), row=1, col=1)
    fig4.add_trace(go.Bar(x=months, y=vat, name="VAT", marker_color="#00CC96"), row=1, col=1)
    fig4.add_trace(go.Bar(x=months, y=corp, name="Corporate Tax", marker_color="#EF553B"), row=1, col=1)
    fig4.update_layout(barmode="stack", height=400, margin=dict(t=40, b=40))
    fig4.add_trace(go.Scatter(x=months, y=total_tax, name="Total Tax", line=dict(color="#636EFA", width=2)), row=1, col=2)
    fig4.add_trace(go.Scatter(x=months, y=redist, name="Redistribution", line=dict(color="#EF553B", width=2, dash="dash")), row=1, col=2)
    st.plotly_chart(fig4, use_container_width=True)
