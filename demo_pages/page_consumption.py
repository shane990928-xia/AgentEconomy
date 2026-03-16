"""页面4: 单户消费 Live Demo"""
import streamlit as st
import asyncio
import json
import time
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_one_household(idx: int):
    """轻量加载单个家庭"""
    from test_consumption import _load_households_light
    hhs = _load_households_light(limit=idx + 1)
    return hhs[idx] if idx < len(hhs) else hhs[-1]


def render():
    st.title("🛒 Live Consumption Demo")
    st.markdown("Watch an LLM-driven household make real-time consumption decisions.")
    st.markdown("---")

    # ── 参数面板 ──
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Household Parameters")
        hh_idx = st.number_input("Household Index", min_value=0, max_value=299, value=0, step=1)
        balance_override = st.number_input("Balance ($)", value=0, step=1000,
                                           help="0 = use PSID default")
        income_override = st.number_input("Monthly Income ($)", value=0, step=500,
                                          help="0 = use PSID default")

        st.markdown("**Macro Indicators**")
        inflation = st.slider("Inflation Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
        unemployment = st.slider("Unemployment Rate (%)", 0.0, 20.0, 5.0, 0.5) / 100
        interest = st.slider("Interest Rate (%)", 0.0, 5.0, 0.4, 0.1) / 100

        run_btn = st.button("🚀 Run LLM Consumption", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Loading household data..."):
                hh = _load_one_household(hh_idx)

            balance = float(balance_override) if balance_override > 0 else float(hh.csv_values.get("ER85692") or 50000)
            income = float(income_override) if income_override > 0 else float(hh.csv_values.get("ER85629") or 3000)

            # 显示家庭信息
            st.subheader(f"Household: {hh.household_id}")
            info_cols = st.columns(3)
            info_cols[0].metric("Balance", f"${balance:,.0f}")
            info_cols[1].metric("Monthly Income", f"${income:,.0f}")
            persona_name = "N/A"
            if hh.persona:
                persona_name = hh.persona.get("persona_name", "N/A")
            info_cols[2].metric("Persona", persona_name)

            if hh.persona:
                with st.expander("📋 Persona Details", expanded=False):
                    chars = hh.persona.get("core_characteristics", "")
                    patterns = hh.persona.get("behavior_patterns", "")
                    st.markdown(f"**Core Characteristics:** {chars}")
                    st.markdown(f"**Behavior Patterns:** {patterns}")

            macro_indicators = {
                "inflation_rate": inflation,
                "unemployment_rate": unemployment,
                "interest_rate": interest,
                "price_index": 100.0,
            }

            # ── Step 0: Budget Allocation ──
            st.markdown("---")
            st.subheader("Step 0: Budget Allocation")
            step0_placeholder = st.empty()
            step0_placeholder.info("⏳ Calling LLM for budget allocation...")

            t0 = time.perf_counter()
            try:
                step0 = asyncio.run(hh.consumption_step0_major_budget_allocation(
                    available_balance=balance,
                    expected_income=income,
                    available_budget=balance,
                    macro_indicators=macro_indicators,
                ))
                t_step0 = time.perf_counter() - t0
                step0_placeholder.success(f"✅ Step 0 complete ({t_step0:.1f}s)")

                # 预算分配可视化
                import plotly.graph_objects as go
                budgets = step0.budgets or {}
                cats = list(budgets.keys())
                vals = list(budgets.values())

                bc1, bc2 = st.columns(2)
                with bc1:
                    fig = go.Figure(data=[go.Pie(labels=cats, values=vals, hole=0.4,
                                                  textinfo="label+percent",
                                                  marker=dict(colors=["#636EFA", "#EF553B", "#00CC96",
                                                                      "#AB63FA", "#FFA15A", "#19D3F3"]))])
                    fig.update_layout(title=f"Total Budget: ${step0.total_budget:,.0f}", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with bc2:
                    fig2 = go.Figure(data=[go.Bar(x=cats, y=vals,
                                                   marker_color=["#636EFA", "#EF553B", "#00CC96",
                                                                  "#AB63FA", "#FFA15A", "#19D3F3"],
                                                   text=[f"${v:,.0f}" for v in vals],
                                                   textposition="outside")])
                    fig2.update_layout(title="Budget by Category", height=350,
                                       yaxis_title="$", xaxis_tickangle=-30)
                    st.plotly_chart(fig2, use_container_width=True)

                if step0.note:
                    st.info(f"💭 **LLM Reasoning:** {step0.note}")

                # ── Step 1: Need Generation ──
                st.markdown("---")
                st.subheader("Step 1: Need Generation (Retail Categories)")
                step1_placeholder = st.empty()
                step1_placeholder.info("⏳ Generating specific needs per category...")

                retail_budget = float(budgets.get("Retail merchandise", 0))
                t1 = time.perf_counter()
                step1 = asyncio.run(hh.consumption_step1_needs_by_category(
                    total_budget=retail_budget,
                    available_balance=balance,
                    expected_income=income,
                    available_budget=balance,
                ))
                t_step1 = time.perf_counter() - t1
                step1_placeholder.success(f"✅ Step 1 complete ({t_step1:.1f}s) — {len(step1.category_plans)} categories")

                # 需求展示
                for cp in sorted(step1.category_plans, key=lambda x: -x.budget_amount):
                    if cp.budget_amount > 0:
                        with st.expander(f"**{cp.category}** — ${cp.budget_amount:,.0f}", expanded=cp.budget_amount > retail_budget * 0.1):
                            for need in cp.need_descriptions:
                                st.markdown(f"- {need}")

                # 总耗时
                st.markdown("---")
                total_time = t_step0 + t_step1
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Step 0 Time", f"{t_step0:.1f}s")
                tc2.metric("Step 1 Time", f"{t_step1:.1f}s")
                tc3.metric("Total", f"{total_time:.1f}s")

            except Exception as e:
                step0_placeholder.error(f"❌ LLM call failed: {e}")
                st.markdown("Check your `.env` API key and balance.")

        else:
            st.markdown("### 👈 Configure parameters and click **Run**")
            st.markdown("""
            This demo executes the first two steps of the consumption pipeline in real-time:

            1. **Step 0 — Budget Allocation**: The LLM reads the household's persona, financial state,
               and macro indicators, then allocates the monthly budget across 6 categories.

            2. **Step 1 — Need Generation**: For the retail category, the LLM generates specific
               need descriptions for each of 20 NAICS sub-categories (food, apparel, electronics, etc.).

            Steps 2–3 (vector retrieval + purchase decision) require the full Qdrant product database
            and are omitted in this demo.
            """)
