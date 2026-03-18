"""页面4: 单户消费 Live Demo + 对比实验"""
import streamlit as st
import asyncio
import json
import time
import sys
import os
import warnings
from pathlib import Path
from contextlib import contextmanager

# 屏蔽 litellm 异步日志的 RuntimeWarning（asyncio.run 销毁 pending tasks 时触发）
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 让 asyncio.run() 在 Streamlit 已有 event loop 中也能工作
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Agent method config 路径
_CONFIG_PATH = ROOT / "agenteconomy" / "llm" / "config.yaml"

# 可选的 agent method 列表
_AGENT_METHODS = ["none (direct)", "cot", "self_refine", "reflexion", "debate", "discussion"]


def _load_one_household(idx: int):
    """轻量加载单个家庭"""
    from test_consumption import _load_households_light
    hhs = _load_households_light(limit=idx + 1)
    return hhs[idx] if idx < len(hhs) else hhs[-1]


@contextmanager
def _temporary_agent_method(method_name: str):
    """临时切换 config.yaml 中的 agent_name，退出时恢复。"""
    original = _CONFIG_PATH.read_text(encoding="utf-8") if _CONFIG_PATH.exists() else ""
    try:
        if method_name.startswith("none"):
            new_value = "none"
        else:
            new_value = method_name

        # 替换 agent_name 行
        lines = original.splitlines(keepends=True)
        found = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("agent_name:"):
                lines[i] = f"agent_name: {new_value}\n"
                found = True
                break
        if not found:
            lines.insert(0, f"agent_name: {new_value}\n")
        _CONFIG_PATH.write_text("".join(lines), encoding="utf-8")
        yield
    finally:
        # 恢复原始内容
        _CONFIG_PATH.write_text(original, encoding="utf-8")


# ── 共用：参数面板 ──
def _param_panel(prefix: str, defaults: dict | None = None, show_agent_method: bool = False):
    """渲染一组参数控件，返回参数字典。prefix 用于 key 去重。"""
    d = defaults or {}
    hh_idx = st.number_input("Household Index", 0, 299, d.get("hh_idx", 0), 1, key=f"{prefix}_hh")
    balance = st.number_input("Balance ($)", value=d.get("balance", 0), step=1000,
                              help="0 = use PSID default", key=f"{prefix}_bal")
    income = st.number_input("Monthly Income ($)", value=d.get("income", 0), step=500,
                             help="0 = use PSID default", key=f"{prefix}_inc")
    st.markdown("**Macro Indicators**")
    inflation = st.slider("Inflation (%)", 0.0, 10.0, d.get("inflation", 2.0), 0.1, key=f"{prefix}_infl") / 100
    unemployment = st.slider("Unemployment (%)", 0.0, 20.0, d.get("unemployment", 5.0), 0.5, key=f"{prefix}_unemp") / 100
    interest = st.slider("Interest Rate (%)", 0.0, 5.0, d.get("interest", 0.4), 0.1, key=f"{prefix}_ir") / 100

    agent_method = d.get("agent_method", "none (direct)")
    if show_agent_method:
        st.markdown("**Agent Method**")
        default_idx = _AGENT_METHODS.index(agent_method) if agent_method in _AGENT_METHODS else 0
        agent_method = st.selectbox("Reasoning Strategy", _AGENT_METHODS, index=default_idx,
                                    key=f"{prefix}_am",
                                    help="How the LLM reasons: direct (1 call), CoT, self-refine, reflexion, debate, discussion")

    return dict(hh_idx=hh_idx, balance=balance, income=income,
                inflation=inflation, unemployment=unemployment, interest=interest,
                agent_method=agent_method)


# ── 共用：运行一次消费流程，返回结果 ──
def _run_consumption(params: dict):
    """执行 Step0 + Step1，返回结果字典或抛异常。"""
    hh = _load_one_household(params["hh_idx"])
    balance = float(params["balance"]) if params["balance"] > 0 else float(hh.csv_values.get("ER85692") or 50000)
    income = float(params["income"]) if params["income"] > 0 else float(hh.csv_values.get("ER85629") or 3000)
    macro = {
        "inflation_rate": params["inflation"],
        "unemployment_rate": params["unemployment"],
        "interest_rate": params["interest"],
        "price_index": 100.0,
    }

    agent_method = params.get("agent_method", "none (direct)")

    with _temporary_agent_method(agent_method):
        t0 = time.perf_counter()
        step0 = asyncio.run(hh.consumption_step0_major_budget_allocation(
            available_balance=balance, expected_income=income,
            available_budget=balance, macro_indicators=macro,
        ))
        dur0 = time.perf_counter() - t0

        retail_budget = float((step0.budgets or {}).get("Retail merchandise", 0))
        t1 = time.perf_counter()
        step1 = asyncio.run(hh.consumption_step1_needs_by_category(
            total_budget=retail_budget, available_balance=balance,
            expected_income=income, available_budget=balance,
        ))
        dur1 = time.perf_counter() - t1

    return dict(hh=hh, balance=balance, income=income,
                step0=step0, step1=step1, dur0=dur0, dur1=dur1,
                agent_method=agent_method)


# ── 共用：展示单次结果 ──
def _show_result(res: dict, container=st):
    """在给定容器中展示一次消费结果。"""
    import plotly.graph_objects as go

    hh, step0, step1 = res["hh"], res["step0"], res["step1"]
    balance, income = res["balance"], res["income"]

    # 家庭信息
    info_cols = container.columns(3)
    info_cols[0].metric("Balance", f"${balance:,.0f}")
    info_cols[1].metric("Income", f"${income:,.0f}")
    persona_name = hh.persona.get("persona_name", "N/A") if hh.persona else "N/A"
    info_cols[2].metric("Persona", persona_name)

    # Agent method 标签
    am = res.get("agent_method", "")
    if am and am != "none (direct)":
        container.success(f"🧠 Agent Method: **{am}**")

    # 预算饼图
    budgets = step0.budgets or {}
    cats = list(budgets.keys())
    vals = list(budgets.values())
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    fig = go.Figure(data=[go.Pie(labels=cats, values=vals, hole=0.4,
                                  textinfo="label+percent",
                                  marker=dict(colors=colors[:len(cats)]))])
    fig.update_layout(title=f"Total: ${step0.total_budget:,.0f}", height=300,
                      margin=dict(t=40, b=10, l=10, r=10))
    container.plotly_chart(fig, use_container_width=True)

    if step0.note:
        container.info(f"💭 {step0.note}")

    # LLM 原始输出
    raw = getattr(step0, "raw_llm_output", None)
    if raw:
        with container.expander("🔍 Raw LLM Output", expanded=False):
            st.code(raw, language="json")

    # Step1 需求
    container.markdown(f"**Step 1:** {len(step1.category_plans)} categories")
    retail_budget = float(budgets.get("Retail merchandise", 0))
    for cp in sorted(step1.category_plans, key=lambda x: -x.budget_amount):
        if cp.budget_amount > 0:
            with container.expander(f"{cp.category} — ${cp.budget_amount:,.0f}",
                                    expanded=cp.budget_amount > retail_budget * 0.15):
                for need in cp.need_descriptions:
                    st.markdown(f"- {need}")

    # 耗时
    tc1, tc2 = container.columns(2)
    tc1.metric("Step 0", f"{res['dur0']:.1f}s")
    tc2.metric("Step 1", f"{res['dur1']:.1f}s")


# ═══════════════════════════════════════════
# 主渲染
# ═══════════════════════════════════════════
def render():
    st.title("🛒 Live Consumption Demo")
    st.markdown("Watch LLM-driven households make real-time consumption decisions.")

    mode = st.radio("Mode", ["Single Run", "⚖️ Comparison"], horizontal=True)

    st.markdown("---")

    if mode == "Single Run":
        _render_single()
    else:
        _render_comparison()


# ── Single Run ──
def _render_single():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Parameters")
        params = _param_panel("single", show_agent_method=True)
        run_btn = st.button("🚀 Run", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Running LLM consumption pipeline..."):
                try:
                    res = _run_consumption(params)
                    st.subheader(f"Household: {res['hh'].household_id}")
                    _show_result(res)
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    st.caption("Check `.env` API key and balance.")
        else:
            st.markdown("### 👈 Configure parameters and click **Run**")
            st.markdown("""
            This demo executes the first two steps of the consumption pipeline in real-time:

            1. **Step 0 — Budget Allocation**: LLM allocates monthly budget across 6 categories.
            2. **Step 1 — Need Generation**: LLM generates specific need descriptions for retail sub-categories.

            Steps 2–3 (vector retrieval + purchase decision) require the full Qdrant product database
            and are omitted in this demo.
            """)


# ═══════════════════════════════════════════
# Comparison Mode
# ═══════════════════════════════════════════

# 预设对比模板
_PRESETS = {
    "Custom (free edit)": (
        dict(hh_idx=0, balance=0, income=0, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=0, income=0, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
    ),
    "💰 High vs Low Income": (
        dict(hh_idx=0, balance=0, income=8000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=0, income=2000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
    ),
    "🏦 Rich vs Poor (Balance)": (
        dict(hh_idx=0, balance=200000, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=10000, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
    ),
    "📈 High vs Low Inflation": (
        dict(hh_idx=0, balance=0, income=5000, inflation=8.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=0, income=5000, inflation=1.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
    ),
    "📉 Boom vs Recession": (
        dict(hh_idx=0, balance=80000, income=6000, inflation=2.0, unemployment=3.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=20000, income=3000, inflation=7.0, unemployment=15.0, interest=3.0, agent_method="none (direct)"),
    ),
    "👤 Different Households (same params)": (
        dict(hh_idx=0, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=5, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
    ),
    "💼 Employed vs Unemployed": (
        dict(hh_idx=0, balance=50000, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=50000, income=0, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
    ),
    "🧠 Direct vs CoT": (
        dict(hh_idx=0, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="cot"),
    ),
    "🧠 Direct vs Debate": (
        dict(hh_idx=0, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="none (direct)"),
        dict(hh_idx=0, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="debate"),
    ),
    "🧠 CoT vs Reflexion": (
        dict(hh_idx=0, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="cot"),
        dict(hh_idx=0, balance=0, income=5000, inflation=2.0, unemployment=5.0, interest=0.4, agent_method="reflexion"),
    ),
}


def _render_comparison():
    import plotly.graph_objects as go

    st.markdown(
        "Configure two scenarios with **any combination of different parameters and agent methods** to compare "
        "how they affect budget allocation. Use a preset or freely edit both sides."
    )

    # ── 预设选择 ──
    preset_name = st.selectbox("Quick Preset", list(_PRESETS.keys()),
                               help="Select a preset to auto-fill both scenarios, then tweak as needed.")
    preset_a, preset_b = _PRESETS[preset_name]

    # ── 高亮差异参数 ──
    if preset_name != "Custom (free edit)":
        diffs = [k for k in preset_a if preset_a[k] != preset_b[k]]
        if diffs:
            labels = {"hh_idx": "Household", "balance": "Balance", "income": "Income",
                      "inflation": "Inflation", "unemployment": "Unemployment", "interest": "Interest",
                      "agent_method": "Agent Method"}
            diff_str = ", ".join(f"**{labels.get(k, k)}**" for k in diffs)
            st.info(f"🔀 This preset varies: {diff_str}. All other parameters are held constant.")

    # ── 两组参数面板 ──
    left, right = st.columns(2)
    with left:
        st.subheader("🅰️ Scenario A")
        params_a = _param_panel("cmp_a", defaults=preset_a, show_agent_method=True)
    with right:
        st.subheader("🅱️ Scenario B")
        params_b = _param_panel("cmp_b", defaults=preset_b, show_agent_method=True)

    # ── 参数差异提示 ──
    _show_param_diff_hint(params_a, params_b)

    run_btn = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

    if not run_btn:
        st.caption("💡 Tip: Try the 🧠 presets to compare different reasoning strategies (Direct vs CoT vs Debate etc.) "
                   "with identical economic parameters — this isolates the effect of the agent method.")
        return

    st.markdown("---")

    # ── 执行两组 ──
    res_a, res_b = None, None
    am_a = params_a.get("agent_method", "none (direct)")
    am_b = params_b.get("agent_method", "none (direct)")
    prog = st.progress(0, text=f"Running Scenario A ({am_a})...")
    try:
        res_a = _run_consumption(params_a)
        prog.progress(50, text=f"Running Scenario B ({am_b})...")
        res_b = _run_consumption(params_b)
        prog.progress(100, text="Done!")
    except Exception as e:
        st.error(f"❌ Failed: {e}")
        st.caption("Check `.env` API key and balance.")
        return

    # ── 并排展示结果 ──
    st.subheader("Results")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🅰️ Scenario A")
        _show_param_tag(params_a)
        _show_result(res_a)
    with col_b:
        st.markdown("#### 🅱️ Scenario B")
        _show_param_tag(params_b)
        _show_result(res_b)

    # ═══════════════════════════════════════
    # 差异对比分析
    # ═══════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 Difference Analysis")

    budgets_a = res_a["step0"].budgets or {}
    budgets_b = res_b["step0"].budgets or {}
    all_cats = list(dict.fromkeys(list(budgets_a.keys()) + list(budgets_b.keys())))

    vals_a = [budgets_a.get(c, 0) for c in all_cats]
    vals_b = [budgets_b.get(c, 0) for c in all_cats]
    diffs = [b - a for a, b in zip(vals_a, vals_b)]

    # 分组柱状图：绝对金额
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="🅰️ Scenario A", x=all_cats, y=vals_a,
                             marker_color="#636EFA",
                             text=[f"${v:,.0f}" for v in vals_a], textposition="outside"))
    fig_bar.add_trace(go.Bar(name="🅱️ Scenario B", x=all_cats, y=vals_b,
                             marker_color="#EF553B",
                             text=[f"${v:,.0f}" for v in vals_b], textposition="outside"))
    fig_bar.update_layout(barmode="group", height=400, yaxis_title="Budget ($)",
                          title="Category Budget: A vs B", margin=dict(t=40, b=40),
                          xaxis_tickangle=-20)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 差异柱状图
    diff_colors = ["#00CC96" if d >= 0 else "#EF553B" for d in diffs]
    fig_diff = go.Figure()
    fig_diff.add_trace(go.Bar(x=all_cats, y=diffs, marker_color=diff_colors,
                              text=[f"{d:+,.0f}" for d in diffs], textposition="outside"))
    fig_diff.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_diff.update_layout(height=350, yaxis_title="Δ Budget (B − A)",
                           title="Budget Difference (Scenario B − A)",
                           margin=dict(t=40, b=40), xaxis_tickangle=-20)
    st.plotly_chart(fig_diff, use_container_width=True)

    # 百分比分配对比
    total_a = res_a["step0"].total_budget or 1
    total_b = res_b["step0"].total_budget or 1
    pcts_a = [budgets_a.get(c, 0) / total_a * 100 for c in all_cats]
    pcts_b = [budgets_b.get(c, 0) / total_b * 100 for c in all_cats]
    pct_diffs = [b - a for a, b in zip(pcts_a, pcts_b)]

    fig_pct = go.Figure()
    fig_pct.add_trace(go.Bar(name="🅰️ A (%)", x=all_cats, y=pcts_a,
                             marker_color="#636EFA",
                             text=[f"{p:.1f}%" for p in pcts_a], textposition="outside"))
    fig_pct.add_trace(go.Bar(name="🅱️ B (%)", x=all_cats, y=pcts_b,
                             marker_color="#EF553B",
                             text=[f"{p:.1f}%" for p in pcts_b], textposition="outside"))
    fig_pct.update_layout(barmode="group", height=350, yaxis_title="Share (%)",
                          title="Allocation Proportion: A vs B",
                          margin=dict(t=40, b=40), xaxis_tickangle=-20)
    st.plotly_chart(fig_pct, use_container_width=True)

    # 比例变化柱状图
    pct_diff_colors = ["#00CC96" if d >= 0 else "#EF553B" for d in pct_diffs]
    fig_pct_diff = go.Figure()
    fig_pct_diff.add_trace(go.Bar(x=all_cats, y=pct_diffs, marker_color=pct_diff_colors,
                                  text=[f"{d:+.1f}pp" for d in pct_diffs], textposition="outside"))
    fig_pct_diff.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_pct_diff.update_layout(height=300, yaxis_title="Δ Share (pp)",
                               title="Proportion Shift (B − A, percentage points)",
                               margin=dict(t=40, b=40), xaxis_tickangle=-20)
    st.plotly_chart(fig_pct_diff, use_container_width=True)

    # ── 汇总 ──
    st.markdown("---")
    st.subheader("Summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Total A", f"${total_a:,.0f}")
    summary_cols[1].metric("Total B", f"${total_b:,.0f}")
    diff_total = total_b - total_a
    pct_change = (diff_total / total_a * 100) if total_a != 0 else 0
    summary_cols[2].metric("Δ Total", f"${diff_total:+,.0f}")
    summary_cols[3].metric("Change", f"{pct_change:+.1f}%")

    # 参数差异表
    st.markdown("#### Parameter Comparison")
    _show_param_table(params_a, params_b)

    # LLM reasoning 对比
    note_a = res_a["step0"].note or ""
    note_b = res_b["step0"].note or ""
    if note_a or note_b:
        st.markdown("#### LLM Reasoning Comparison")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.info(f"🅰️ {note_a}" if note_a else "🅰️ (no reasoning returned)")
        with rc2:
            st.info(f"🅱️ {note_b}" if note_b else "🅱️ (no reasoning returned)")


# ═══════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════

def _show_param_tag(params: dict):
    """在结果上方显示紧凑的参数标签。"""
    parts = []
    if params["income"] > 0:
        parts.append(f"Inc ${params['income']:,.0f}")
    else:
        parts.append("Inc: PSID")
    if params["balance"] > 0:
        parts.append(f"Bal ${params['balance']:,.0f}")
    else:
        parts.append("Bal: PSID")
    parts.append(f"Infl {params['inflation']*100:.1f}%")
    parts.append(f"Unemp {params['unemployment']*100:.1f}%")
    parts.append(f"IR {params['interest']*100:.1f}%")
    am = params.get("agent_method", "")
    if am and am != "none (direct)":
        parts.append(f"🧠 {am}")
    st.caption(f"HH {params['hh_idx']} | " + " | ".join(parts))


def _show_param_diff_hint(a: dict, b: dict):
    """显示两组参数之间的差异提示。"""
    labels = {"hh_idx": "Household", "balance": "Balance", "income": "Income",
              "inflation": "Inflation", "unemployment": "Unemployment", "interest": "Interest",
              "agent_method": "Agent Method"}
    changed = []
    for k in labels:
        va, vb = a.get(k), b.get(k)
        if va != vb:
            if k in ("inflation", "unemployment", "interest"):
                changed.append(f"{labels[k]}: {va*100:.1f}% → {vb*100:.1f}%")
            elif k in ("balance", "income"):
                sa = f"${va:,.0f}" if va > 0 else "PSID"
                sb = f"${vb:,.0f}" if vb > 0 else "PSID"
                changed.append(f"{labels[k]}: {sa} → {sb}")
            elif k == "agent_method":
                changed.append(f"{labels[k]}: {va} → {vb}")
            else:
                changed.append(f"{labels[k]}: {va} → {vb}")
    if changed:
        st.success(f"🔀 **Varying parameters (A→B):** " + " · ".join(changed))
    else:
        st.warning("⚠️ Both scenarios have identical parameters — results will be similar (only LLM stochasticity may differ).")


def _show_param_table(a: dict, b: dict):
    """显示参数对比表，高亮差异。"""
    rows = [
        ("Household", str(a["hh_idx"]), str(b["hh_idx"])),
        ("Balance", f"${a['balance']:,.0f}" if a["balance"] > 0 else "PSID default",
                    f"${b['balance']:,.0f}" if b["balance"] > 0 else "PSID default"),
        ("Income", f"${a['income']:,.0f}" if a["income"] > 0 else "PSID default",
                   f"${b['income']:,.0f}" if b["income"] > 0 else "PSID default"),
        ("Inflation", f"{a['inflation']*100:.1f}%", f"{b['inflation']*100:.1f}%"),
        ("Unemployment", f"{a['unemployment']*100:.1f}%", f"{b['unemployment']*100:.1f}%"),
        ("Interest Rate", f"{a['interest']*100:.1f}%", f"{b['interest']*100:.1f}%"),
        ("Agent Method", a.get("agent_method", "direct"), b.get("agent_method", "direct")),
    ]
    lines = ["| Parameter | Scenario A | Scenario B | Changed? |", "|---|---|---|---|"]
    for label, va, vb in rows:
        diff = "✅ **Yes**" if va != vb else ""
        if va != vb:
            lines.append(f"| **{label}** | **{va}** | **{vb}** | {diff} |")
        else:
            lines.append(f"| {label} | {va} | {vb} | {diff} |")
    st.markdown("\n".join(lines))
