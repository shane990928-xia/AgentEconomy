"""页面: 系统能力概览"""
import streamlit as st
import json
from pathlib import Path


def render():
    st.title("🏛️ AgentEconomy Platform")
    st.markdown("An LLM-driven multi-agent simulation platform for economic research. "
                "The system supports three types of experiments, covering macro/micro economics, financial markets, and supply chain networks.")

    st.markdown("---")

    # ── 平台能力总览 ──
    st.subheader("What Can This Platform Do?")

    st.markdown("""
| | Macroeconomic Simulation | Stock Market Simulation | Supply Chain Analysis |
|---|---|---|---|
| **Goal** | Reproduce macro stylized facts | Model price discovery & trading | Evaluate supplier selection |
| **Agents** | 300 Households + 66 Firms | 15,000+ Traders | FactSet Entity Graph |
| **LLM Role** | Consumption & employment decisions | Institutional investing | Supplier assessment |
| **Validation** | Phillips Curve (r = −0.56) | Price discovery accuracy | Precision / Recall vs ground-truth |
| **Data** | PSID + O*NET + BLS | Real market data (JNJ) | FactSet supply chain |
    """)

    st.markdown("---")

    # ── Agent Method（推理策略） ──
    st.subheader("🧠 Agent Reasoning Methods")
    st.markdown(
        "All LLM-driven decisions (consumption, employment, trading, supplier assessment) "
        "go through a configurable **Agent Method** layer. This controls *how* the LLM reasons "
        "before producing a final answer — from simple single-pass to multi-round self-correction."
    )

    methods = {
        "Direct": {
            "calls": 1,
            "desc": "Single LLM call, no extra reasoning. Fastest and cheapest baseline.",
            "flow": "Prompt → LLM → Answer",
        },
        "CoT (Chain-of-Thought)": {
            "calls": 1,
            "desc": "Instructs the LLM to think step-by-step before answering. Same cost as Direct but often better quality.",
            "flow": "Prompt → LLM (think step-by-step) → Answer",
        },
        "Self-Refine": {
            "calls": "2–3",
            "desc": "LLM generates an initial answer, audits it, then refines if needed. Catches obvious errors.",
            "flow": "Prompt → Initial Answer → Audit → Refined Answer (if needed)",
        },
        "Reflexion": {
            "calls": "2–3",
            "desc": "LLM drafts, then reflects on its own output to identify issues, and revises. Similar to Self-Refine but with explicit reflection.",
            "flow": "Prompt → Draft → Reflection → Revised Answer (if needed)",
        },
        "Debate": {
            "calls": 3,
            "desc": "Internal debate: a proposer drafts, a critic challenges, a judge synthesizes the final answer.",
            "flow": "Prompt → Proposer → Critic → Judge → Final Answer",
        },
        "Discussion": {
            "calls": 4,
            "desc": "Three experts with different focuses (correctness, practicality, edge cases) each propose answers, then a moderator synthesizes.",
            "flow": "Prompt → Expert×3 → Moderator → Final Answer",
        },
    }

    cols = st.columns(3)
    for i, (name, info) in enumerate(methods.items()):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.caption(f"LLM calls: {info['calls']}")
            st.markdown(f"{info['desc']}")
            st.code(info["flow"], language=None)

    st.info(
        "💡 **How to experiment**: In the **🛒 Live Consumption Demo** page (Comparison mode), "
        "you can select different Agent Methods for Scenario A and B to compare how reasoning "
        "strategies affect budget allocation decisions."
    )

    st.markdown("---")

    # ── 三种实验 ──
    st.subheader("Experiments")

    tab1, tab2, tab3 = st.tabs(["🏛️ Macroeconomic", "📈 Stock Market", "🔗 Supply Chain"])

    with tab1:
        st.markdown("""
        **Macroeconomic Simulation** — 300 PSID-initialized households and 66 multi-sector firms interact
        over 30 simulated months. LLM agents make consumption, savings, and employment decisions
        based on their persona, financial state, and macro indicators.

        Key results:
        - Phillips Curve reproduced (r = −0.56) without hand-crafted behavioral rules
        - Skill-based labor market matching with asymmetric z-score loss
        - Government taxation & redistribution loop
        - Gini coefficient dynamics tracking wealth inequality
        """)
        c1, c2, c3 = st.columns(3)
        c1.metric("Households", "300")
        c2.metric("Firms", "66")
        c3.metric("Duration", "30 months")

    with tab2:
        st.markdown("""
        **Stock Market Simulation** — 15,000+ heterogeneous trading agents operate in a NASDAQ-style
        continuous double auction. LLM-powered institutional investors make news-driven and
        earnings-driven trading decisions at nanosecond resolution.
        """)

        # 加载 LLM cache 统计
        cache_path = Path(__file__).resolve().parents[1] / "marketsim" / "llm_cache.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
                meta = cache.get("metadata", {})
                st.info(f"📊 LLM calls: {meta.get('total_calls', '?')} | Config: {meta.get('simulation_config', '?')}")

                results = cache.get("result_list_cache", [])
                if results and results[0]:
                    sample = results[0][0]
                    with st.expander("💭 Sample LLM Agent Reasoning"):
                        st.markdown(f"**Price prediction:** ${sample.get('price', '?')}")
                        reason = sample.get("reason", "")
                        st.markdown(f"**Reasoning:** {reason[:300]}...")
            except Exception:
                pass

        c1, c2 = st.columns(2)
        c1.metric("Agents", "15,000+")
        c2.metric("Case Study", "JNJ")

    with tab3:
        st.markdown("""
        **Supply Chain Analysis** — LLM-powered supplier assessment pipeline using FactSet entity
        relationship data. The system profiles company characteristics, scores candidates through
        a 6-stage selection pipeline, and validates against ground-truth supply chain relationships.
        """)

        # 加载实验结果
        result_dir = Path(__file__).resolve().parents[1] / "supply_chain" / "result"
        result_files = sorted(result_dir.glob("experiment_*.json"))
        if result_files:
            try:
                with open(result_files[-1]) as f:
                    exp = json.load(f)
                ev = exp.get("evaluation", {})
                basic = ev.get("basic_metrics", {})

                st.info(f"📊 Entity: {exp.get('target_entity_id', '?')} | Year: {exp.get('decision_year', '?')}")

                with st.expander("📋 Evaluation Metrics"):
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy", f"{basic.get('accuracy', 0):.1%}")
                    m2.metric("Candidates", f"{basic.get('candidate_count', 0)}")
                    m1.metric("Selected", f"{basic.get('selected_count', 0)}")
                    m2.metric("Actual Suppliers", f"{basic.get('actual_count', 0)}")

                report = exp.get("selection_report", {})
                selected = report.get("selected_suppliers", [])
                if selected:
                    with st.expander("🏭 Selected Suppliers"):
                        for s in selected[:5]:
                            name = s.get("entity_name", s.get("entity_id", "?"))
                            score = s.get("total_score", s.get("score", 0))
                            st.markdown(f"- **{name}** (score: {score:.1f})")
            except Exception:
                pass

            st.metric("Experiments", f"{len(result_files)} runs")

    st.markdown("---")

    # ── 技术栈 ──
    st.subheader("Technology Stack")
    st.markdown("""
| Component | Technology | Usage |
|---|---|---|
| LLM Engine | DeepSeek-V3 / GPT via LiteLLM | Agent decision-making across all experiments |
| Agent Methods | CoT / Debate / Discussion / Reflexion / Self-Refine | Configurable reasoning strategies |
| Vector DB | Qdrant + MiniLM-L6-v2 | Semantic product matching (consumption pipeline) |
| Distributed Compute | Ray | Parallel agent execution |
| Async Concurrency | asyncio (400 parallel) | High-throughput LLM calls |
| Data Sources | PSID, O*NET, BLS, FactSet | Experiment-specific initialization |
    """)

    st.markdown("---")
    st.subheader("Research Contributions")
    st.markdown("""
1. First LLM-driven macroeconomic ABM that replaces utility functions with natural language reasoning, reproducing the Phillips curve without hand-crafted behavioral rules.
2. Large-scale stock market simulation with LLM institutional investors making news-driven trading decisions in a realistic NASDAQ auction environment.
3. LLM-powered supplier selection pipeline with personality profiling, validated against ground-truth FactSet supply chain data.
    """)
