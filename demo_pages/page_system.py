"""页面: 三模块系统集成概览"""
import streamlit as st
import json
from pathlib import Path


def render():
    st.title("🔗 Integrated System: Three Modules")
    st.markdown("The platform integrates three complementary simulation modules, each targeting a different layer of the economic system.")

    st.markdown("---")

    # ── 三模块架构 ──
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                    AgentEconomy Platform                        │
    ├──────────────────┬──────────────────┬────────────────────────────┤
    │  🏛️ AgentEconomy  │  📈 MarketSim     │  🔗 SupplyChain            │
    │  Macro Simulation │  Stock Market     │  Supplier Selection        │
    │                  │  Simulation       │  & Network Reconstruction  │
    ├──────────────────┼──────────────────┼────────────────────────────┤
    │ 300 Households   │ 15K+ Traders     │ FactSet Entity Graph       │
    │ 66 Firms         │ NASDAQ Auction   │ LLM Personality Profiles   │
    │ LLM Consumption  │ LLM Inst.Invest. │ LLM Supplier Assessment    │
    │ Skill Matching   │ News-driven      │ Precision/Recall Eval      │
    │ PSID + O*NET     │ JNJ Case Study   │ Ground-truth Validation    │
    └──────────────────┴──────────────────┴────────────────────────────┘
    ```
    """)

    # ── 三列展示 ──
    col1, col2, col3 = st.columns(3)

    # ═══ Module 1: AgentEconomy ═══
    with col1:
        st.subheader("🏛️ AgentEconomy")
        st.markdown("**Macroeconomic Simulation**")
        st.markdown("""
        - 300 PSID-initialized household agents
        - 66 multi-sector firms (mfg/retail/service)
        - LLM-driven consumption & employment
        - Skill-based labor market matching
        - Government taxation & redistribution
        - 30-month simulation with preheat
        """)
        st.success("✅ Phillips Curve reproduced (r = −0.56)")
        st.metric("Agents", "300 households + 66 firms")

    # ═══ Module 2: MarketSim ═══
    with col2:
        st.subheader("📈 MarketSim")
        st.markdown("**Stock Market Simulation**")
        st.markdown("""
        - 15,000+ heterogeneous trading agents
        - NASDAQ continuous double auction
        - LLM institutional investors
        - News & earnings-driven decisions
        - Nanosecond order matching
        - Real market data (JNJ case study)
        """)

        # 加载 LLM cache 统计
        cache_path = Path(__file__).resolve().parents[1] / "marketsim" / "llm_cache.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
                meta = cache.get("metadata", {})
                st.info(f"📊 LLM calls: {meta.get('total_calls', '?')} | Config: {meta.get('simulation_config', '?')}")

                # 展示一条 LLM 推理
                results = cache.get("result_list_cache", [])
                if results and results[0]:
                    sample = results[0][0]
                    with st.expander("💭 Sample LLM Agent Reasoning"):
                        st.markdown(f"**Price prediction:** ${sample.get('price', '?')}")
                        reason = sample.get("reason", "")
                        st.markdown(f"**Reasoning:** {reason[:300]}...")
            except Exception:
                pass
        st.metric("Agents", "15,000+ traders")

    # ═══ Module 3: SupplyChain ═══
    with col3:
        st.subheader("🔗 SupplyChain")
        st.markdown("**Supplier Selection & Network**")
        st.markdown("""
        - LLM-powered supplier assessment
        - FactSet entity relationship data
        - Company personality profiling
        - 6-stage selection pipeline
        - Ground-truth validation (precision/recall)
        - Network reconstruction from IO tables
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
                ranking = ev.get("ranking_metrics", {})

                st.info(f"📊 Entity: {exp.get('target_entity_id', '?')} | Year: {exp.get('decision_year', '?')}")

                with st.expander("📋 Evaluation Metrics"):
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy", f"{basic.get('accuracy', 0):.1%}")
                    m2.metric("Candidates", f"{basic.get('candidate_count', 0)}")
                    m1.metric("Selected", f"{basic.get('selected_count', 0)}")
                    m2.metric("Actual Suppliers", f"{basic.get('actual_count', 0)}")

                # 展示 selection report
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

    # ── 共享 LLM 基础设施 ──
    st.subheader("Shared Infrastructure")
    st.markdown("""
    | Component | Technology | Shared Across |
    |---|---|---|
    | LLM Engine | DeepSeek-V3 / GPT via LiteLLM Router | All 3 modules |
    | Vector DB | Qdrant + MiniLM-L6-v2 | AgentEconomy (product matching) |
    | Distributed Compute | Ray | AgentEconomy + MarketSim |
    | Data Sources | PSID, O*NET, BLS, FactSet | Module-specific |
    | Async Concurrency | asyncio (400 parallel) | AgentEconomy + MarketSim |
    """)

    st.markdown("---")
    st.subheader("Research Contributions")
    st.markdown("""
    1. **AgentEconomy**: First LLM-driven macroeconomic ABM that replaces utility functions with natural language reasoning, reproducing the Phillips curve without hand-crafted behavioral rules.
    2. **MarketSim**: Large-scale stock market simulation with LLM institutional investors making news-driven trading decisions in a realistic NASDAQ auction environment.
    3. **SupplyChain**: LLM-powered supplier selection pipeline with personality profiling, validated against ground-truth FactSet supply chain data.
    """)
