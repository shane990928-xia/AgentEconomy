"""页面1: 系统概览"""
import streamlit as st


def render():
    st.title("System Overview")
    st.markdown("---")

    # ── 核心指标 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households", "300", help="PSID-initialized heterogeneous agents")
    c2.metric("Firms", "66", help="20 mfg + 4 retail + 42 service")
    c3.metric("Simulation", "30 months", help="Excluding 4 preheat months")
    c4.metric("LLM", "DeepSeek-V3", help="Budget allocation & purchase decisions")

    st.markdown("---")

    # ── 架构图 ──
    st.subheader("System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Simulation Engine                        │
    │                  (Monthly Loop × 30)                        │
    ├──────────┬──────────┬──────────┬────────────┬──────────────┤
    │ 300      │ 66 Firms │Government│   Bank     │  Markets     │
    │Households│ (Mfg/Ret │(Tax &    │(Deposits & │              │
    │          │  /Svc)   │ Redist)  │ Interest)  │              │
    ├──────────┴──────────┴──────────┴────────────┤              │
    │                Agent Layer                   │              │
    │  ┌─────────────────────────────────────┐    │  ┌──────────┐│
    │  │  LLM Decision Engine (DeepSeek)     │    │  │  Labor   ││
    │  │  • Budget allocation (Step0)        │    │  │  Market  ││
    │  │  • Need generation  (Step1)         │    │  ├──────────┤│
    │  │  • Purchase decision (Step3)        │    │  │ Product  ││
    │  │  • Job application / offer          │    │  │  Market  ││
    │  └──────────┬──────────────────────────┘    │  ├──────────┤│
    │             │                                │  │Intermed. ││
    │  ┌──────────▼──────────────────────────┐    │  │  Goods   ││
    │  │  Qdrant Vector DB + MiniLM-L6-v2    │    │  │Procurem. ││
    │  │  (Semantic product matching, Step2)  │    │  └──────────┘│
    │  └─────────────────────────────────────┘    │              │
    ├──────────────────────────────────────────────┴──────────────┤
    │                     Data Layer                              │
    │  PSID Survey │ O*NET Skills │ BLS IO Table │ BLS Products  │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    # ── 消费流水线 ──
    st.subheader("Household Consumption Pipeline")
    cols = st.columns(5)
    steps = [
        ("Step 0", "Budget\nAllocation", "LLM divides income across 6 categories based on persona & macro indicators"),
        ("Step 1", "Need\nGeneration", "LLM generates specific need descriptions for 20 retail sub-categories"),
        ("Step 2", "Vector\nRetrieval", "Qdrant + MiniLM-L6-v2 matches products to need descriptions (top-k)"),
        ("Step 3", "Purchase\nDecision", "LLM selects final products from candidates, considering price & relevance"),
        ("Step 4", "Validation\n& Settlement", "Budget constraint check, order creation, payment via EconomicCenter"),
    ]
    for col, (title, label, desc) in zip(cols, steps):
        col.markdown(f"**{title}**")
        col.info(label)
        col.caption(desc)

    st.markdown("---")

    # ── 劳动力市场 ──
    st.subheader("Labor Market Mechanism")
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("""
        **Job Matching (Skill-Based)**
        - Z-score matching with asymmetric penalties
        - Over-qualified: weight 0.1 (welcome)
        - Slightly under: weight 0.3 (trainable)
        - Severely under: weight 0.5 (hard gap)
        - Workers apply to top-3 ranked jobs
        """)
    with l2:
        st.markdown("""
        **LLM-Driven Decisions**
        - Households use LLM to decide which jobs to apply for
        - LLM evaluates offers considering wage, skill match, persona
        - Multi-round deferred acceptance with backup candidates
        - Firms set posted wages from O*NET data
        """)

    st.markdown("---")

    # ── 数据来源 ──
    st.subheader("Data Sources")
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown("**PSID**\n\nPanel Study of Income Dynamics — 300 household profiles with demographics, income, wealth, expenditure")
    d2.markdown("**O*NET**\n\nOccupational skills & abilities database — skill requirements for job matching")
    d3.markdown("**BLS IO Table**\n\nInter-industry input-output relationships — multi-sector production network")
    d4.markdown("**BLS Products**\n\nConsumer product catalog — 10,000+ SKUs with NAICS classification")
