"""
AgentEconomy Demo Dashboard
启动: streamlit run demo_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="AgentEconomy Demo",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 侧边栏导航 ──
st.sidebar.title("🏛️ AgentEconomy")
st.sidebar.markdown("LLM-Driven Multi-Agent\nEconomic Simulation Platform")
page = st.sidebar.radio(
    "Navigate",
    ["🔗 System Integration",
     "📋 System Overview",
     "📈 Macroeconomic Dynamics",
     "🔬 Stylized Facts",
     "🔧 Labor Market Matching",
     "🛒 Live Consumption Demo",
     "⚖️ LLM vs Rule Ablation"],
)

# ── 页面路由 ──
if page == "🔗 System Integration":
    from demo_pages import page_system
    page_system.render()
elif page == "📋 System Overview":
    from demo_pages import page_overview
    page_overview.render()
elif page == "📈 Macroeconomic Dynamics":
    from demo_pages import page_macro
    page_macro.render()
elif page == "🔬 Stylized Facts":
    from demo_pages import page_stylized
    page_stylized.render()
elif page == "🔧 Labor Market Matching":
    from demo_pages import page_labor
    page_labor.render()
elif page == "🛒 Live Consumption Demo":
    from demo_pages import page_consumption
    page_consumption.render()
elif page == "⚖️ LLM vs Rule Ablation":
    from demo_pages import page_ablation
    page_ablation.render()
