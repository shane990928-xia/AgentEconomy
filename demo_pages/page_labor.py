"""页面: 劳动力匹配可视化"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render():
    st.title("🔧 Labor Market: Skill-Based Matching")
    st.markdown("The labor market uses an asymmetric z-score matching loss to pair workers with jobs based on O*NET skill/ability profiles.")

    st.markdown("---")

    # ── 匹配机制说明 ──
    st.subheader("Matching Loss Function")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        For each skill dimension, the system computes:

        $$z = \\frac{\\text{worker\\_value} - \\mu_{\\text{job}}}{\\sigma_{\\text{job}}}$$

        Then applies **asymmetric penalties**:

        | Condition | Interpretation | Weight |
        |---|---|---|
        | $z > 0$ | Over-qualified | **0.1** (welcome) |
        | $-1 < z \\leq 0$ | Slightly under | **0.3** (trainable) |
        | $z \\leq -1$ | Severely under | **0.5** (hard gap) |

        $$\\text{loss} = \\sum_{s} \\text{importance}_s \\times w(z_s) \\times z_s^2$$

        Workers apply to the **top-3 lowest-loss** jobs.
        """)

    with col2:
        # 可视化惩罚函数
        z = np.linspace(-3, 3, 200)
        penalty = np.where(z > 0, 0.1 * z ** 2,
                           np.where(z > -1, 0.3 * z ** 2, 0.5 * z ** 2))

        fig = go.Figure()
        # 三段分别画，颜色不同
        mask_over = z > 0
        mask_slight = (z > -1) & (z <= 0)
        mask_severe = z <= -1

        fig.add_trace(go.Scatter(x=z[mask_severe], y=penalty[mask_severe],
                                 mode="lines", name="Severely under (w=0.5)",
                                 line=dict(color="#EF553B", width=3)))
        fig.add_trace(go.Scatter(x=z[mask_slight], y=penalty[mask_slight],
                                 mode="lines", name="Slightly under (w=0.3)",
                                 line=dict(color="#FFA15A", width=3)))
        fig.add_trace(go.Scatter(x=z[mask_over], y=penalty[mask_over],
                                 mode="lines", name="Over-qualified (w=0.1)",
                                 line=dict(color="#00CC96", width=3)))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=-1, line_dash="dot", line_color="gray",
                      annotation_text="z = -1", annotation_position="top left")
        fig.update_layout(title="Asymmetric Penalty Function",
                          xaxis_title="z-score (worker − requirement)",
                          yaxis_title="Penalty",
                          height=350, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 交互式匹配模拟 ──
    st.subheader("Interactive Matching Simulation")
    st.markdown("Simulate a worker matching against 5 jobs with different skill requirements.")

    np.random.seed(42)
    skills = ["Programming", "Communication", "Analysis", "Leadership", "Creativity"]

    st.markdown("**Worker Skill Profile** (drag sliders)")
    worker = {}
    wcols = st.columns(5)
    for i, s in enumerate(skills):
        worker[s] = wcols[i].slider(s, 0.0, 100.0, float(np.random.randint(40, 80)), 1.0, key=f"w_{s}")

    # 生成 5 个岗位
    jobs = []
    job_names = ["Software Engineer", "Project Manager", "Data Analyst", "Sales Lead", "UX Designer"]
    job_profiles = [
        {"Programming": 75, "Communication": 50, "Analysis": 70, "Leadership": 40, "Creativity": 45},
        {"Programming": 30, "Communication": 80, "Analysis": 50, "Leadership": 85, "Creativity": 40},
        {"Programming": 60, "Communication": 45, "Analysis": 85, "Leadership": 35, "Creativity": 50},
        {"Programming": 20, "Communication": 90, "Analysis": 40, "Leadership": 70, "Creativity": 55},
        {"Programming": 50, "Communication": 55, "Analysis": 45, "Leadership": 30, "Creativity": 90},
    ]

    # 计算匹配损失
    results = []
    for name, profile in zip(job_names, job_profiles):
        total_loss = 0
        details = {}
        for s in skills:
            req_mean = profile[s]
            req_std = max(10, req_mean * 0.2)  # 模拟标准差
            z = (worker[s] - req_mean) / req_std
            if z > 0:
                w = 0.1
                label = "Over ✅"
            elif z > -1:
                w = 0.3
                label = "Slight ⚠️"
            else:
                w = 0.5
                label = "Severe ❌"
            loss = w * z ** 2
            total_loss += loss
            details[s] = {"z": z, "w": w, "loss": loss, "label": label}
        results.append({"name": name, "loss": total_loss, "details": details, "profile": profile})

    results.sort(key=lambda x: x["loss"])

    # 排名展示
    st.markdown("**Matching Results** (sorted by loss, lower = better match)")

    for rank, r in enumerate(results):
        color = "🟢" if rank < 3 else "⚪"
        badge = " → **APPLY**" if rank < 3 else ""
        with st.expander(f"{color} #{rank+1} {r['name']} — Loss: {r['loss']:.2f}{badge}", expanded=(rank == 0)):
            dcols = st.columns(5)
            for i, s in enumerate(skills):
                d = r["details"][s]
                dcols[i].metric(s, f"z={d['z']:.2f}", d["label"])

    # 雷达图
    st.markdown("---")
    st.subheader("Skill Profile Comparison")
    best = results[0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=[worker[s] for s in skills] + [worker[skills[0]]],
        theta=skills + [skills[0]],
        fill="toself", name="Worker", line=dict(color="#636EFA"),
    ))
    fig2.add_trace(go.Scatterpolar(
        r=[best["profile"][s] for s in skills] + [best["profile"][skills[0]]],
        theta=skills + [skills[0]],
        fill="toself", name=f"Best Match: {best['name']}", line=dict(color="#EF553B"),
    ))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                       height=400, margin=dict(t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)
