"""
pages/3_Clustering.py  —  K-Means User Segmentation
=====================================================
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import PERSONA_COLORS, load_data, run_clustering

st.set_page_config(page_title="Clustering — UAE Fintech", page_icon="🗂", layout="wide")

st.title("🗂 Clustering — User Persona Segmentation")
st.markdown(
    "**Algorithm:** K-Means (k=4)  \n"
    "**Features:** Age, Income, Financial Literacy, App Sessions, Investment Amount, Satisfaction"
)
st.divider()

df = load_data()
cl = run_clustering(df)
profiles = cl["profiles"]

# ── KPI row ────────────────────────────────────────────────────────────────
for persona, color in PERSONA_COLORS.items():
    row = profiles[profiles["persona"] == persona]
    if len(row) == 0:
        continue
    row = row.iloc[0]

cols = st.columns(4)
for i, (_, row) in enumerate(profiles.iterrows()):
    color = PERSONA_COLORS.get(row["persona"], "#2E86C1")
    cols[i].markdown(f"""
    <div style="background:#1e2130;border:1px solid {color};border-radius:10px;
                padding:14px;text-align:center;border-top:3px solid {color}">
        <div style="font-size:1.4rem;font-weight:800;color:{color}">{int(row['n'])}</div>
        <div style="font-size:0.75rem;color:#8892b0">{row['persona']}</div>
        <div style="font-size:0.7rem;color:#aaa;margin-top:4px">
            Adoption: {row['adoption_rate']:.0%}
        </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Elbow + Doughnut ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Elbow Method — Optimal k")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cl["elbow_k"], y=cl["inertias"],
        mode="lines+markers",
        line=dict(color="#2E86C1", width=2.5),
        marker=dict(
            size=[12 if k == 4 else 7 for k in cl["elbow_k"]],
            color=["#F39C12" if k == 4 else "#2E86C1" for k in cl["elbow_k"]],
        ),
        name="Inertia",
    ))
    fig.add_vline(
        x=4, line_dash="dash", line_color="#F39C12", line_width=1.5,
        annotation_text="Chosen k=4", annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Within-Cluster SSE (Inertia)",
        margin=dict(t=10, b=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The elbow at k=4 shows diminishing inertia gains beyond four clusters, "
        "confirming that four personas best balances granularity and simplicity."
    )

with col2:
    st.markdown("### Cluster Size Distribution")
    fig2 = px.pie(
        profiles, names="persona", values="n",
        color="persona", color_discrete_map=PERSONA_COLORS,
        hole=0.55, template="plotly_dark",
    )
    fig2.update_traces(textinfo="label+percent", textfont_size=11)
    fig2.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10), height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Passive Professionals and Aspiring Investors are the two largest segments, "
        "together representing over half the user base — the primary growth opportunity."
    )

st.divider()

# ── Scatter ────────────────────────────────────────────────────────────────
st.markdown("### Income vs Monthly Investment — Coloured by Persona")
scatter_df = cl["df"]
fig3 = px.scatter(
    scatter_df.sample(min(400, len(scatter_df)), random_state=42),
    x="monthly_income_aed",
    y="monthly_investment_aed",
    color="persona",
    color_discrete_map=PERSONA_COLORS,
    opacity=0.7,
    template="plotly_dark",
    labels={
        "monthly_income_aed":    "Monthly Income (AED)",
        "monthly_investment_aed":"Monthly Investment (AED)",
        "persona":               "Persona",
    },
    hover_data=["age", "financial_literacy_score", "app_sessions_per_week"],
)
fig3.update_traces(marker=dict(size=6))
fig3.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=10, b=10), height=380,
)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Engaged High-Earners (red) dominate the high-income, high-investment quadrant. "
    "Aspiring Investors (orange) show disproportionately high investment relative to "
    "income — the highest-priority acquisition segment."
)

st.divider()

# ── Radar + Profiles table ─────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Normalised Cluster Radar")
    radar_norm = cl["radar_norm"]
    radar_labels = cl["radar_cols"]

    fig4 = go.Figure()
    for persona in radar_norm.index:
        values = radar_norm.loc[persona].tolist()
        values_closed = values + [values[0]]
        labels_closed = radar_labels + [radar_labels[0]]
        fig4.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=PERSONA_COLORS.get(persona, "#2E86C1") + "30",
            line=dict(color=PERSONA_COLORS.get(persona, "#2E86C1"), width=2),
            name=persona,
        ))
    fig4.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2d3250"),
            angularaxis=dict(gridcolor="#2d3250"),
            bgcolor="#1e2130",
        ),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=10, b=60), height=380,
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "Aspiring Investors score highest on Literacy, Sessions, and Adoption. "
        "Engaged High-Earners lead on Income and Investment but have moderate literacy."
    )

with col4:
    st.markdown("### Cluster Profile Table")
    display_cols = {
        "persona":        "Persona",
        "n":              "Users",
        "avg_age":        "Avg Age",
        "avg_income":     "Avg Income (AED)",
        "avg_literacy":   "Literacy (1–5)",
        "avg_sessions":   "Sessions/wk",
        "avg_investment": "Avg Invest (AED)",
        "adoption_rate":  "Adoption Rate",
    }
    disp = profiles[list(display_cols.keys())].rename(columns=display_cols)
    disp["Avg Income (AED)"]  = disp["Avg Income (AED)"].map(lambda v: f"{v:,.0f}")
    disp["Avg Invest (AED)"]  = disp["Avg Invest (AED)"].map(lambda v: f"{v:,.0f}")
    disp["Adoption Rate"]     = disp["Adoption Rate"].map(lambda v: f"{v:.0%}")

    st.dataframe(disp.set_index("Persona"), use_container_width=True, height=230)

    st.divider()
    st.markdown("### 💡 Marketing Strategy per Persona")
    for _, row in profiles.iterrows():
        persona = row["persona"]
        color   = PERSONA_COLORS.get(persona, "#2E86C1")
        strategies = {
            "Cautious Savers":       "Education-led onboarding, low-risk entry points, savings goals gamification.",
            "Aspiring Investors":    "Investment challenges, leaderboard features, Auto-Invest push notifications.",
            "Passive Professionals": "Re-engagement nudges, round-up investing, automated savings reminders.",
            "Engaged High-Earners":  "Premium portfolio tools, Sharia-compliant product upsell, diversified ETFs.",
        }
        st.markdown(f"""
        <div style="border-left:3px solid {color};padding:8px 12px;margin-bottom:8px;background:#1e2130;border-radius:0 6px 6px 0">
            <strong style="color:{color}">{persona}</strong><br>
            <span style="font-size:0.8rem;color:#aaa">{strategies.get(persona,'')}</span>
        </div>""", unsafe_allow_html=True)
