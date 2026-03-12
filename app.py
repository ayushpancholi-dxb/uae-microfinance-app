"""
app.py  —  Main entry point / Overview page
============================================
UAE Personal Finance & Micro-Investment App
MBA Data Analytics — Individual PBL
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.data_loader import (
    PERSONA_COLORS,
    load_data,
    run_classification,
    run_clustering,
    run_regression,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Fintech App — Analytics",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: #1e2130; border: 1px solid #2d3250; border-radius: 10px;
        padding: 18px 20px; text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: 800; line-height: 1.1; }
    .kpi-label { font-size: 0.75rem; color: #8892b0; text-transform: uppercase;
                 letter-spacing: 0.5px; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #85C1E9;
        border-bottom: 1px solid #2d3250; padding-bottom: 6px; margin-bottom: 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/investment-portfolio.png", width=64)
    st.title("UAE Fintech App")
    st.caption("MBA Data Analytics — Individual PBL")
    st.divider()
    st.markdown("**Dr. Anshul Gupta**")
    st.caption("Navigate using the pages below ↓")
    st.divider()
    st.markdown("""
    **Pages**
    - 🏠 Overview *(this page)*
    - 🔍 EDA
    - 🎯 Classification
    - 🗂 Clustering
    - 🔗 Association Rules
    - 📈 Regression
    """)

# ── Load data ──────────────────────────────────────────────────────────────
df  = load_data()
clf = run_classification(df)
reg = run_regression(df)
cl  = run_clustering(df)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏠 UAE Personal Finance & Micro-Investment App")
st.markdown(
    "**Business Question:** Which users are most likely to adopt micro-investment "
    "features, and what feature combinations drive higher monthly investment?"
)
st.divider()

# ── KPI Row ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Dataset KPIs</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Respondents",       f"{len(df):,}")
k2.metric("Adoption Rate",     f"{df['will_adopt_microinvestment'].mean():.1%}")
k3.metric("Avg Investment",    f"AED {df['monthly_investment_aed'].mean():,.0f}")
k4.metric("Avg Literacy",      f"{df['financial_literacy_score'].mean():.1f} / 5")
k5.metric("Sharia Preference", f"{df['sharia_compliant_preference'].mean():.1%}")
k6.metric("Avg Sessions/wk",   f"{df['app_sessions_per_week'].mean():.1f}")

st.divider()

# ── Algorithm summary cards ────────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 Algorithm Results at a Glance</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#2E86C1">🎯 {clf['accuracy']}%</div>
        <div class="kpi-label">Classification Accuracy</div>
        <div style="font-size:0.7rem;color:#8892b0;margin-top:6px">Random Forest · 200 trees</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#F39C12">🗂 4</div>
        <div class="kpi-label">User Personas (K-Means)</div>
        <div style="font-size:0.7rem;color:#8892b0;margin-top:6px">Cautious · Aspiring · Passive · Engaged</div>
    </div>""", unsafe_allow_html=True)
with c3:
    arm_df = __import__("utils.data_loader", fromlist=["run_arm"]).run_arm(df)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#27AE60">🔗 {len(arm_df['rules'])}</div>
        <div class="kpi-label">Association Rules Found</div>
        <div style="font-size:0.7rem;color:#8892b0;margin-top:6px">Apriori · min support 30%</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:#1abc9c">📈 {reg['r2']}</div>
        <div class="kpi-label">Regression R²</div>
        <div style="font-size:0.7rem;color:#8892b0;margin-top:6px">Linear · MAE = AED {reg['mae']:.0f}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Two charts side by side ────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="section-header">Adoption Rate by Financial Literacy</div>', unsafe_allow_html=True)
    lit = (
        df.groupby("financial_literacy_score")["will_adopt_microinvestment"]
        .mean()
        .reset_index()
    )
    lit.columns = ["Literacy Score", "Adoption Rate"]
    fig = px.bar(
        lit, x="Literacy Score", y="Adoption Rate",
        color="Adoption Rate",
        color_continuous_scale=["#1B4F72", "#2E86C1", "#1abc9c", "#27AE60"],
        text=lit["Adoption Rate"].map(lambda v: f"{v:.0%}"),
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_tickformat=".0%", yaxis_range=[0, 0.85],
        margin=dict(t=20, b=20), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown('<div class="section-header">User Segment Distribution</div>', unsafe_allow_html=True)
    profiles = cl["profiles"]
    fig2 = px.pie(
        profiles, names="persona", values="n",
        color="persona",
        color_discrete_map=PERSONA_COLORS,
        hole=0.55,
        template="plotly_dark",
    )
    fig2.update_traces(textinfo="label+percent", textfont_size=12)
    fig2.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=20, b=60), height=320,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Dataset preview ────────────────────────────────────────────────────────
with st.expander("📋 View raw dataset (first 20 rows)"):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Full dataset: {len(df)} rows × {len(df.columns)} columns — `data/uae_finapp_dataset.csv`")
