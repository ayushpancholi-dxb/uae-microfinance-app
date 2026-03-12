"""
pages/1_EDA.py  —  Exploratory Data Analysis
=============================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.data_loader import FEATURE_COLS, FEATURE_LABELS, load_data

st.set_page_config(page_title="EDA — UAE Fintech", page_icon="🔍", layout="wide")

st.title("🔍 Exploratory Data Analysis")
st.markdown("Distributions, adoption patterns, and feature usage across the 500-respondent dataset.")
st.divider()

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 EDA Filters")
    emp_filter = st.multiselect(
        "Employment Status",
        options=df["employment_status"].unique().tolist(),
        default=df["employment_status"].unique().tolist(),
    )
    nat_filter = st.multiselect(
        "Nationality",
        options=df["nationality"].unique().tolist(),
        default=df["nationality"].unique().tolist(),
    )

filtered = df[
    df["employment_status"].isin(emp_filter) &
    df["nationality"].isin(nat_filter)
]
st.caption(f"Showing **{len(filtered):,}** of {len(df):,} respondents after filters")

# ── Row 1: Adoption by employment + literacy ───────────────────────────────
st.markdown("### Adoption Rate Breakdown")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**By Employment Status**")
    emp_adopt = (
        filtered.groupby("employment_status")["will_adopt_microinvestment"]
        .mean()
        .reset_index()
        .sort_values("will_adopt_microinvestment", ascending=False)
    )
    emp_adopt.columns = ["Employment", "Adoption Rate"]
    fig = px.bar(
        emp_adopt, x="Employment", y="Adoption Rate",
        color="Adoption Rate",
        color_continuous_scale=["#1B4F72", "#2E86C1", "#1abc9c", "#27AE60"],
        text=emp_adopt["Adoption Rate"].map(lambda v: f"{v:.1%}"),
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_tickformat=".0%", yaxis_range=[0, 0.8],
        margin=dict(t=10, b=10), height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Mid-career professionals show the highest adoption rate, driven by "
        "income stability and greater exposure to investment products."
    )

with col2:
    st.markdown("**By Financial Literacy Score (1 = Low, 5 = High)**")
    lit_adopt = (
        filtered.groupby("financial_literacy_score")["will_adopt_microinvestment"]
        .mean()
        .reset_index()
    )
    lit_adopt.columns = ["Literacy Score", "Adoption Rate"]
    fig2 = px.bar(
        lit_adopt, x="Literacy Score", y="Adoption Rate",
        color="Adoption Rate",
        color_continuous_scale=["#E74C3C", "#F39C12", "#2E86C1", "#1abc9c", "#27AE60"],
        text=lit_adopt["Adoption Rate"].map(lambda v: f"{v:.1%}"),
        template="plotly_dark",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        coloraxis_showscale=False,
        yaxis_tickformat=".0%", yaxis_range=[0, 0.85],
        margin=dict(t=10, b=10), height=320,
        xaxis=dict(tickmode="linear"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Adoption rises steeply from score 3 to 5, confirming financial literacy "
        "as the strongest single predictor of micro-investment adoption."
    )

st.divider()

# ── Row 2: Income distribution ─────────────────────────────────────────────
st.markdown("### Income Distribution by Adoption Status")
fig3 = go.Figure()
for label, val, color in [
    ("Will NOT Adopt", 0, "#85C1E9"),
    ("Will Adopt",     1, "#1B4F72"),
]:
    subset = filtered[filtered["will_adopt_microinvestment"] == val]["monthly_income_aed"]
    fig3.add_trace(go.Histogram(
        x=subset, name=label, nbinsx=35,
        marker_color=color, opacity=0.8,
    ))
median_val = filtered["monthly_income_aed"].median()
fig3.add_vline(
    x=median_val, line_dash="dash", line_color="#F39C12", line_width=2,
    annotation_text=f"Median AED {median_val:,.0f}",
    annotation_position="top right",
)
fig3.update_layout(
    barmode="overlay",
    template="plotly_dark",
    xaxis_title="Monthly Income (AED)",
    yaxis_title="Count",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=10, b=10), height=320,
)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Adopters are concentrated in the AED 10k–25k band. The substantial overlap "
    "confirms that income alone does not predict adoption — literacy and behaviour "
    "matter equally, justifying a multivariate modelling approach."
)

st.divider()

# ── Row 3: Feature usage heatmap + Nationality savings ────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Feature Usage — Adopters vs Non-Adopters")
    feat_rates = filtered.groupby("will_adopt_microinvestment")[FEATURE_COLS].mean()
    feat_rates.index = ["Non-Adopter", "Adopter"]
    feat_rates.columns = FEATURE_LABELS

    fig4 = go.Figure(data=go.Heatmap(
        z=feat_rates.values,
        x=FEATURE_LABELS,
        y=["Non-Adopter", "Adopter"],
        colorscale="Blues",
        zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in feat_rates.values],
        texttemplate="%{text}",
        textfont={"size": 13},
        showscale=True,
    ))
    fig4.update_layout(
        template="plotly_dark",
        margin=dict(t=10, b=10), height=220,
        xaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "Adopters use Auto-Invest and Portfolio View significantly more, "
        "confirming that investment-adjacent features are the strongest "
        "engagement gateway to micro-investment adoption."
    )

with col4:
    st.markdown("### Savings Level by Nationality Group")
    sav_order = ["<5k AED", "5k-20k AED", ">20k AED"]
    nat_sav = (
        filtered.groupby(["nationality", "current_savings_level"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=sav_order, fill_value=0)
    )
    nat_sav_pct = (nat_sav.div(nat_sav.sum(axis=1), axis=0) * 100).round(1).reset_index()

    fig5 = go.Figure()
    colors = ["#1B4F72", "#2E86C1", "#1abc9c"]
    for col_name, color in zip(sav_order, colors):
        fig5.add_trace(go.Bar(
            name=col_name,
            x=nat_sav_pct["nationality"],
            y=nat_sav_pct[col_name],
            marker_color=color,
        ))
    fig5.update_layout(
        barmode="stack", template="plotly_dark",
        yaxis_title="% of Group", yaxis_range=[0, 110],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=10), height=280,
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Expat-Western and Emirati groups show the highest proportion of "
        "respondents with savings > AED 20k, suggesting stronger existing "
        "financial discipline and higher investment readiness."
    )

st.divider()

# ── Row 4: Correlation matrix ──────────────────────────────────────────────
st.markdown("### Correlation Matrix — Key Numeric Variables")
num_cols = [
    "age", "monthly_income_aed", "financial_literacy_score",
    "has_existing_investments", "app_sessions_per_week",
    "satisfaction_score", "monthly_investment_aed",
    "will_adopt_microinvestment",
]
corr = filtered[num_cols].corr().round(2)

fig6 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale="RdBu",
    zmid=0, zmin=-1, zmax=1,
    text=corr.values,
    texttemplate="%{text:.2f}",
    textfont={"size": 10},
    showscale=True,
))
fig6.update_layout(
    template="plotly_dark",
    margin=dict(t=10, b=10), height=420,
)
st.plotly_chart(fig6, use_container_width=True)
st.caption(
    "Monthly income and financial literacy show the strongest positive correlations "
    "with monthly investment amount. Satisfaction score is weakly correlated with "
    "all variables, suggesting it captures independent UX quality signals."
)
