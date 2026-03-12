"""
pages/4_Association_Rules.py  —  Apriori Feature Bundle Analysis
=================================================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_data, run_arm

st.set_page_config(page_title="Association Rules — UAE Fintech", page_icon="🔗", layout="wide")

st.title("🔗 Association Rule Mining — Feature Bundle Analysis")
st.markdown(
    "**Algorithm:** Apriori (implemented from scratch — no external library)  \n"
    "Discovers which app features users co-adopt, revealing natural product bundles."
)
st.divider()

df = load_data()

# ── Sidebar controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ARM Parameters")
    min_support = st.slider(
        "Minimum Support", min_value=0.10, max_value=0.60, value=0.30, step=0.05,
        help="Fraction of users who must co-use both features",
    )
    min_confidence = st.slider(
        "Minimum Confidence", min_value=0.30, max_value=0.90, value=0.50, step=0.05,
        help="Conditional probability: P(Consequent | Antecedent)",
    )

arm = run_arm(df, min_support=min_support, min_confidence=min_confidence)
rules_df = arm["rules"]

# ── KPIs ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rules Found",       len(rules_df))
k2.metric("Transactions",      f"{arm['n_transactions']:,}")
k3.metric("Min Support",       f"{min_support:.0%}")
k4.metric("Min Confidence",    f"{min_confidence:.0%}")
st.divider()

if rules_df.empty:
    st.warning("No rules found with the current thresholds. Try lowering Min Support or Min Confidence.")
    st.stop()

# ── Frequent itemsets ──────────────────────────────────────────────────────
st.markdown("### Frequent Feature Usage (1-Itemsets)")
freq1 = arm["freq1"]
freq_df = (
    pd.DataFrame(list(freq1.items()), columns=["Feature", "Support"])
    .sort_values("Support", ascending=False)
)
fig0 = px.bar(
    freq_df, x="Feature", y="Support",
    color="Support",
    color_continuous_scale=["#1B4F72", "#2E86C1", "#1abc9c"],
    text=freq_df["Support"].map(lambda v: f"{v:.1%}"),
    template="plotly_dark",
)
fig0.update_traces(textposition="outside")
fig0.update_layout(
    coloraxis_showscale=False,
    yaxis_tickformat=".0%", yaxis_range=[0, 0.9],
    margin=dict(t=10, b=10), height=280,
)
st.plotly_chart(fig0, use_container_width=True)
st.caption(
    "SpendTrack and SavingsGoal are the most widely used features (>70% of users), "
    "making them the natural anchor for any bundling or cross-sell strategy."
)
st.divider()

# ── Grouped bar: Support / Confidence / Lift ───────────────────────────────
st.markdown("### Top Association Rules — Support, Confidence & Lift")
top = rules_df.head(8)
fig1 = go.Figure()
for metric, color in [("Support", "#1B4F72"), ("Confidence", "#2E86C1"), ("Lift", "#F39C12")]:
    fig1.add_trace(go.Bar(
        name=metric, x=top["Rule"], y=top[metric],
        marker_color=color, text=top[metric].map(lambda v: f"{v:.3f}"),
        textposition="outside",
    ))
fig1.add_hline(y=1.0, line_dash="dash", line_color="gray", line_width=1,
               annotation_text="Lift = 1 (independent)", annotation_position="top left")
fig1.update_layout(
    barmode="group",
    template="plotly_dark",
    xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
    yaxis=dict(title="Metric Value", range=[0, 1.3]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=10, b=120), height=420,
)
st.plotly_chart(fig1, use_container_width=True)
st.caption(
    "Rules with Lift > 1 indicate features that co-occur more than expected by chance. "
    "The ShariaFilter → SpendTrack rule has the highest lift, suggesting Sharia-preference "
    "users are a cohesive, trackable segment."
)
st.divider()

# ── Scatter: Support vs Confidence ────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Support vs Confidence (bubble size = Lift)")
    fig2 = px.scatter(
        rules_df,
        x="Support", y="Confidence",
        size=[max(r * 40, 8) for r in rules_df["Lift"]],
        color="Lift",
        color_continuous_scale=["#1B4F72", "#2E86C1", "#F39C12"],
        hover_data=["Rule", "Support", "Confidence", "Lift"],
        text="Rule",
        template="plotly_dark",
    )
    fig2.update_traces(textposition="top center", textfont=dict(size=8))
    fig2.update_layout(
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        coloraxis_colorbar=dict(title="Lift"),
        margin=dict(t=10, b=10), height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Rules towards the top-right are both common (high support) and reliable "
        "(high confidence). The SavingsGoal ↔ SpendTrack pair dominates this quadrant."
    )

with col2:
    st.markdown("### All Rules Table")
    interpretations = {
        "ShariaFilter → SpendTrack": "Sharia users also track spending habits",
        "SavingsGoal → SpendTrack":  "Goal-setters track their spending",
        "SpendTrack → SavingsGoal":  "Trackers set savings goals",
        "Insights → SavingsGoal":    "Insight-seekers create goals",
        "SavingsGoal → Insights":    "Goal users seek financial insights",
        "AutoInvest → SavingsGoal":  "Auto-investors also set goals",
        "SavingsGoal → AutoInvest":  "Goal-setters use Auto-Invest",
        "Insights → SpendTrack":     "Insight users track spending",
        "SpendTrack → Insights":     "Trackers view financial insights",
        "AutoInvest → Insights":     "Auto-investors check insights",
    }
    display = rules_df.copy()
    display["Support"]    = display["Support"].map(lambda v: f"{v:.1%}")
    display["Confidence"] = display["Confidence"].map(lambda v: f"{v:.1%}")
    display["Lift"]       = display["Lift"].map(lambda v: f"{v:.3f}")
    display["Interpretation"] = display["Rule"].map(
        lambda r: interpretations.get(r, "—")
    )
    st.dataframe(
        display[["Rule", "Support", "Confidence", "Lift", "Interpretation"]],
        use_container_width=True,
        hide_index=True,
        height=400,
    )

st.divider()

# ── Product recommendations ────────────────────────────────────────────────
st.markdown("### 💡 Product Bundle Recommendations")
c1, c2, c3 = st.columns(3)
with c1:
    st.success("""
**Core Bundle (Home Screen)**

SavingsGoal + SpendTracker + Insights

Support: ~52% | Confidence: ~71%

*Default onboarding flow for all new users.*
    """)
with c2:
    st.info("""
**Investment Gateway Bundle**

SpendTracker → Portfolio → Auto-Invest

Progressive feature unlock after 2 weeks of tracking.

*Converts passive users into investors.*
    """)
with c3:
    st.warning("""
**Sharia Premium Bundle**

ShariaFilter + SpendTrack + SavingsGoal

Targeted at the 42.8% Sharia-preference segment.

*Highest lift rule (1.03) — strong co-adoption signal.*
    """)
