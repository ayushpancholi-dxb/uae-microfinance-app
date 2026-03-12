"""
pages/2_Classification.py  —  Random Forest Classification
===========================================================
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import load_data, run_classification

st.set_page_config(page_title="Classification — UAE Fintech", page_icon="🎯", layout="wide")

st.title("🎯 Classification — Predicting Micro-Investment Adoption")
st.markdown(
    "**Algorithm:** Random Forest Classifier  \n"
    "**Target:** `will_adopt_microinvestment` (1 = will adopt, 0 = will not)"
)
st.divider()

df  = load_data()
clf = run_classification(df)

# ── KPI metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Test Accuracy",       f"{clf['accuracy']}%")
m2.metric("CV Accuracy (5-fold)", f"{clf['cv_accuracy']}%", f"± {clf['cv_std']}%")
m3.metric("Train Records",       f"{clf['n_train']:,}")
m4.metric("Test Records",        f"{clf['n_test']:,}")
m5.metric("Trees in Forest",     "200")
st.divider()

# ── Feature importance + Confusion matrix ─────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Feature Importances")
    imp = clf["importances"]
    fig = px.bar(
        imp.sort_values("Importance"),
        x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#1B4F72", "#2E86C1", "#85C1E9"],
        template="plotly_dark",
        text=imp.sort_values("Importance")["Importance"].map(lambda v: f"{v:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Mean Decrease in Impurity",
        margin=dict(t=10, b=10), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Monthly income and age are the top two predictors. Financial literacy "
        "ranks 4th, confirming that knowledge — not just wealth — drives "
        "micro-investment adoption."
    )

with col2:
    st.markdown("### Confusion Matrix")
    cm   = np.array(clf["cm"])
    labs = clf["report"]

    fig2 = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        colorscale=[[0, "#0f1117"], [1, "#1B4F72"]],
        showscale=False,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="<b>%{text}</b>",
        textfont={"size": 28},
    ))
    fig2.update_layout(
        template="plotly_dark",
        margin=dict(t=10, b=10), height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Derived metrics
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | True Positives  | **{tp}** |
    | True Negatives  | **{tn}** |
    | False Positives | **{fp}** |
    | False Negatives | **{fn}** |
    | Precision (Adopter) | **{precision:.1%}** |
    | Recall (Adopter)    | **{recall:.1%}** |
    | F1-Score (Adopter)  | **{f1:.3f}** |
    """)

st.divider()

# ── Full classification report ─────────────────────────────────────────────
st.markdown("### Full Classification Report")
report = clf["report"]
report_data = []
for cls in ["Non-Adopter", "Adopter", "macro avg", "weighted avg"]:
    if cls in report:
        r = report[cls]
        report_data.append({
            "Class":     cls,
            "Precision": f"{r['precision']:.3f}",
            "Recall":    f"{r['recall']:.3f}",
            "F1-Score":  f"{r['f1-score']:.3f}",
            "Support":   int(r["support"]),
        })

st.dataframe(
    pd.DataFrame(report_data).set_index("Class"),
    use_container_width=True,
)

st.divider()

# ── Business interpretation ────────────────────────────────────────────────
st.markdown("### 💡 Business Interpretation")
col_a, col_b = st.columns(2)
with col_a:
    st.info("""
**Top Predictors → Targeting Strategy**

- **Monthly Income** (rank 1): Target users earning > AED 10,000/month
- **Age** (rank 2): 28–38 year-olds are the prime segment
- **App Sessions/wk** (rank 3): Early engagement is a strong leading indicator — invest in onboarding
- **Financial Literacy** (rank 4): In-app education content lifts adoption probability
    """)
with col_b:
    st.success("""
**Model Use in Production**

A 66.4% accuracy model is useful for **ranked prospect lists**, not hard cutoffs:
- Score all new users at signup using the 13 features
- Route top-quartile scores into "Investment Journey" onboarding
- Use recall optimisation (lower threshold) to reduce missed adopters
- Retrain monthly as real behavioural data accumulates
    """)
