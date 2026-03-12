"""
data_loader.py
==============
Generates the synthetic UAE fintech dataset in memory and runs all four
ML pipelines. No file I/O — works on Streamlit Cloud without any CSV.
All functions cached with @st.cache_data so they run once per session.
"""

import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42

FEATURE_COLS = [
    "uses_spending_tracker",
    "uses_savings_goals",
    "uses_auto_invest",
    "uses_portfolio_view",
    "uses_sharia_filter",
    "uses_financial_insights",
]
FEATURE_LABELS = [
    "Spend Tracker",
    "Savings Goals",
    "Auto Invest",
    "Portfolio",
    "Sharia Filter",
    "Insights",
]

PERSONA_COLORS = {
    "Cautious Savers":       "#1B4F72",
    "Aspiring Investors":    "#F39C12",
    "Passive Professionals": "#27AE60",
    "Engaged High-Earners":  "#E74C3C",
}


# ── Dataset (always generated in memory — no CSV required) ─────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Generate the 500-row synthetic survey dataset entirely in memory."""
    rng = np.random.default_rng(SEED)
    N   = 500

    employment = rng.choice(
        ["Student", "Early-Career", "Mid-Career", "Freelancer"],
        N, p=[0.25, 0.35, 0.28, 0.12],
    )
    income_ranges = {
        "Student":      (2_000,  6_000),
        "Early-Career": (6_000, 15_000),
        "Mid-Career":  (15_000, 35_000),
        "Freelancer":   (5_000, 20_000),
    }
    monthly_income = np.array(
        [rng.integers(income_ranges[e][0], income_ranges[e][1]) for e in employment]
    )

    financial_literacy = rng.choice([1, 2, 3, 4, 5], N, p=[0.08, 0.17, 0.30, 0.28, 0.17])
    has_investments    = rng.choice([0, 1], N, p=[0.60, 0.40])
    sharia_pref        = rng.choice([0, 1], N, p=[0.55, 0.45])

    feat_spending  = rng.choice([0, 1], N, p=[0.25, 0.75])
    feat_savings   = rng.choice([0, 1], N, p=[0.30, 0.70])
    feat_auto      = rng.choice([0, 1], N, p=[0.45, 0.55])
    feat_portfolio = rng.choice([0, 1], N, p=[0.50, 0.50])
    feat_sharia    = np.where(
        sharia_pref == 1,
        rng.choice([0, 1], N, p=[0.20, 0.80]),
        rng.choice([0, 1], N, p=[0.90, 0.10]),
    )
    feat_insights = rng.choice([0, 1], N, p=[0.35, 0.65])

    app_sessions = np.clip(
        rng.poisson(5, N) + financial_literacy - 2, 1, 20
    ).astype(int)

    satisfaction = np.clip(
        3.0
        + 0.3 * (feat_spending + feat_savings + feat_insights)
        - 0.2 * (1 - feat_auto)
        + rng.normal(0, 0.6, N),
        1.0, 5.0,
    ).round(1)

    monthly_investment = np.clip(
        monthly_income * 0.04
        + financial_literacy * 200
        + feat_auto      * 300
        + feat_portfolio * 250
        + has_investments * 400
        + rng.normal(0, 200, N),
        50, 5_000,
    ).astype(int)

    prob_adopt = np.clip(
        0.15
        + 0.20 * (financial_literacy >= 4)
        + 0.15 * (has_investments == 1)
        + 0.10 * (feat_auto == 1)
        + 0.05 * (monthly_income > 10_000)
        + 0.08 * (feat_portfolio == 1)
        + rng.uniform(0, 0.15, N),
        0.0, 1.0,
    )
    will_adopt = (rng.random(N) < prob_adopt).astype(int)

    return pd.DataFrame({
        "respondent_id":               range(1, N + 1),
        "age":                         rng.integers(21, 45, N),
        "gender":                      rng.choice(["Male", "Female"], N, p=[0.55, 0.45]),
        "nationality":                 rng.choice(
            ["Emirati", "Expat-Arab", "Expat-South-Asian", "Expat-Western", "Other"],
            N, p=[0.15, 0.20, 0.35, 0.20, 0.10],
        ),
        "employment_status":           employment,
        "monthly_income_aed":          monthly_income,
        "financial_literacy_score":    financial_literacy,
        "current_savings_level":       rng.choice(
            ["None", "<5k AED", "5k-20k AED", ">20k AED"],
            N, p=[0.22, 0.30, 0.28, 0.20],
        ),
        "has_existing_investments":    has_investments,
        "sharia_compliant_preference": sharia_pref,
        "uses_spending_tracker":       feat_spending,
        "uses_savings_goals":          feat_savings,
        "uses_auto_invest":            feat_auto,
        "uses_portfolio_view":         feat_portfolio,
        "uses_sharia_filter":          feat_sharia,
        "uses_financial_insights":     feat_insights,
        "app_sessions_per_week":       app_sessions,
        "satisfaction_score":          satisfaction,
        "monthly_investment_aed":      monthly_investment,
        "will_adopt_microinvestment":  will_adopt,
    })


# ── Classification ─────────────────────────────────────────────────────────
@st.cache_data
def run_classification(df: pd.DataFrame) -> dict:
    le  = LabelEncoder()
    df2 = df.copy()
    df2["employment_enc"] = le.fit_transform(df2["employment_status"])
    df2["savings_enc"]    = le.fit_transform(df2["current_savings_level"])

    features = [
        "age", "monthly_income_aed", "financial_literacy_score",
        "has_existing_investments", "sharia_compliant_preference",
        "uses_spending_tracker", "uses_savings_goals", "uses_auto_invest",
        "uses_portfolio_view", "uses_financial_insights",
        "app_sessions_per_week", "employment_enc", "savings_enc",
    ]
    display_names = [
        "Age", "Monthly Income", "Fin. Literacy", "Has Investments",
        "Sharia Pref", "Spend Tracker", "Savings Goals", "Auto-Invest",
        "Portfolio", "Insights", "Sessions/wk", "Employment", "Savings Level",
    ]

    X = df2[features]
    y = df2["will_adopt_microinvestment"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    imp_df = (
        pd.DataFrame({"Feature": display_names, "Importance": clf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "accuracy":    round((y_pred == y_te).mean() * 100, 1),
        "cv_accuracy": round(cv_acc.mean() * 100, 1),
        "cv_std":      round(cv_acc.std() * 100, 1),
        "cm":          confusion_matrix(y_te, y_pred).tolist(),
        "report":      classification_report(
                           y_te, y_pred,
                           target_names=["Non-Adopter", "Adopter"],
                           output_dict=True,
                       ),
        "importances": imp_df,
        "n_train":     len(X_tr),
        "n_test":      len(X_te),
    }


# ── Clustering ─────────────────────────────────────────────────────────────
@st.cache_data
def run_clustering(df: pd.DataFrame) -> dict:
    clust_features = [
        "age", "monthly_income_aed", "financial_literacy_score",
        "app_sessions_per_week", "monthly_investment_aed", "satisfaction_score",
    ]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[clust_features])

    inertias = []
    k_range  = list(range(2, 10))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(round(km.inertia_, 1))

    km_final = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    df2 = df.copy()
    df2["cluster"] = km_final.fit_predict(X_scaled)

    inc_order = (
        df2.groupby("cluster")["monthly_income_aed"]
        .mean().sort_values().index.tolist()
    )
    name_map = dict(zip(
        inc_order,
        ["Cautious Savers", "Aspiring Investors", "Passive Professionals", "Engaged High-Earners"],
    ))
    df2["persona"] = df2["cluster"].map(name_map)

    profiles = (
        df2.groupby("persona")
        .agg(
            n=("respondent_id", "count"),
            avg_age=("age", "mean"),
            avg_income=("monthly_income_aed", "mean"),
            avg_literacy=("financial_literacy_score", "mean"),
            avg_sessions=("app_sessions_per_week", "mean"),
            avg_investment=("monthly_investment_aed", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            adoption_rate=("will_adopt_microinvestment", "mean"),
        )
        .round(2)
        .reset_index()
    )

    radar_cols = ["avg_age", "avg_income", "avg_literacy",
                  "avg_sessions", "avg_investment", "adoption_rate"]
    radar_norm = profiles.set_index("persona")[radar_cols].copy()
    for col in radar_cols:
        col_range = radar_norm[col].max() - radar_norm[col].min()
        radar_norm[col] = (
            (radar_norm[col] - radar_norm[col].min()) / (col_range if col_range > 0 else 1)
        )
    radar_norm = radar_norm.round(3)

    return {
        "df":         df2,
        "elbow_k":    k_range,
        "inertias":   inertias,
        "profiles":   profiles,
        "radar_norm": radar_norm,
        "radar_cols": ["Age", "Income", "Literacy", "Sessions", "Investment", "Adoption"],
    }


# ── Association Rule Mining ────────────────────────────────────────────────
@st.cache_data
def run_arm(df: pd.DataFrame, min_support: float = 0.30, min_confidence: float = 0.50) -> dict:
    item_names = {
        "uses_spending_tracker":   "SpendTrack",
        "uses_savings_goals":      "SavingsGoal",
        "uses_auto_invest":        "AutoInvest",
        "uses_portfolio_view":     "Portfolio",
        "uses_sharia_filter":      "ShariaFilter",
        "uses_financial_insights": "Insights",
    }

    transactions = []
    for _, row in df[list(item_names.keys())].iterrows():
        basket = frozenset(item_names[c] for c in item_names if row[c] == 1)
        if basket:
            transactions.append(basket)

    def support(itemset):
        return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)

    all_items = sorted({i for t in transactions for i in t})
    freq1 = {frozenset([i]): support(frozenset([i])) for i in all_items}
    freq1 = {k: v for k, v in freq1.items() if v >= min_support}

    freq2 = {}
    for a, b in combinations(freq1.keys(), 2):
        pair = a | b
        s    = support(pair)
        if s >= min_support:
            freq2[pair] = s

    rules = []
    for itemset, sup in freq2.items():
        items = sorted(itemset)
        for i in range(len(items)):
            ant  = frozenset([items[i]])
            cons = frozenset([items[1 - i]])
            conf = sup / support(ant)
            lift = conf / support(cons)
            if conf >= min_confidence:
                rules.append({
                    "Antecedent": list(ant)[0],
                    "Consequent": list(cons)[0],
                    "Rule":       f"{list(ant)[0]} \u2192 {list(cons)[0]}",
                    "Support":    round(sup,  4),
                    "Confidence": round(conf, 4),
                    "Lift":       round(lift, 4),
                })

    rules_df = (
        pd.DataFrame(rules).drop_duplicates()
        .sort_values("Lift", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "rules":          rules_df,
        "n_transactions": len(transactions),
        "freq1":          {list(k)[0]: round(v, 4) for k, v in freq1.items()},
    }


# ── Regression ─────────────────────────────────────────────────────────────
@st.cache_data
def run_regression(df: pd.DataFrame) -> dict:
    le  = LabelEncoder()
    df2 = df.copy()
    df2["employment_enc"] = le.fit_transform(df2["employment_status"])

    features = [
        "monthly_income_aed", "financial_literacy_score",
        "has_existing_investments", "uses_auto_invest", "uses_portfolio_view",
        "uses_savings_goals", "uses_financial_insights",
        "app_sessions_per_week", "satisfaction_score", "employment_enc",
    ]
    display_names = [
        "Monthly Income", "Fin. Literacy", "Has Investments",
        "Auto-Invest", "Portfolio View", "Savings Goals", "Insights",
        "Sessions/wk", "Satisfaction", "Employment",
    ]

    X = df2[features]
    y = df2["monthly_investment_aed"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)

    cv_r2 = cross_val_score(lr, X, y, cv=5, scoring="r2")

    coef_df = (
        pd.DataFrame({"Feature": display_names, "Coefficient": lr.coef_})
        .sort_values("Coefficient")
        .reset_index(drop=True)
    )

    return {
        "r2":           round(r2_score(y_te, y_pred), 4),
        "mae":          round(mean_absolute_error(y_te, y_pred), 1),
        "cv_r2":        round(cv_r2.mean(), 4),
        "cv_r2_std":    round(cv_r2.std(), 4),
        "intercept":    round(lr.intercept_, 2),
        "coefficients": coef_df,
        "y_test":       y_te.values,
        "y_pred":       y_pred,
        "residuals":    y_te.values - y_pred,
        "n_train":      len(X_tr),
        "n_test":       len(X_te),
    }
