"""
generate_dataset.py
===================
Standalone script to regenerate the synthetic UAE fintech survey dataset.
Run this once before launching the Streamlit app if the data/ folder is empty.

Usage
-----
    python generate_dataset.py

Output
------
    data/uae_finapp_dataset.csv   (500 rows × 20 columns)
"""

import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
N    = 500

# ── Demographics ───────────────────────────────────────────────────────────
age = np.random.choice(range(21, 45), N)

gender = np.random.choice(["Male", "Female"], N, p=[0.55, 0.45])

nationality = np.random.choice(
    ["Emirati", "Expat-Arab", "Expat-South-Asian", "Expat-Western", "Other"],
    N, p=[0.15, 0.20, 0.35, 0.20, 0.10],
)

employment = np.random.choice(
    ["Student", "Early-Career", "Mid-Career", "Freelancer"],
    N, p=[0.25, 0.35, 0.28, 0.12],
)

income_map = {
    "Student":      (2_000,  6_000),
    "Early-Career": (6_000, 15_000),
    "Mid-Career":  (15_000, 35_000),
    "Freelancer":   (5_000, 20_000),
}
monthly_income = np.array([np.random.randint(*income_map[e]) for e in employment])

# ── Financial profile ──────────────────────────────────────────────────────
financial_literacy = np.random.choice([1, 2, 3, 4, 5], N, p=[0.08, 0.17, 0.30, 0.28, 0.17])

current_savings = np.random.choice(
    ["None", "<5k AED", "5k-20k AED", ">20k AED"], N, p=[0.22, 0.30, 0.28, 0.20]
)

has_investments = np.random.choice([0, 1], N, p=[0.60, 0.40])
sharia_pref     = np.random.choice([0, 1], N, p=[0.55, 0.45])

# ── Feature usage flags ────────────────────────────────────────────────────
feat_spending  = np.random.choice([0, 1], N, p=[0.25, 0.75])
feat_savings   = np.random.choice([0, 1], N, p=[0.30, 0.70])
feat_auto      = np.random.choice([0, 1], N, p=[0.45, 0.55])
feat_portfolio = np.random.choice([0, 1], N, p=[0.50, 0.50])
feat_sharia    = np.where(
    sharia_pref == 1,
    np.random.choice([0, 1], N, p=[0.20, 0.80]),
    np.random.choice([0, 1], N, p=[0.90, 0.10]),
)
feat_insights  = np.random.choice([0, 1], N, p=[0.35, 0.65])

# ── Behavioural metrics ────────────────────────────────────────────────────
app_sessions = np.clip(np.random.poisson(5, N) + financial_literacy - 2, 1, 20).astype(int)

satisfaction = np.clip(
    3.0
    + 0.3 * (feat_spending + feat_savings + feat_insights)
    - 0.2 * (1 - feat_auto)
    + np.random.normal(0, 0.6, N),
    1.0, 5.0,
).round(1)

# ── Targets ────────────────────────────────────────────────────────────────
monthly_investment = np.clip(
    monthly_income * 0.04
    + financial_literacy * 200
    + feat_auto      * 300
    + feat_portfolio * 250
    + has_investments * 400
    + np.random.normal(0, 200, N),
    50, 5_000,
).astype(int)

prob_adopt = np.clip(
    0.15
    + 0.20 * (financial_literacy >= 4).astype(float)
    + 0.15 * (has_investments == 1).astype(float)
    + 0.10 * (feat_auto == 1).astype(float)
    + 0.05 * (monthly_income > 10_000).astype(float)
    + 0.08 * (feat_portfolio == 1).astype(float)
    + np.random.uniform(0, 0.15, N),
    0.0, 1.0,
)
will_adopt = (np.random.rand(N) < prob_adopt).astype(int)

# ── Assemble ───────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "respondent_id":               range(1, N + 1),
    "age":                         age,
    "gender":                      gender,
    "nationality":                 nationality,
    "employment_status":           employment,
    "monthly_income_aed":          monthly_income,
    "financial_literacy_score":    financial_literacy,
    "current_savings_level":       current_savings,
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

os.makedirs("data", exist_ok=True)
df.to_csv("data/uae_finapp_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows × {len(df.columns)} columns → data/uae_finapp_dataset.csv")
print(f"Adoption rate : {df['will_adopt_microinvestment'].mean():.1%}")
print(f"Avg income    : AED {df['monthly_income_aed'].mean():,.0f}")
