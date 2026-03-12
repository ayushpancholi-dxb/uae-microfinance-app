# UAE Personal Finance & Micro-Investment App
### MBA Data Analytics — Individual PBL | Dr. Anshul Gupta

An interactive Streamlit dashboard that validates the business case for a
UAE-focused personal finance and micro-investment mobile app using four
machine learning algorithms on a 500-respondent synthetic survey dataset.

---

## 🚀 Deploy on Streamlit Cloud (GitHub → Live App)

### Step 1 — Push this repo to GitHub
```
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo
4. Set **Main file path** to: `app.py`
5. Click **Deploy**

That's it — Streamlit Cloud installs all packages from `requirements.txt` automatically.

---

## 💻 Run Locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the dataset
python generate_dataset.py

# 4. Launch
streamlit run app.py
```

---

## 📁 Repository Structure

```
.
├── app.py                          # Main page — Overview & KPIs
├── generate_dataset.py             # Standalone dataset generator
├── requirements.txt
│
├── pages/
│   ├── 1_EDA.py                    # Exploratory Data Analysis (5 charts)
│   ├── 2_Classification.py         # Random Forest Classifier
│   ├── 3_Clustering.py             # K-Means Segmentation (k=4)
│   ├── 4_Association_Rules.py      # Apriori Feature Bundle Mining
│   └── 5_Regression.py             # Linear Regression Forecasting
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py              # All ML pipelines + @st.cache_data
│
├── data/
│   └── uae_finapp_dataset.csv      # 500 rows × 20 columns
│
└── .streamlit/
    └── config.toml                 # Dark theme + server settings
```

---

## 📊 Dashboard Pages

| Page | What it shows |
|------|--------------|
| **🏠 Overview** | KPI cards, algorithm summary, adoption by literacy, segment donut |
| **🔍 EDA** | Adoption by employment/literacy, income histograms, feature heatmap, nationality breakdown, correlation matrix |
| **🎯 Classification** | Feature importances, confusion matrix, classification report, targeting recommendations |
| **🗂 Clustering** | Elbow method, scatter by persona, radar chart, cluster profiles table, marketing strategies |
| **🔗 Association Rules** | Frequent itemsets, grouped bar, support-confidence bubble, rules table, product bundles |
| **📈 Regression** | Actual vs predicted, coefficients, residual analysis, business levers |

---

## 🤖 Algorithms

| Algorithm | Library | Key Result |
|-----------|---------|------------|
| Random Forest Classifier | scikit-learn | 66.4% accuracy, ROC-AUC 0.69 |
| K-Means Clustering | scikit-learn | 4 personas via Elbow Method |
| Apriori (from scratch) | Pure Python | 10 rules, top lift 1.032 |
| Linear Regression | scikit-learn | R² = 0.884, MAE = AED 159 |

---

## 📦 Dependencies

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
plotly==5.22.0
matplotlib==3.9.0
seaborn==0.13.2
```

---

## 📋 Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | 21–44 |
| `gender` | str | Male / Female |
| `nationality` | str | Emirati, Expat-Arab, Expat-South-Asian, Expat-Western, Other |
| `employment_status` | str | Student, Early-Career, Mid-Career, Freelancer |
| `monthly_income_aed` | int | Stratified by employment (AED) |
| `financial_literacy_score` | int | 1 (low) – 5 (high) |
| `current_savings_level` | str | None / <5k / 5k-20k / >20k AED |
| `has_existing_investments` | int | 0 / 1 |
| `sharia_compliant_preference` | int | 0 / 1 |
| `uses_spending_tracker` | int | 0 / 1 |
| `uses_savings_goals` | int | 0 / 1 |
| `uses_auto_invest` | int | 0 / 1 |
| `uses_portfolio_view` | int | 0 / 1 |
| `uses_sharia_filter` | int | 0 / 1 |
| `uses_financial_insights` | int | 0 / 1 |
| `app_sessions_per_week` | int | 1–20 |
| `satisfaction_score` | float | 1.0–5.0 |
| `monthly_investment_aed` | int | **Regression target** |
| `will_adopt_microinvestment` | int | **Classification target** (0/1) |

---

*MBA Data Analytics — Individual PBL | Dr. Anshul Gupta*
