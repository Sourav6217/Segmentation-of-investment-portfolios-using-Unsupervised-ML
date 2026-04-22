# 📈 Portfolio Intelligence Suite

**S&P 500 Cluster-Based Segmentation & Advisory System**  
WeSchool Mumbai · PGDM-RBA 2024–26 · Group 5 · Hackathon Project

---

## 🚀 Live App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segmentation-of-investment-portfolios-using-unsupervised-ml.streamlit.app/)

---

## 📌 Project Overview

This app applies **KMeans clustering (k=5)** to 499 S&P 500 stocks, segmenting them into distinct financial profiles and providing a personalised stock advisory system based on user risk appetite.

### Cluster Profiles

| Cluster | Label | Description |
|---------|-------|-------------|
| 0 | ⚖️ Balanced Core | Moderate growth, diversified risk |
| 1 | 🚀 High Growth | High revenue expansion, smaller-cap |
| 2 | 🏛️ Mega-Cap Stable | Large-cap stabilisers, capital preservation |
| 3 | 🛡️ Defensive Income | Utilities, healthcare, real estate |
| 4 | 🔬 Emerging Niche | Niche or emerging-sector firms |

**Silhouette Score: ~0.47** — Moderate to good cluster separation

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive UI
- **Plotly** — Interactive charts
- **scikit-learn** — KMeans, StandardScaler, RandomForest
- **pandas / numpy** — Data processing

---

## 📂 Repository Structure

```
├── app.py                  # Main Streamlit application
├── Hackathon.csv           # S&P 500 dataset (499 stocks)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — live in ~2 minutes!

---

## 🔬 Methodology

### Data Preprocessing
- Median imputation for `Ebitda`, `Revenuegrowth`, `Fulltimeemployees`
- Mode imputation for `State`

### Feature Engineering
- Log₁₀ transforms for `Marketcap` and `Ebitda` (right-skew correction)
- Cap Tier ordinal encoding (Small / Mid / Large / Mega)
- Domain-driven risk group one-hot encoding (High Growth / Moderate / Defensive)
- Quality Score: EBITDA > 0 AND Revenue Growth > 0

### Model Selection
- **KMeans (k=5):** WCSS elbow at k=5; silhouette score ~0.47
- **DBSCAN rejected:** No noise structure in financial data
- **Hierarchical rejected:** No clear dendrogram cut-point

### Advisory Logic
- Low Risk → Cluster 2 (Mega-Cap Stable)
- Medium Risk → Cluster 0 (Balanced Core)
- High Risk → Cluster 1 (High Growth)

---

## 👥 Team — Group 5

| Roll No | Name |
|---------|------|
| RBA35 | Vaishnavi Dube |
| RBA46 | Parth Murdeshwar |
| RBA51 | Palak Sahu |
| RBA54 | Kaustubh Pawar |
| RBA64 | Ajinkya Sarwankar |
| RBA70 | Sourav Manna |

---

*WeSchool Mumbai · PGDM-RBA · Batch 2024–26*
