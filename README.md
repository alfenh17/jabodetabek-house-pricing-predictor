# 🏠 Jakarta House Price Predictor
### Machine Learning Portfolio Project — 2026

> **ML-powered property valuation for Greater Jakarta (Jabodetabek)** using XGBoost / LightGBM, SHAP explainability, MLflow experiment tracking, and a Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3x-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-2.x-blue)
![SHAP](https://img.shields.io/badge/SHAP-0.4x-purple)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Demo Screenshot](#-demo-screenshot)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Quick Start](#-quick-start)
6. [Notebook Walkthrough](#-notebook-walkthrough)
7. [Feature Engineering](#-feature-engineering)
8. [Model Performance](#-model-performance)
9. [SHAP Explainability](#-shap-explainability)
10. [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
11. [Deployment](#-deployment)
12. [Roadmap](#-roadmap)

---

## 🎯 Project Overview

This project builds an end-to-end machine learning pipeline to **predict residential property prices across the Jabodetabek area** (Jakarta, Bogor, Depok, Tangerang, Bekasi).

### Key Highlights

| Feature | Details |
|---------|---------|
| **Data Sources** | Kaggle Indonesia House Price dataset + supplementary scraping from Rumah123 / 99.co |
| **Geospatial Feature** | Distance to nearest Jakarta MRT station via Geopy |
| **Models** | Ridge (baseline), XGBoost, LightGBM — tuned with Optuna (TPE, 50 trials each) |
| **Explainability** | SHAP waterfall, beeswarm, dependence, and feature importance plots |
| **Tracking** | MLflow experiment tracking with parameter + metric logging |
| **Deployment** | Interactive Streamlit web app with live SHAP explanations |

### Business Value

- **For home buyers**: Get an independent market valuation estimate before negotiating
- **For real estate agents**: Justify pricing with data-driven, explainable insights
- **For portfolio showcase**: Demonstrates full ML lifecycle: data → feature engineering → model → explainability → deployment

---

## 📸 Demo Screenshot

```
┌─────────────────────────────────────────────────────────────────────┐
│  🏠 Jakarta House Price Predictor                                   │
│  ML-powered valuation · XGBoost · SHAP                             │
├─────────────────────────────────────────────────────────────────────┤
│  [Building: 120m²] [Land: 150m²] [MRT: 2.1km] [Age: 10yr]         │
│                                                                     │
│  ┌─────────────────────────────────────────┐                        │
│  │   Estimated Market Value                │                        │
│  │   IDR 4.85 Miliar                       │ 🔍 Key Price Drivers   │
│  │   Range: IDR 4.12B – IDR 5.57B         │ ↑ luas_bangunan_m2    │
│  └─────────────────────────────────────────┘ ↑ city (Jak Sel)     │
│                                               ↓ dist_mrt_km        │
│  [Waterfall Chart] [Feature Importance] [Sensitivity Analysis]     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

```
Data Layer          : Pandas · NumPy · BeautifulSoup · Requests · Geopy
Visualization       : Matplotlib · Seaborn · Plotly · Folium
Machine Learning    : Scikit-learn · XGBoost · LightGBM
Hyperparameter Opt  : Optuna (TPE Sampler)
Explainability      : SHAP (TreeExplainer)
Experiment Tracking : MLflow
Web Application     : Streamlit
Serialization       : Joblib
```

---

## 📂 Project Structure

```
jakarta-house-price/
│
├── notebooks/
│   └── jakarta_house_price_predictor.ipynb   # Full ML pipeline (10 sections)
│
├── data/
│   ├── raw/
│   │   ├── indonesia_house_price.csv          # Kaggle dataset
│   │   └── scrape_rumah123.csv                # Web-scraped listings
│   └── processed/
│       ├── jabodetabek_houses.csv             # Cleaned & feature-engineered
│       └── price_heatmap.html                 # Interactive Folium map
│
├── models/
│   ├── best_model.pkl                         # Best trained model (XGB or LGB)
│   ├── preprocessing_pipeline.pkl             # Sklearn ColumnTransformer
│   ├── shap_explainer.pkl                     # SHAP TreeExplainer
│   └── feature_metadata.json                  # Feature lists, encodings, city coords
│
├── mlruns/                                    # MLflow tracking data (auto-generated)
│
├── app.py                                     # Streamlit web application
├── requirements.txt                           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/jakarta-house-price.git
cd jakarta-house-price

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download Data

**Option A — Kaggle API:**
```bash
# Place kaggle.json at ~/.kaggle/kaggle.json first
kaggle datasets download -d nafisbarizki/daftar-harga-rumah-jabodetabek \
    -p data/raw/ --unzip
```

**Option B — Manual:**
Download from [Kaggle](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek) and place the CSV in `data/raw/indonesia_house_price.csv`.

> **No Kaggle account?** The notebook automatically generates a synthetic dataset so you can run everything end-to-end.

### 3. Run the Notebook

```bash
jupyter notebook notebooks/jakarta_house_price_predictor.ipynb
```

Run all cells in order (Kernel → Restart & Run All). This will:
- Load and preprocess data
- Engineer features (including MRT distances)
- Train and tune models
- Log experiments to MLflow
- Generate SHAP explanations
- Save all artifacts to `models/`

**Expected runtime: ~15–30 minutes** (depending on Optuna trials and dataset size)

### 4. Launch the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 5. (Optional) View MLflow UI

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

Open `http://localhost:5000` to compare all experiment runs.

---

## 📓 Notebook Walkthrough

The notebook is organized into **10 numbered sections**:

| Section | Topic | Key Outputs |
|---------|-------|-------------|
| **1** | Environment Setup | Libraries, config, MRT coordinates |
| **2** | Data Acquisition | Kaggle load + Rumah123 scraper + merge/dedup |
| **3** | EDA | Distribution plots, heatmap, correlation matrix |
| **4** | Preprocessing | Missing value treatment, outlier capping, log transform |
| **5** | Feature Engineering | MRT distance, room ratios, age buckets, zone encoding |
| **6** | Model Training | Ridge baseline → XGBoost → LightGBM → Optuna tuning |
| **7** | MLflow Tracking | Log params, metrics, model artifacts for all 5 runs |
| **8** | Evaluation | Test MAE/RMSE/MAPE/R², residual analysis |
| **9** | SHAP | Summary, beeswarm, waterfall, dependence plots |
| **10** | Serialization | Save model, pipeline, SHAP explainer, metadata |

---

## ⚙️ Feature Engineering

### Input Features (15 numerical + 1 categorical + zone OHE)

| Feature | Description | Engineering |
|---------|-------------|-------------|
| `luas_bangunan_m2` | Building area (m²) | Raw |
| `luas_tanah_m2` | Land area (m²) | Raw |
| `jumlah_kamar_tidur` | Bedrooms | Raw |
| `jumlah_kamar_mandi` | Bathrooms | Raw |
| `garasi` | Garage capacity | Raw |
| `property_age` | Age in years | `2025 - tahun_dibangun` |
| `dist_mrt_km` | Distance to nearest MRT (km) | Geodesic via Geopy |
| `bath_bed_ratio` | Bathroom-to-bedroom ratio | `km / kt` |
| `area_per_bed` | Building area per bedroom | `luas_b / kt` |
| `bcr` | Building coverage ratio | `luas_b / luas_t` |
| `total_rooms` | Total room count | `kt + km` |
| `kondisi_enc` | Condition (ordinal 0–3) | Manual ordinal map |
| `cert_enc` | Certificate strength (0–2) | SHM=2, HGB=1, Girik=0 |
| `latitude` | Property latitude | Raw / city centroid |
| `longitude` | Property longitude | Raw / city centroid |
| `city` | City/area | OrdinalEncoder |
| `zone_*` | Regional zone | One-hot (4 zones) |

### MRT Distance — Key Innovation

```python
from geopy.distance import geodesic

def min_mrt_distance_km(lat, lon):
    return min(geodesic((lat, lon), station).km 
               for station in MRT_COORDS)
```

Properties within 1 km of an MRT station command a measurable price premium — this is a novel feature not present in the raw dataset.

---

## 📊 Model Performance

Results on held-out **test set** (15% of data):

| Model | MAE (IDR) | RMSE (IDR) | MAPE (%) | R² |
|-------|-----------|------------|----------|-----|
| Ridge (Baseline) | ~850M | ~1.3B | ~28% | ~0.72 |
| XGBoost (default) | ~620M | ~950M | ~18% | ~0.84 |
| LightGBM (default) | ~600M | ~920M | ~17% | ~0.85 |
| XGBoost (Optuna) | ~480M | ~760M | ~14% | ~0.90 |
| **LightGBM (Optuna)** | **~460M** | **~730M** | **~13%** | **~0.91** |

> Actual metrics will vary depending on the dataset used. Synthetic demo data produces approximate results.

### Top SHAP Features (typical ranking)

1. `luas_bangunan_m2` — Building size dominates pricing
2. `luas_tanah_m2` — Land area premium
3. `dist_mrt_km` — MRT proximity (urban price gradient)
4. `latitude` / `longitude` — Geographic location
5. `city` — Area-level premium
6. `property_age` — Depreciation effect
7. `kondisi_enc` — Condition score
8. `cert_enc` — Certificate quality (SHM premium)
9. `total_rooms` — Overall size indicator
10. `bath_bed_ratio` — Luxury ratio

---

## 🔍 SHAP Explainability

Every prediction comes with a full SHAP breakdown:

```
Base value (average log-price)  →  22.41
+ luas_bangunan_m2              →  +0.82  (large building)
+ city: Jakarta Selatan         →  +0.45  (premium area)
- dist_mrt_km: 3.2km            →  -0.18  (far from MRT)
+ cert_enc: SHM                 →  +0.12  (strongest cert)
...
= Predicted log-price           →  23.59  →  IDR 4.85 Miliar
```

The Streamlit app renders:
- **Waterfall plot** — contribution of each feature for this specific property
- **Feature importance bar** — mean |SHAP| across all test predictions
- **Sensitivity analysis** — how price changes as you vary one feature

---

## 📈 MLflow Experiment Tracking

All 5 runs are automatically logged:

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

Logged per run:
- **Parameters**: all hyperparameters (learning_rate, n_estimators, max_depth, ...)
- **Metrics**: val_mae, val_rmse, val_r2, val_mape, val_rmse_log
- **Model artifact**: serialized model (XGBoost or LightGBM native format)
- **Tags**: model_type, tuned, project, author

---

## 🌐 Deployment

### Local

```bash
streamlit run app.py
```

### Streamlit Cloud (free hosting)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file to `app.py`
4. Upload model artifacts via `st.secrets` or Git LFS

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t jakarta-house-price .
docker run -p 8501:8501 jakarta-house-price
```

---

## 🗺️ Roadmap

- [ ] **v1.1** — Add LRT / KRL Commuter Line station proximity features
- [ ] **v1.2** — Integrate Google Maps API for real-time POI (schools, hospitals, malls)
- [ ] **v1.3** — Price trend time series using 99.co historical data
- [ ] **v2.0** — Neural network ensemble (TabNet / FT-Transformer)
- [ ] **v2.1** — Automated monthly data refresh pipeline (Airflow / Prefect)
- [ ] **v2.2** — REST API endpoint (FastAPI) for B2B integration

---

## 📚 References

- [Kaggle Dataset — Daftar Harga Rumah Jabodetabek](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna TPE Sampler](https://optuna.readthedocs.io/)
- [Jakarta MRT Official](https://jakartamrt.co.id/)

---

## 📄 License

MIT License — feel free to use, adapt, and build on this for your own portfolio.

---

*Built as Portfolio Project 2 — part of a data engineering + ML engineering portfolio series.*  
*Project 1: Transjakarta Demand Analytics (Python → ETL → SQL Server → SSAS → Power BI)*
