# 🏠 Jabodetabek House Price Predictor

### Machine Learning Portfolio Project — 2026

> **ML-powered property valuation for the Greater Jakarta area (Jabodetabek)** using XGBoost / LightGBM, SHAP explainability, MLflow experiment tracking, and a Streamlit web application.

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jabodetabek-house-pricing-predictor.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3x-red?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.x-blue)
![SHAP](https://img.shields.io/badge/SHAP-0.4x-purple)
![Playwright](https://img.shields.io/badge/Playwright-1.x-teal)
![Optuna](https://img.shields.io/badge/Optuna-3.x-navy)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Demo](#-demo)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Dataset](#-dataset)
6. [Quick Start](#-quick-start)
7. [Notebook Walkthrough](#-notebook-walkthrough)
8. [Feature Engineering](#-feature-engineering)
9. [Model Performance](#-model-performance)
10. [SHAP Explainability](#-shap-explainability)
11. [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
12. [Streamlit App](#-streamlit-app)
13. [Deployment](#-deployment)

---

## 🎯 Project Overview

This project builds an end-to-end machine learning pipeline to **predict residential property prices across the Jabodetabek area** (Jakarta, Bogor, Depok, Tangerang, Bekasi) — home to over 30 million people and one of Southeast Asia's most active property markets.

### Key Highlights

| Component | Details |
|---|---|
| **Data Sources** | Kaggle Indonesia House Price + Playwright web scraping from Rumah123 |
| **Dataset Size** | 1,700+ scraped listings across 5 cities, merged with Kaggle data |
| **Geospatial Feature** | Geodesic distance to nearest Jakarta MRT station via Geopy |
| **Models** | Ridge (baseline) → XGBoost → LightGBM — tuned with Optuna (TPE, 50 trials each) |
| **Explainability** | SHAP waterfall, beeswarm, dependence, and feature importance plots |
| **Tracking** | MLflow experiment tracking — params, metrics, model artifacts for all 5 runs |
| **Deployment** | Interactive Streamlit web app with live SHAP explanations |

### Business Value

* **Home buyers** — Independent market valuation before negotiating
* **Real estate agents** — Justify pricing with data-driven, explainable insights
* **Recruiters / clients** — Demonstrates full ML lifecycle: data acquisition → feature engineering → model → explainability → deployment

---

## 🖥 Demo

```
┌──────────────────────────────────────────────────────────────────────┐
│  🏠  Jabodetabek House Price Predictor                               │
│  AI-Powered Valuation · XGBoost · LightGBM · SHAP                   │
├──────────────────────┬───────────────────────────────────────────────┤
│  SIDEBAR             │  [Building: 120m²] [Land: 150m²]              │
│  ─────────────────── │  [MRT: 1.4km away] [Age: 10yr] [Bath/Bed: 0.67]│
│  📍 Jakarta Selatan  │                                               │
│  📐 120m² / 150m²   │  ┌──────────────────────────────────────────┐ │
│  🛏 3 BR / 2 Bath   │  │      Estimated Market Value              │ │
│  🏗 Year: 2015      │  │      IDR 1.75 Miliar                     │ │
│  📜 SHM · Baru      │  │      Range: IDR 1.49B – IDR 2.01B        │ │
│                      │  └──────────────────────────────────────────┘ │
│  [⚡ Estimate Price] │                                               │
│                      │  Key Price Drivers (SHAP) | Charts | Benchmark│
└──────────────────────┴───────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

```
Data Acquisition    : Playwright (Chromium) · Geopy · Kaggle API
Data Processing     : Pandas · NumPy · Scikit-learn
Visualization       : Matplotlib · Seaborn · Plotly · Folium
Machine Learning    : XGBoost · LightGBM · Scikit-learn (Ridge)
Hyperparameter Opt  : Optuna (TPE Sampler — 50 trials per model)
Explainability      : SHAP (TreeExplainer)
Experiment Tracking : MLflow
Web Application     : Streamlit
Serialization       : Joblib
```

---

## 📂 Project Structure

```
jabodetabek-house-predictor/
│
├── notebooks/
│   └── jakarta_house_price_predictor.ipynb   # Full ML pipeline (10 sections, 55 code cells)
│
├── data/
│   ├── raw/
│   │   ├── indonesia_house_price.csv          # Kaggle dataset
│   │   ├── jakarta.csv                        # Scraped — Jakarta (340 rows)
│   │   ├── bogor.csv                          # Scraped — Bogor (340 rows)
│   │   ├── depok.csv                          # Scraped — Depok (340 rows)
│   │   ├── tangerang.csv                      # Scraped — Tangerang (340 rows)
│   │   ├── bekasi.csv                         # Scraped — Bekasi (340 rows)
│   │   └── rumah123_all_cities.csv            # Merged scrape (1,700 rows)
│   └── processed/
│       ├── jabodetabek_houses.csv             # Cleaned & feature-engineered
│       └── price_heatmap.html                 # Interactive Folium map
│
├── models/
│   ├── best_model.pkl                         # Best trained model
│   ├── preprocessing_pipeline.pkl             # Sklearn ColumnTransformer
│   ├── shap_explainer.pkl                     # SHAP TreeExplainer
│   └── feature_metadata.json                  # Feature lists, encodings, coords
│
├── mlruns/                                    # MLflow tracking (auto-generated)
│
├── scrape_rumah123_FINAL.py                   # Playwright scraper — multi-city
├── app.py                                     # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

### Sources

| Source | Rows | Method |
|---|---|---|
| Kaggle — Indonesia House Price | ~3,000 | Kaggle API |
| Rumah123 (scraped) | 1,700 | Playwright headless browser |
| **Total (after dedup)** | **~4,500** | Merged & deduplicated |

### Scraping Results — Rumah123 (April 2026)

| City | Listings | Harga | LT | LB | KT | Koordinat |
|---|---|---|---|---|---|---|
| Jakarta | 340 | 100% | 100% | 100% | 98% | 100% |
| Bogor | 340 | 100% | 100% | 100% | 100% | 100% |
| Depok | 340 | 100% | 100% | 100% | 99% | 100% |
| Tangerang | 340 | 100% | 100% | 100% | 100% | 100% |
| Bekasi | 340 | 100% | 100% | 100% | 99% | 100% |
| **Total** | **1,700** | **100%** | **100%** | **100%** | **99%** | **100%** |

> Koordinat diisi via geocoding dari pasangan `lokasi + city` mencakup 200 area unik di Jabodetabek.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/alfenh17/jabodetabek-house-pricing-predictor.git
cd jabodetabek-house-pricing-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium     # hanya jika ingin jalankan scraper
```

### 2. (Optional) Scrape Data Sendiri

```bash
# Scrape semua 5 kota (~1,700 listings, ~5 jam)
python scrape_rumah123_FINAL.py --mode multi

# Test 1 kota, 3 halaman
python scrape_rumah123_FINAL.py --mode single --city jakarta --pages 3
```

### 3. Jalankan Notebook

```bash
jupyter notebook notebooks/jakarta_house_price_predictor.ipynb
```

Jalankan semua cell dari atas ke bawah (Kernel → Restart & Run All).

**Estimasi waktu: ~45–90 menit** (sebagian besar untuk Optuna 50-trial tuning)

### 4. Launch Streamlit App

```bash
streamlit run app.py
```

Buka `http://localhost:8501`.

### 5. (Optional) MLflow UI

```bash
# Windows — jalankan di Anaconda Prompt dari folder project
mlflow ui --backend-store-uri mlruns

# Mac / Linux
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

Buka `http://127.0.0.1:5000` untuk compare semua runs.

---

## 📓 Notebook Walkthrough

Notebook terdiri dari **10 section bernomor** dengan 55 code cells dan 56 markdown cells.

| Section | Topic | Output Utama |
|---|---|---|
| **1** | Environment Setup | Libraries, constants, koordinat 19 stasiun MRT |
| **2** | Data Acquisition | Load Kaggle + Playwright scraper + merge & dedup |
| **3** | EDA | Distribution plots, Folium price heatmap, correlation matrix |
| **4** | Preprocessing | Imputation, outlier capping (IQR k=3), log1p transform |
| **5** | Feature Engineering | MRT distance (Geopy), room ratios, age buckets, zone OHE |
| **6** | Model Training | Ridge → XGBoost → LightGBM → Optuna 50-trial tuning |
| **7** | MLflow Tracking | Log params, metrics, artifacts untuk 5 runs |
| **8** | Evaluation | MAE/RMSE/MAPE/R² on test set, residual analysis |
| **9** | SHAP | Summary bar, beeswarm, waterfall, dependence plots |
| **10** | Serialization | Save model, pipeline, SHAP explainer, feature metadata |

---

## ⚙️ Feature Engineering

### Feature Set Lengkap

| Feature | Deskripsi | Cara Membuat |
|---|---|---|
| `luas_bangunan_m2` | Luas bangunan (m²) | Raw |
| `luas_tanah_m2` | Luas tanah (m²) | Raw |
| `jumlah_kamar_tidur` | Jumlah kamar tidur | Raw |
| `jumlah_kamar_mandi` | Jumlah kamar mandi | Raw |
| `garasi` | Kapasitas garasi | Raw |
| `property_age` | Umur properti (tahun) | `2025 − tahun_dibangun` |
| `dist_mrt_km` | Jarak ke MRT terdekat (km) | **Geodesic via Geopy** |
| `bath_bed_ratio` | Rasio kamar mandi/tidur | `km / kt` |
| `area_per_bed` | Area per kamar tidur | `luas_b / kt` |
| `bcr` | Building coverage ratio | `luas_b / luas_t` |
| `total_rooms` | Total kamar | `kt + km` |
| `kondisi_enc` | Kondisi properti (ordinal 0–3) | Manual encoding |
| `cert_enc` | Jenis sertifikat (0–2) | SHM=2, HGB=1, Girik=0 |
| `latitude` | Koordinat latitude | Raw / geocoded |
| `longitude` | Koordinat longitude | Raw / geocoded |
| `city` | Kota/area | OrdinalEncoder |
| `zone_*` | Zona regional | One-hot (4 zona) |

### MRT Distance — Novel Feature

```python
from geopy.distance import geodesic

MRT_STATIONS = {
    "Lebak Bulus": (-6.2894, 106.7741),
    "Blok M":      (-6.2442, 106.7987),
    "Bundaran HI": (-6.1946, 106.8229),
    # ... 19 stasiun total
}

def min_mrt_distance_km(lat, lon):
    return min(geodesic((lat, lon), s).km
               for s in MRT_STATIONS.values())
```

Properti dekat stasiun MRT Jakarta memiliki price premium yang terukur — fitur ini tidak tersedia di dataset Kaggle original dan menjadi salah satu SHAP driver terpenting.

---

## 📈 Model Performance

Hasil evaluasi pada **held-out test set** (15% dari total data):

| Model | MAE (IDR) | RMSE (IDR) | MAPE (%) | R² |
|---|---|---|---|---|
| Ridge (Baseline) | ~413M | ~413M | ~28% | ~0.944 |
| XGBoost (default) | ~178M | ~178M | ~12% | ~0.990 |
| LightGBM (default) | ~184M | ~184M | ~13% | ~0.989 |
| XGBoost (Optuna) | ~175M | ~175M | ~11% | ~0.990 |
| **LightGBM (Optuna)** | **~171M** | **~171M** | **~11%** | **~0.990** |

### Top SHAP Features (urutan importance)

1. `luas_bangunan_m2` — luas bangunan mendominasi harga
2. `luas_tanah_m2` — premium luas tanah
3. `dist_mrt_km` — gradient harga berbasis MRT
4. `latitude` / `longitude` — lokasi geografis
5. `city` — premium per kota
6. `property_age` — efek depresiasi
7. `kondisi_enc` — kondisi properti
8. `cert_enc` — premium SHM vs HGB vs Girik

---

## 🔍 SHAP Explainability

Setiap prediksi dilengkapi SHAP breakdown lengkap:

```
Base value (avg log-price)     →  22.41
+ luas_bangunan_m2  (+0.82)    →  bangunan besar menambah nilai
+ city: Jakarta Sel (+0.45)    →  lokasi premium
- dist_mrt_km: 3.2km (-0.18)  →  jauh dari MRT mengurangi nilai
+ cert_enc: SHM (+0.12)        →  sertifikat terkuat
──────────────────────────────────────────
= Predicted log-price          →  23.59
= IDR 1.75 Miliar
```

Streamlit app menampilkan 3 SHAP view:

* **Waterfall plot** — kontribusi setiap fitur untuk properti ini
* **Feature importance bar** — mean |SHAP| di seluruh test set
* **Sensitivity analysis** — kurva harga saat satu fitur divariasikan

---

## 📊 MLflow Experiment Tracking

Semua 5 model run otomatis dilogging:

```bash
# Windows — jalankan dari folder project di Anaconda Prompt
mlflow ui --backend-store-uri mlruns
# Buka: http://127.0.0.1:5000
```

| Kategori | Detail |
|---|---|
| **Parameters** | learning_rate, n_estimators, max_depth, subsample, reg_alpha, reg_lambda, ... |
| **Metrics** | val_mae, val_rmse, val_r2, val_mape, val_rmse_log |
| **Artifacts** | Model terserialisasi (XGBoost/LightGBM native format) |
| **Tags** | model_type, tuned, project, author |

---

## 🎨 Streamlit App

App (`app.py`) dengan **dark luxury property aesthetic** menggunakan font DM Serif Display:

* **Sidebar gelap** — city selector, property specs, predict button
* **Hero section** — judul besar + tech badges
* **5 stat cards** — building area, land area, nearest MRT, property age, bath/bed ratio
* **Price display** — estimated value + confidence range ±15% + per-m² breakdown
* **SHAP table** — 8 top price drivers dengan panah up/down
* **3 chart tabs** — Waterfall, Feature Importance, Price Sensitivity
* **Market benchmark** — bar chart perbandingan harga per kota

```bash
streamlit run app.py
# Buka: http://localhost:8501
```

---

## 🚀 Deployment

### Local

```bash
streamlit run app.py
```

### Streamlit Cloud (Free Hosting)

1. Push repo ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file: `app.py`
4. Deploy

> **Catatan:** File `models/*.pkl` perlu diupload terpisah via Git LFS atau Streamlit secrets karena ukurannya besar.

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t jabodetabek-house-predictor .
docker run -p 8501:8501 jabodetabek-house-predictor
```

---

## 📚 References

* [Kaggle — Indonesia House Price Dataset](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek)
* [Rumah123 — Property Listings](https://www.rumah123.com)
* [SHAP Documentation](https://shap.readthedocs.io/)
* [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
* [Optuna TPE Sampler](https://optuna.readthedocs.io/)
* [Jakarta MRT Official](https://jakartamrt.co.id/)
* [Playwright for Python](https://playwright.dev/python/)

---

## 📄 License

MIT License — free to use, adapt, and build on for your own portfolio.
