"""
app.py — Jakarta House Price Predictor
Redesigned with premium property aesthetic
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
import joblib
import shap
from pathlib import Path
from geopy.distance import geodesic

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jabodetabek House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "models" / "best_model.pkl"
PIPELINE_PATH  = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
EXPLAINER_PATH = BASE_DIR / "models" / "shap_explainer.pkl"
META_PATH      = BASE_DIR / "models" / "feature_metadata.json"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem; }

/* ── Sembunyikan tombol collapse sidebar PERMANEN ── */
[data-testid="collapsedControl"] { display: none !important; }
button[data-testid="baseButton-header"] { display: none !important; }
button[kind="headerNoPadding"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > div > button { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1rem 0 !important;
}

/* ── Main background ── */
.stApp {
    background: #060d1a;
}

/* ── Hero title ── */
.hero-wrap {
    background: linear-gradient(135deg, #0d1b2e 0%, #0f2744 50%, #0a1929 100%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 2.5rem 3rem 2rem;
    margin: 0 -1rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(246,173,85,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #f7fafc;
    line-height: 1.15;
    margin: 0 0 0.6rem;
}
.hero-title span { color: #63b3ed; }
.hero-sub {
    font-size: 0.88rem;
    color: #718096;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.7rem;
    color: #63b3ed;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* ── Stat cards row ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: #0d1b2e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(99,179,237,0.3); }
.stat-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5568;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 600;
    color: #e2e8f0;
    line-height: 1;
}
.stat-unit {
    font-size: 0.7rem;
    color: #4a5568;
    margin-top: 0.2rem;
}
.stat-card.accent .stat-value { color: #63b3ed; }
.stat-card.gold .stat-value { color: #f6ad55; }

/* ── Price hero ── */
.price-hero {
    background: linear-gradient(135deg, #0d1b2e 0%, #1a365d 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.price-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,179,237,0.08) 0%, transparent 60%);
}
.price-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.5rem;
    position: relative;
}
.price-main {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #f7fafc;
    margin: 0.2rem 0;
    position: relative;
}
.price-range {
    font-size: 0.8rem;
    color: #4a5568;
    margin-top: 0.5rem;
    position: relative;
}
.price-range span { color: #718096; }

/* ── Mini metric cards below price ── */
.mini-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 1rem;
}
.mini-card {
    background: #060d1a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.mini-label {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a5568;
}
.mini-value {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-top: 0.2rem;
}

/* ── Section headings ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.section-head-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #e2e8f0;
    font-weight: 400;
}
.section-dot {
    width: 6px; height: 6px;
    background: #63b3ed;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── SHAP table ── */
.shap-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.shap-table th {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a5568;
    padding: 0.5rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.shap-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #cbd5e0;
}
.shap-table tr:hover td { background: rgba(255,255,255,0.02); }
.up   { color: #fc8181; font-weight: 600; }
.down { color: #68d391; font-weight: 600; }

/* ── Sidebar predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2b6cb0, #1a4a7a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    transform: translateY(-1px) !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1b2e !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
.stTabs [data-baseweb="tab"] {
    color: #718096 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.15) !important;
    color: #63b3ed !important;
}

/* ── Matplotlib dark ── */
.element-container { color: #e2e8f0; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    padding: 1.5rem 1rem 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1rem;
}
.sidebar-logo-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #f7fafc;
}
.sidebar-logo-sub {
    font-size: 0.7rem;
    color: #4a5568;
    margin-top: 0.2rem;
    letter-spacing: 0.05em;
}
.sidebar-section-label {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #2d3748;
    padding: 0.8rem 0 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    missing = [p for p in [MODEL_PATH, PIPELINE_PATH, EXPLAINER_PATH, META_PATH] if not p.exists()]
    if missing:
        return None, None, None, None
    model     = joblib.load(MODEL_PATH)
    pipeline  = joblib.load(PIPELINE_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, pipeline, explainer, meta


model, pipeline, explainer, meta = load_artifacts()
DEMO_MODE = model is None

MRT_STATIONS = {
    "Lebak Bulus":(-6.2894,106.7741),"Fatmawati":(-6.2947,106.7946),
    "Cipete Raya":(-6.2886,106.8016),"Haji Nawi":(-6.2748,106.7989),
    "Blok A":(-6.2607,106.7981),"Blok M":(-6.2442,106.7987),
    "ASEAN":(-6.2378,106.7983),"Senayan":(-6.2278,106.8009),
    "Istora":(-6.2210,106.8037),"Bendungan Hilir":(-6.2109,106.8206),
    "Setiabudi":(-6.2100,106.8283),"Dukuh Atas":(-6.2014,106.8229),
    "Bundaran HI":(-6.1946,106.8229),"Monas":(-6.1869,106.8229),
    "Harmoni":(-6.1706,106.8181),"Sawah Besar":(-6.1591,106.8293),
    "Mangga Besar":(-6.1510,106.8372),"Glodok":(-6.1467,106.8443),
    "Jakarta Kota":(-6.1378,106.8127),
}

CITY_COORDINATES = {
    "Jakarta Selatan":(-6.2615,106.8106),"Jakarta Pusat":(-6.1751,106.8272),
    "Jakarta Barat":(-6.1682,106.7632),"Jakarta Timur":(-6.2251,106.9004),
    "Jakarta Utara":(-6.1382,106.8742),"Depok":(-6.4025,106.7942),
    "Bogor":(-6.5971,106.8060),"Tangerang":(-6.1781,106.6300),
    "Tangerang Selatan":(-6.2877,106.7164),"Bekasi":(-6.2383,107.0050),
}

KONDISI_ORDER = {"Butuh Renovasi":0,"Sedang":1,"Bagus":2,"Baru":3}
CERT_ORDER    = {"Girik":0,"HGB":1,"SHM":2}
ZONE_MAP = {
    "Jakarta Selatan":"Jakarta Core","Jakarta Pusat":"Jakarta Core",
    "Jakarta Barat":"Jakarta Outer","Jakarta Timur":"Jakarta Outer",
    "Jakarta Utara":"Jakarta Outer","Depok":"Satellite South",
    "Bogor":"Satellite South","Tangerang":"Satellite West",
    "Tangerang Selatan":"Satellite West","Bekasi":"Satellite East",
}
CURRENT_YEAR = 2025
ALL_CITIES   = list(CITY_COORDINATES.keys())


def min_mrt_distance_km(lat, lon):
    return min(geodesic((lat,lon), s).km for s in MRT_STATIONS.values())


def build_features(d, meta):
    d = dict(d)
    d["property_age"]   = CURRENT_YEAR - int(d.get("tahun_dibangun", 2010))
    d["bath_bed_ratio"] = d["jumlah_kamar_mandi"] / max(d["jumlah_kamar_tidur"],1)
    d["area_per_bed"]   = d["luas_bangunan_m2"] / max(d["jumlah_kamar_tidur"],1)
    d["bcr"]            = d["luas_bangunan_m2"] / max(d["luas_tanah_m2"],1)
    d["total_rooms"]    = d["jumlah_kamar_tidur"] + d["jumlah_kamar_mandi"]
    d["kondisi_enc"]    = KONDISI_ORDER.get(d.get("kondisi","Bagus"),2)
    d["cert_enc"]       = CERT_ORDER.get(d.get("sertifikat","SHM"),2)
    d["dist_mrt_km"]    = min_mrt_distance_km(d["latitude"], d["longitude"])
    zone = ZONE_MAP.get(d.get("city",""),"Other")
    for col in meta.get("features_ohe",[]):
        zone_label = col.replace("zone_","").replace("_"," ")
        d[col] = 1 if zone_label in zone else 0
    all_features = meta.get("all_features",[])
    return pd.DataFrame([{k: d.get(k,0) for k in all_features}])


def predict_price(input_dict):
    row   = build_features(input_dict, meta)
    X_inf = pipeline.transform(row)
    price = float(np.expm1(model.predict(X_inf)[0]))
    sv    = explainer.shap_values(X_inf)
    if isinstance(sv, list): sv = sv[0]
    sv = sv[0] if hasattr(sv,'ndim') and sv.ndim==2 else sv
    return {
        "price_idr": price, "price_miliar": price/1e9,
        "shap_values": sv, "feature_names": meta.get("all_features",[]),
        "base_value": float(explainer.expected_value
                            if not isinstance(explainer.expected_value, np.ndarray)
                            else explainer.expected_value[0]),
    }


def demo_predict(input_dict):
    city    = input_dict.get("city","Jakarta Selatan")
    luas_b  = input_dict.get("luas_bangunan_m2",100)
    luas_t  = input_dict.get("luas_tanah_m2",120)
    kt      = input_dict.get("jumlah_kamar_tidur",3)
    km      = input_dict.get("jumlah_kamar_mandi",2)
    garasi  = input_dict.get("garasi",1)
    cert    = input_dict.get("sertifikat","SHM")
    kondisi = input_dict.get("kondisi","Bagus")
    lat     = input_dict.get("latitude",-6.26)
    lon     = input_dict.get("longitude",106.81)
    year    = input_dict.get("tahun_dibangun",2015)
    cp = {"Jakarta Selatan":1.4,"Jakarta Pusat":1.35,"Jakarta Barat":1.1,
          "Jakarta Timur":1.05,"Jakarta Utara":1.0,"Bogor":0.7,
          "Depok":0.75,"Tangerang":0.8,"Tangerang Selatan":0.9,"Bekasi":0.75}.get(city,1.0)
    price = (cp*(luas_b*6e6+luas_t*3e6)+kt*50e6+garasi*30e6)
    price *= {"SHM":1.0,"HGB":0.92,"Girik":0.80}.get(cert,1.0)
    price *= {"Baru":1.1,"Bagus":1.0,"Sedang":0.9,"Butuh Renovasi":0.75}.get(kondisi,1.0)
    dist   = min_mrt_distance_km(lat,lon)
    price *= max(0.7,1.0-dist*0.03)
    price *= max(0.7,1.0-(CURRENT_YEAR-year)*0.003)
    price  = max(price,200e6)
    sv = np.array([luas_b*0.002,luas_t*0.0008,kt*0.05,km*0.03,garasi*0.02,
                   -(CURRENT_YEAR-year)*0.003,-dist*0.04,km/max(kt,1)*0.02,
                   luas_b/max(kt,1)*0.001,luas_b/max(luas_t,1)*0.01,
                   (kt+km)*0.01,KONDISI_ORDER.get(kondisi,2)*0.02,
                   CERT_ORDER.get(cert,2)*0.03,lat*-0.01,lon*0.01])
    feats = ["luas_bangunan_m2","luas_tanah_m2","jumlah_kamar_tidur","jumlah_kamar_mandi",
             "garasi","property_age","dist_mrt_km","bath_bed_ratio","area_per_bed",
             "bcr","total_rooms","kondisi_enc","cert_enc","latitude","longitude"]
    return {"price_idr":price,"price_miliar":price/1e9,"shap_values":sv,
            "feature_names":feats,"base_value":22.0}


# ─── Matplotlib dark theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0d1b2e",
    "axes.facecolor"   : "#0d1b2e",
    "axes.edgecolor"   : "#1e2d45",
    "axes.labelcolor"  : "#718096",
    "xtick.color"      : "#4a5568",
    "ytick.color"      : "#4a5568",
    "text.color"       : "#e2e8f0",
    "grid.color"       : "#1a2744",
    "grid.linewidth"   : 0.5,
    "figure.dpi"       : 110,
})


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-title">🏠 HargaRumahku.id</div>
            <div class="sidebar-logo-sub">JAKARTA HOUSE PREDICTOR</div>
        </div>
    """, unsafe_allow_html=True)

    if DEMO_MODE:
        st.warning("⚠ Demo Mode — train model first", icon="⚠️")

    # Location
    st.markdown('<div class="sidebar-section-label">📍 Location</div>', unsafe_allow_html=True)
    city = st.selectbox("City / Area", ALL_CITIES, index=0, label_visibility="collapsed")
    lat  = CITY_COORDINATES[city][0]
    lon  = CITY_COORDINATES[city][1]

    # Size
    st.markdown('<div class="sidebar-section-label">📐 Property Size</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        luas_bangunan = st.number_input("Building (m²)", 15, 2000, 120, 10)
    with col2:
        luas_tanah = st.number_input("Land (m²)", 15, 5000, 150, 10)

    # Rooms
    st.markdown('<div class="sidebar-section-label">🛏 Rooms</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        kamar_tidur = st.number_input("Bed", 1, 15, 3)
    with col4:
        kamar_mandi = st.number_input("Bath", 1, 15, 2)
    with col5:
        garasi = st.number_input("Garage", 0, 10, 1)

    # Details
    st.markdown('<div class="sidebar-section-label">📋 Details</div>', unsafe_allow_html=True)
    tahun_dibangun = st.number_input("Year Built", 1950, CURRENT_YEAR, 2015)
    sertifikat     = st.selectbox("Certificate", ["SHM","HGB","Girik"])
    kondisi        = st.selectbox("Condition", ["Baru","Bagus","Sedang","Butuh Renovasi"])

    with st.expander("🌐 Custom Coordinates"):
        lat = st.number_input("Latitude",  -6.9, -5.8, lat, format="%.4f")
        lon = st.number_input("Longitude", 106.4, 107.2, lon, format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Estimate Price", use_container_width=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

# Hero
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">AI-Powered Valuation · Jabodetabek</div>
    <div class="hero-title">Jabodetabek <span>House Price</span><br>Predictor</div>
    <div class="hero-sub">
        Machine learning valuation engine · XGBoost / LightGBM ·
        SHAP Explainability · {len(ALL_CITIES)} cities covered
    </div>
    <div class="hero-badges">
        <span class="badge">XGBoost</span>
        <span class="badge">LightGBM</span>
        <span class="badge">SHAP</span>
        <span class="badge">MLflow</span>
        <span class="badge">Optuna</span>
        <span class="badge">Geopy MRT Distance</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Stat cards
dist_mrt = min_mrt_distance_km(lat, lon)
age      = CURRENT_YEAR - tahun_dibangun
b_ratio  = kamar_mandi / max(kamar_tidur, 1)

st.markdown(f"""
<div class="stat-row">
    <div class="stat-card accent">
        <div class="stat-label">Building Area</div>
        <div class="stat-value">{luas_bangunan}</div>
        <div class="stat-unit">m²</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Land Area</div>
        <div class="stat-value">{luas_tanah}</div>
        <div class="stat-unit">m²</div>
    </div>
    <div class="stat-card gold">
        <div class="stat-label">Nearest MRT</div>
        <div class="stat-value">{dist_mrt:.1f}</div>
        <div class="stat-unit">km away</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Property Age</div>
        <div class="stat-value">{age}</div>
        <div class="stat-unit">years</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Bath / Bed</div>
        <div class="stat-value">{b_ratio:.2f}</div>
        <div class="stat-unit">luxury ratio</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Prediction ───────────────────────────────────────────────────────────────
input_dict = {
    "city": city, "luas_bangunan_m2": float(luas_bangunan),
    "luas_tanah_m2": float(luas_tanah), "jumlah_kamar_tidur": int(kamar_tidur),
    "jumlah_kamar_mandi": int(kamar_mandi), "garasi": int(garasi),
    "tahun_dibangun": int(tahun_dibangun), "sertifikat": sertifikat,
    "kondisi": kondisi, "latitude": float(lat), "longitude": float(lon),
}

with st.spinner("Computing prediction..."):
    try:
        result = demo_predict(input_dict) if DEMO_MODE else predict_price(input_dict)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

price     = result["price_idr"]
price_mil = result["price_miliar"]
shap_vals = result["shap_values"]
feat_names = result["feature_names"]

lo = price * 0.85
hi = price * 1.15
ppsqm = price / luas_bangunan
ppsqm_t = price / luas_tanah

# Layout: price left, SHAP table right
col_price, col_shap = st.columns([3, 2], gap="large")

with col_price:
    st.markdown(f"""
    <div class="price-hero">
        <div class="price-eyebrow">Estimated Market Value</div>
        <div class="price-main">IDR {price_mil:.2f} Miliar</div>
        <div class="price-range">
            <span>Confidence range:</span>
            IDR {lo/1e9:.2f}B – IDR {hi/1e9:.2f}B
        </div>
        <div class="mini-metrics">
            <div class="mini-card">
                <div class="mini-label">Per Building m²</div>
                <div class="mini-value">IDR {ppsqm/1e6:.1f}Jt</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Per Land m²</div>
                <div class="mini-value">IDR {ppsqm_t/1e6:.1f}Jt</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Certificate</div>
                <div class="mini-value">{sertifikat}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_shap:
    st.markdown("""
    <div class="section-head">
        <div class="section-dot"></div>
        <div class="section-head-title">Key Price Drivers</div>
    </div>
    """, unsafe_allow_html=True)

    n     = min(8, len(shap_vals))
    top_i = np.argsort(np.abs(shap_vals))[-n:][::-1]

    rows = ""
    for i in top_i:
        sv  = float(shap_vals[i])
        fn  = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        cls = "up" if sv > 0 else "down"
        arr = "↑ raises" if sv > 0 else "↓ lowers"
        rows += f"""
        <tr>
            <td>{fn}</td>
            <td class="{cls}">{arr}</td>
            <td style="text-align:right;color:#4a5568;">{sv:+.3f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="shap-table">
        <thead><tr>
            <th>Feature</th><th>Impact</th><th style="text-align:right">SHAP</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ─── Charts ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <div class="section-dot"></div>
    <div class="section-head-title">SHAP Explainability</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌊 Waterfall (This Property)", "📊 Feature Importance", "📈 Price Sensitivity"])

with tab1:
    n_d    = min(12, len(shap_vals))
    top_n  = np.argsort(np.abs(shap_vals))[-n_d:]
    names_ = [feat_names[i] if i<len(feat_names) else f"feat_{i}" for i in top_n]
    vals_  = [shap_vals[i] for i in top_n]
    pairs  = sorted(zip(vals_, names_), key=lambda x: x[0])
    vs, ns = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#fc8181" if v>0 else "#68d391" for v in vs]
    ax.barh(range(len(vs)), vs, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns, fontsize=9)
    ax.axvline(0, color="#2d4a6b", lw=1)
    ax.set_xlabel("SHAP value (impact on log-price)", fontsize=9)
    ax.set_title("Feature Contributions — This Property", fontsize=11, color="#e2e8f0", pad=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    abs_sv  = np.abs(shap_vals)
    top15   = np.argsort(abs_sv)[-15:][::-1]
    names15 = [feat_names[i] if i<len(feat_names) else f"feat_{i}" for i in top15]
    vals15  = abs_sv[top15]

    fig, ax = plt.subplots(figsize=(9, 5))
    norm    = vals15 / vals15.max()
    colors  = [plt.cm.YlOrRd(v) for v in norm]
    ax.barh(range(len(vals15))[::-1], vals15, color=colors, edgecolor="none")
    ax.set_yticks(range(len(names15))[::-1])
    ax.set_yticklabels(names15, fontsize=9)
    ax.set_xlabel("|SHAP Value|", fontsize=9)
    ax.set_title("Feature Importance (mean |SHAP|)", fontsize=11, color="#e2e8f0", pad=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    feat_sel = st.selectbox("Vary feature", [
        "luas_bangunan_m2", "luas_tanah_m2", "jumlah_kamar_tidur", "MRT distance"
    ])

    fig, ax = plt.subplots(figsize=(9, 4))

    if feat_sel == "luas_bangunan_m2":
        xs = np.linspace(30, 500, 40)
        ys = [
            (demo_predict if DEMO_MODE else predict_price)(
                dict(input_dict, luas_bangunan_m2=float(v),
                     luas_tanah_m2=max(float(v), float(luas_tanah)))
            )["price_idr"]/1e9 for v in xs
        ]
        ax.plot(xs, ys, color="#63b3ed", lw=2.5)
        ax.axvline(luas_bangunan, color="#f6ad55", ls="--", lw=1.5, label=f"Current: {luas_bangunan} m²")
        ax.set_xlabel("Building Area (m²)")

    elif feat_sel == "luas_tanah_m2":
        xs = np.linspace(30, 600, 40)
        ys = [(demo_predict if DEMO_MODE else predict_price)(
            dict(input_dict, luas_tanah_m2=float(v)))["price_idr"]/1e9 for v in xs]
        ax.plot(xs, ys, color="#68d391", lw=2.5)
        ax.axvline(luas_tanah, color="#f6ad55", ls="--", lw=1.5, label=f"Current: {luas_tanah} m²")
        ax.set_xlabel("Land Area (m²)")

    elif feat_sel == "jumlah_kamar_tidur":
        xs = range(1, 10)
        ys = [(demo_predict if DEMO_MODE else predict_price)(
            dict(input_dict, jumlah_kamar_tidur=int(v)))["price_idr"]/1e9 for v in xs]
        ax.bar(list(xs), ys, color="#b794f4", alpha=0.85, edgecolor="none")
        ax.set_xlabel("Bedrooms")

    else:
        lats = np.linspace(-6.45, -6.10, 30)
        ds   = [min_mrt_distance_km(la, lon) for la in lats]
        ys   = [(demo_predict if DEMO_MODE else predict_price)(
            dict(input_dict, latitude=float(la)))["price_idr"]/1e9 for la in lats]
        ax.plot(ds, ys, color="#fc8181", lw=2.5)
        ax.axvline(dist_mrt, color="#f6ad55", ls="--", lw=1.5, label=f"Current: {dist_mrt:.1f} km")
        ax.set_xlabel("Distance to Nearest MRT (km)")

    ax.set_ylabel("Predicted Price (Miliar IDR)")
    ax.set_title(f"Price Sensitivity — {feat_sel}", fontsize=11, color="#e2e8f0", pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"IDR {x:.1f}B"))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─── Market Benchmarks ────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <div class="section-dot"></div>
    <div class="section-head-title">Market Benchmarks — Jabodetabek</div>
</div>
""", unsafe_allow_html=True)

BENCHMARKS = {
    "Jakarta Selatan":(8.5,25.0),"Jakarta Pusat":(9.0,30.0),
    "Jakarta Barat":(5.0,15.0),"Jakarta Timur":(4.5,12.0),
    "Jakarta Utara":(4.0,11.0),"Depok":(2.5,7.0),
    "Bogor":(2.0,6.0),"Tangerang":(3.0,9.0),
    "Tangerang Selatan":(4.0,12.0),"Bekasi":(2.5,7.5),
}

fig, ax = plt.subplots(figsize=(12, 4))
cities_b = list(BENCHMARKS.keys())
lo_b = [BENCHMARKS[c][0] for c in cities_b]
hi_b = [BENCHMARKS[c][1] for c in cities_b]
x    = np.arange(len(cities_b))

ax.bar(x, hi_b, color="#1a2744", edgecolor="#2d4a6b", linewidth=0.5)
ax.bar(x, lo_b, color="#060d1a", edgecolor="none")

for i, c in enumerate(cities_b):
    if c == city:
        ax.bar(i, hi_b[i]-lo_b[i], bottom=lo_b[i],
               color="#2b6cb0", alpha=0.8, edgecolor="#63b3ed", linewidth=1.5)

ax.axhline(price_mil, color="#f6ad55", ls="--", lw=2,
           label=f"This prediction: IDR {price_mil:.2f}B")
ax.set_xticks(x)
ax.set_xticklabels(cities_b, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Price Range (Miliar IDR)", fontsize=9)
ax.set_title("Typical Price Range by City", fontsize=11, color="#e2e8f0", pad=12)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0;border-top:1px solid rgba(255,255,255,0.05);margin-top:2rem;">
    <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#2d3748;letter-spacing:0.05em;">
        HargaRumahku.id · Jabodetabek House Price Predictor
    </div>
    <div style="font-size:0.72rem;color:#2d3748;margin-top:0.4rem;">
        Portfolio Project 2026 · XGBoost / LightGBM · SHAP · MLflow · Streamlit
        <br>Predictions are estimates for educational purposes only.
    </div>
</div>
""", unsafe_allow_html=True)
