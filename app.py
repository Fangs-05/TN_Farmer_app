import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TN Farmer Crop Revenue Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.main { background: #0f1a0f; }

.stApp {
    background: linear-gradient(135deg, #0a1a0a 0%, #0f2010 50%, #0a1a0a 100%);
    color: #e8f5e8;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #c8e6c9 !important;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #81c784;
    line-height: 1.1;
    text-shadow: 0 2px 20px rgba(129,199,132,0.3);
}

.hero-sub {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1.05rem;
    color: #a5d6a7;
    font-weight: 300;
    letter-spacing: 0.05em;
}

.metric-card {
    background: linear-gradient(135deg, rgba(46,125,50,0.25), rgba(27,94,32,0.15));
    border: 1px solid rgba(129,199,132,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(10px);
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #81c784;
}

.metric-label {
    font-size: 0.82rem;
    color: #a5d6a7;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

.risk-low    { background: linear-gradient(135deg,rgba(46,125,50,0.4),rgba(27,94,32,0.2)); border:1px solid #4caf50; border-radius:10px; padding:1rem; }
.risk-medium { background: linear-gradient(135deg,rgba(230,81,0,0.3),rgba(191,54,12,0.2)); border:1px solid #ff9800; border-radius:10px; padding:1rem; }
.risk-high   { background: linear-gradient(135deg,rgba(183,28,28,0.4),rgba(136,14,79,0.2)); border:1px solid #f44336; border-radius:10px; padding:1rem; }

.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(129,199,132,0.5), transparent);
    margin: 2rem 0;
}

.badge {
    display: inline-block;
    background: rgba(129,199,132,0.15);
    border: 1px solid rgba(129,199,132,0.4);
    color: #a5d6a7;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.2rem;
}

.stSelectbox > div, .stSlider > div {
    background: rgba(27,94,32,0.15) !important;
}

div[data-testid="metric-container"] {
    background: rgba(46,125,50,0.2);
    border: 1px solid rgba(129,199,132,0.25);
    border-radius: 10px;
    padding: 0.8rem;
}

.stButton > button {
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    color: #c8e6c9;
    border: 1px solid #4caf50;
    border-radius: 8px;
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #388e3c, #2e7d32);
    border-color: #81c784;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(129,199,132,0.2);
}

.stTabs [data-baseweb="tab"] {
    color: #a5d6a7 !important;
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #81c784 !important;
    border-bottom: 2px solid #81c784 !important;
}

footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Tamil Nadu Data ────────────────────────────────────────────────────────────
TN_DISTRICTS = [
    "Thanjavur", "Tiruvarur", "Nagapattinam", "Pudukkottai",
    "Dindigul", "Madurai", "Virudhunagar", "Ramanathapuram",
    "Coimbatore", "Tiruppur", "Salem", "Namakkal",
    "Erode", "Karur", "Vellore", "Krishnagiri",
    "Dharmapuri", "Villupuram", "Cuddalore", "Ariyalur"
]

CROPS = ["Paddy (நெல்)", "Sugarcane (கரும்பு)", "Banana (வாழை)",
         "Turmeric (மஞ்சள்)", "Cotton (பருத்தி)", "Groundnut (நிலக்கடலை)"]

SEASONS = ["Kuruvai (June–Sep)", "Samba (Aug–Jan)", "Navarai (Jan–Apr)"]

SOIL_TYPES = ["Clay (களிமண்)", "Loamy (வண்டல்)", "Sandy (மணல்)", "Black Cotton (கருப்பு)"]

WATER_SOURCE = ["Canal (கால்வாய்)", "Borewell (துளை கிணறு)", "Rain-fed (மழைநீர்)", "Tank (ஏரி)"]

# Crop base revenue per acre (₹) and typical yield range
CROP_BASE = {
    "Paddy (நெல்)":         {"base": 28000, "cost_per_acre": 12000, "risk": 0.3},
    "Sugarcane (கரும்பு)":  {"base": 75000, "cost_per_acre": 35000, "risk": 0.2},
    "Banana (வாழை)":        {"base": 90000, "cost_per_acre": 40000, "risk": 0.4},
    "Turmeric (மஞ்சள்)":    {"base": 60000, "cost_per_acre": 25000, "risk": 0.35},
    "Cotton (பருத்தி)":     {"base": 40000, "cost_per_acre": 18000, "risk": 0.45},
    "Groundnut (நிலக்கடலை)":{"base": 35000, "cost_per_acre": 14000, "risk": 0.3},
}

DISTRICT_FACTOR = {d: round(np.random.uniform(0.85, 1.20), 2) for d in TN_DISTRICTS}
DISTRICT_FACTOR["Thanjavur"]   = 1.18
DISTRICT_FACTOR["Tiruvarur"]   = 1.15
DISTRICT_FACTOR["Coimbatore"]  = 1.12
DISTRICT_FACTOR["Erode"]       = 1.10
DISTRICT_FACTOR["Madurai"]     = 1.05

# ── Dataset Generation ─────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=800):
    np.random.seed(2024)
    records = []
    for _ in range(n):
        district    = np.random.choice(TN_DISTRICTS)
        crop        = np.random.choice(CROPS)
        season      = np.random.choice(SEASONS)
        soil        = np.random.choice(SOIL_TYPES)
        water       = np.random.choice(WATER_SOURCE)
        acres       = round(np.random.uniform(0.5, 10.0), 1)
        fertilizer  = np.random.randint(2000, 15000)
        rainfall    = np.random.randint(400, 1200)
        experience  = np.random.randint(1, 35)
        irrigation  = 1 if water != "Rain-fed (மழைநீர்)" else 0

        cb = CROP_BASE[crop]
        df_factor = DISTRICT_FACTOR[district]
        soil_bonus = {"Clay (களிமண்)":1.05,"Loamy (வண்டல்)":1.12,"Sandy (மணல்)":0.90,"Black Cotton (கருப்பு)":1.08}[soil]
        water_bonus = {"Canal (கால்வாய்)":1.10,"Borewell (துளை கிணறு)":1.05,"Rain-fed (மழைநீர்)":0.85,"Tank (ஏரி)":1.00}[water]
        season_bonus = {"Kuruvai (June–Sep)":1.00,"Samba (Aug–Jan)":1.08,"Navarai (Jan–Apr)":0.95}[season]
        exp_bonus   = 1 + (experience * 0.008)
        rain_bonus  = 1 + ((rainfall - 700) / 5000)

        revenue = (
            cb["base"] * acres * df_factor * soil_bonus
            * water_bonus * season_bonus * exp_bonus * rain_bonus
            + fertilizer * 1.8
            + np.random.normal(0, 3000)
        )
        revenue = max(revenue, 5000)

        records.append({
            "District": district, "Crop": crop, "Season": season,
            "Soil_Type": soil, "Water_Source": water,
            "Acres": acres, "Fertilizer_Cost": fertilizer,
            "Annual_Rainfall_mm": rainfall, "Farmer_Experience_yrs": experience,
            "Has_Irrigation": irrigation,
            "Expected_Revenue": int(revenue)
        })
    return pd.DataFrame(records)

# ── Train 3 Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    le = {}
    df2 = df.copy()
    for col in ["District","Crop","Season","Soil_Type","Water_Source"]:
        le[col] = LabelEncoder()
        df2[col+"_enc"] = le[col].fit_transform(df2[col])

    feat = ["Acres","Fertilizer_Cost","Annual_Rainfall_mm","Farmer_Experience_yrs",
            "Has_Irrigation","District_enc","Crop_enc","Season_enc","Soil_Type_enc","Water_Source_enc"]
    X = df2[feat]; y = df2["Expected_Revenue"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10),
        "Polynomial (deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lr",   LinearRegression())
        ])
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        yp = m.predict(X_test)
        results[name] = {
            "model": m,
            "r2":    round(r2_score(y_test, yp), 3),
            "mae":   int(mean_absolute_error(y_test, yp)),
            "y_test": y_test,
            "y_pred": yp
        }
    return results, le, feat, X_test, y_test

df_data = generate_dataset()
model_results, encoders, feature_cols, X_test_g, y_test_g = train_models(df_data)

# ── Helper: Predict ────────────────────────────────────────────────────────────
def predict_revenue(model, acres, fertilizer, rainfall, experience,
                    irrigation, district, crop, season, soil, water):
    row = {
        "Acres": acres, "Fertilizer_Cost": fertilizer,
        "Annual_Rainfall_mm": rainfall, "Farmer_Experience_yrs": experience,
        "Has_Irrigation": irrigation,
        "District_enc": encoders["District"].transform([district])[0],
        "Crop_enc":     encoders["Crop"].transform([crop])[0],
        "Season_enc":   encoders["Season"].transform([season])[0],
        "Soil_Type_enc":encoders["Soil_Type"].transform([soil])[0],
        "Water_Source_enc": encoders["Water_Source"].transform([water])[0],
    }
    inp = pd.DataFrame([row])[feature_cols]
    return max(int(model.predict(inp)[0]), 1000)

def get_risk_score(crop, water, rainfall, acres):
    base_risk = CROP_BASE[crop]["risk"]
    water_risk = 0.3 if water == "Rain-fed (மழைநீர்)" else 0.1
    rain_risk  = 0.2 if rainfall < 600 else (0.05 if rainfall > 1000 else 0.1)
    total = min((base_risk + water_risk + rain_risk) * 100, 95)
    return round(total)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
  <div class="hero-title">🌾 TN Farmer Crop Revenue Predictor</div>
  <div class="hero-sub" style="margin-top:0.5rem;">
    தமிழ்நாடு விவசாயி வருவாய் கணிப்பான் &nbsp;·&nbsp; 
    Empowering Tamil Nadu farmers with data-driven insights
  </div>
  <div style="margin-top:0.8rem;">
    <span class="badge">scikit-learn</span>
    <span class="badge">3 ML Models</span>
    <span class="badge">20 TN Districts</span>
    <span class="badge">6 Crops</span>
    <span class="badge">Risk Analysis</span>
  </div>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Revenue Predictor", "📊 Model Comparison", "🗺️ District Insights", "🗃️ Dataset"])

# ═══════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════
with tab1:
    st.markdown("### உங்கள் விவரங்களை உள்ளிடவும் (Enter Your Farm Details)")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.3])

    with c1:
        st.markdown("**🏡 Farm Info**")
        district   = st.selectbox("District (மாவட்டம்)", TN_DISTRICTS)
        crop       = st.selectbox("Crop (பயிர்)", CROPS)
        season     = st.selectbox("Season (பருவம்)", SEASONS)
        soil       = st.selectbox("Soil Type (மண் வகை)", SOIL_TYPES)

    with c2:
        st.markdown("**💧 Resources**")
        water      = st.selectbox("Water Source (நீர் ஆதாரம்)", WATER_SOURCE)
        acres      = st.slider("Land Area (நிலம்) — Acres", 0.5, 10.0, 2.0, 0.5)
        fertilizer = st.slider("Fertilizer Cost ₹", 2000, 15000, 6000, 500)
        rainfall   = st.slider("Expected Rainfall (மழை) mm", 400, 1200, 750, 50)

    with c3:
        st.markdown("**👨‍🌾 Farmer Profile**")
        experience = st.slider("Farming Experience (அனுபவம்) yrs", 1, 35, 10)
        irrigation = 1 if water != "Rain-fed (மழைநீர்)" else 0
        model_choice = st.selectbox("ML Model to Use", list(model_results.keys()))

        seed_cost   = st.slider("Seed Cost ₹", 500, 8000, 2000, 250)
        labour_cost = st.slider("Labour Cost ₹", 2000, 20000, 8000, 500)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    chosen_model = model_results[model_choice]["model"]
    predicted_rev = predict_revenue(chosen_model, acres, fertilizer, rainfall,
                                    experience, irrigation, district, crop, season, soil, water)

    total_cost    = int(CROP_BASE[crop]["cost_per_acre"] * acres) + seed_cost + labour_cost + fertilizer
    net_profit    = predicted_rev - total_cost
    profit_pct    = round((net_profit / total_cost) * 100, 1) if total_cost > 0 else 0
    risk_score    = get_risk_score(crop, water, rainfall, acres)
    rev_per_acre  = int(predicted_rev / acres)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🌾 Predicted Revenue", f"₹ {predicted_rev:,}")
    r2.metric("💸 Total Cost", f"₹ {total_cost:,}")
    r3.metric("💰 Net Profit / Loss",
              f"₹ {abs(net_profit):,}",
              delta=f"{'Profit ▲' if net_profit >= 0 else 'Loss ▼'} {profit_pct}%")
    r4.metric("📐 Revenue / Acre", f"₹ {rev_per_acre:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    rk1, rk2 = st.columns([1, 2])

    with rk1:
        risk_class = "risk-low" if risk_score < 30 else ("risk-medium" if risk_score < 60 else "risk-high")
        risk_emoji = "🟢" if risk_score < 30 else ("🟡" if risk_score < 60 else "🔴")
        risk_label = "LOW RISK" if risk_score < 30 else ("MODERATE RISK" if risk_score < 60 else "HIGH RISK")
        st.markdown(f"""
        <div class="{risk_class}">
          <div style="font-size:2.5rem; text-align:center">{risk_emoji}</div>
          <div style="text-align:center; font-family:'Playfair Display',serif; font-size:1.3rem; color:#c8e6c9; font-weight:700">{risk_score}/100</div>
          <div style="text-align:center; font-size:0.8rem; letter-spacing:0.1em; color:#a5d6a7; font-weight:600">{risk_label}</div>
          <hr style="border-color:rgba(255,255,255,0.1); margin:0.6rem 0">
          <div style="font-size:0.82rem; color:#c8e6c9;">
            {'✅ Good irrigation & rainfall' if risk_score < 30 else ('⚠️ Moderate drought exposure' if risk_score < 60 else '🚨 High drought / market risk')}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with rk2:
        st.markdown("#### 📊 Cost vs Revenue Breakdown")
        categories  = ["Seed Cost", "Labour", "Fertilizer", "Other Farm Cost", "Net Profit"]
        other_cost  = int(CROP_BASE[crop]["cost_per_acre"] * acres)
        values      = [seed_cost, labour_cost, fertilizer, other_cost, max(net_profit, 0)]
        colors_bar  = ["#ef9a9a","#ffcc80","#fff59d","#a5d6a7","#81c784"]
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#0f2010")
        ax.set_facecolor("#0f2010")
        bars = ax.barh(categories, values, color=colors_bar, edgecolor="none", height=0.55)
        for bar, val in zip(bars, values):
            ax.text(val + 200, bar.get_y() + bar.get_height()/2,
                    f"₹{val:,}", va='center', fontsize=8, color="#c8e6c9")
        ax.set_xlabel("Amount (₹)", color="#a5d6a7")
        ax.tick_params(colors="#a5d6a7")
        for spine in ax.spines.values():
            spine.set_edgecolor((0.506, 0.780, 0.518, 0.2))
        plt.tight_layout()
        st.pyplot(fig)

    # ── 12-Month Forecast ─────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📅 12-Month Revenue Forecast")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal_idx = [0.75,0.80,0.90,0.95,0.85,0.88,1.0,1.05,1.1,1.08,0.95,0.85]
    forecast = [int(predicted_rev * s + np.random.normal(0, 1500)) for s in seasonal_idx]
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    fig2.patch.set_facecolor("#0f2010")
    ax2.set_facecolor("#0f2010")
    ax2.fill_between(months, forecast, alpha=0.25, color="#81c784")
    ax2.plot(months, forecast, color="#81c784", linewidth=2.5, marker='o', markersize=5)
    ax2.axhline(total_cost, color="#ef9a9a", linestyle="--", linewidth=1.2, label="Break-even")
    for i, (m, v) in enumerate(zip(months, forecast)):
        ax2.text(i, v + 500, f"₹{v//1000}k", ha='center', fontsize=7, color="#c8e6c9")
    ax2.set_ylabel("Revenue (₹)", color="#a5d6a7")
    ax2.tick_params(colors="#a5d6a7")
    ax2.legend(facecolor="#0f2010", labelcolor="#c8e6c9")
    for spine in ax2.spines.values():
        spine.set_edgecolor((0.506, 0.780, 0.518, 0.15))
    plt.tight_layout()
    st.pyplot(fig2)

    # ── Advisor Tips ──────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🧠 Smart Advisor Tips (AI பரிந்துரைகள்)")
    tips = []
    if water == "Rain-fed (மழைநீர்)":
        tips.append("💧 **Switch to borewell or tank irrigation** — can increase revenue by up to 25%")
    if rainfall < 600:
        tips.append("☔ **Low rainfall warning** — consider drought-resistant varieties of your crop")
    if net_profit < 0:
        tips.append("⚠️ **Loss predicted** — consider reducing acres or switching to a higher-value crop like Banana or Sugarcane")
    if fertilizer > 10000:
        tips.append("🌿 **High fertilizer cost** — explore organic alternatives or government subsidy schemes")
    if experience < 5:
        tips.append("🎓 **Join a Farmer Producer Organisation (FPO)** — TN government offers mentorship programs")
    if soil == "Sandy (மணல்)":
        tips.append("🪴 **Sandy soil detected** — Groundnut or Cotton performs better than Paddy in this soil")
    if not tips:
        tips.append("✅ **Your farm setup looks healthy!** Maintain good irrigation and soil health for consistent yields.")

    for tip in tips:
        st.markdown(f"> {tip}")

# ═══════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown("### 🤖 Machine Learning Model Comparison")
    st.markdown("Three scikit-learn regression models trained on the same TN farm dataset:")

    mc1, mc2, mc3 = st.columns(3)
    cols_m = [mc1, mc2, mc3]
    model_colors = {"Linear Regression":"#64b5f6","Ridge Regression":"#ffb74d","Polynomial (deg 2)":"#81c784"}

    for i, (mname, mdata) in enumerate(model_results.items()):
        with cols_m[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; border-color:{list(model_colors.values())[i]}55">
              <div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:#c8e6c9;margin-bottom:0.5rem">{mname}</div>
              <div class="metric-value" style="color:{list(model_colors.values())[i]}">{mdata['r2']}</div>
              <div class="metric-label">R² Score</div>
              <hr style="border-color:rgba(255,255,255,0.1);margin:0.6rem 0">
              <div style="font-size:1.2rem;color:#c8e6c9;font-weight:600">₹ {mdata['mae']:,}</div>
              <div class="metric-label">Mean Abs Error</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    best = max(model_results, key=lambda k: model_results[k]["r2"])
    st.success(f"🏆 **Best Model: {best}** with R² = {model_results[best]['r2']}")

    fig3, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig3.patch.set_facecolor("#0f2010")
    for ax, (mname, mdata), color in zip(axes, model_results.items(), model_colors.values()):
        ax.set_facecolor("#0f2010")
        ax.scatter(mdata["y_test"], mdata["y_pred"], alpha=0.35, color=color, s=15, edgecolors='none')
        mn = min(mdata["y_test"].min(), mdata["y_pred"].min())
        mx = max(mdata["y_test"].max(), mdata["y_pred"].max())
        ax.plot([mn,mx],[mn,mx],'w--',linewidth=1,alpha=0.5)
        ax.set_title(f"{mname}\nR²={mdata['r2']}", color="#c8e6c9", fontsize=9)
        ax.set_xlabel("Actual ₹", color="#a5d6a7", fontsize=8)
        ax.set_ylabel("Predicted ₹", color="#a5d6a7", fontsize=8)
        ax.tick_params(colors="#a5d6a7", labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor((0.506, 0.780, 0.518, 0.15))
    plt.tight_layout()
    st.pyplot(fig3)

# ═══════════════════════════════════════════════════
# TAB 3 — DISTRICT INSIGHTS
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown("### 🗺️ Tamil Nadu District-wise Crop Insights")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("#### Average Revenue by District")
        dist_avg = df_data.groupby("District")["Expected_Revenue"].mean().sort_values(ascending=True)
        fig4, ax4 = plt.subplots(figsize=(6, 7))
        fig4.patch.set_facecolor("#0f2010")
        ax4.set_facecolor("#0f2010")
        colors_dist = ["#81c784" if v == dist_avg.max() else "#4caf5080" for v in dist_avg.values]
        bars4 = ax4.barh(dist_avg.index, dist_avg.values, color=colors_dist, edgecolor="none", height=0.65)
        for bar, val in zip(bars4, dist_avg.values):
            ax4.text(val+300, bar.get_y()+bar.get_height()/2,
                     f"₹{int(val/1000)}k", va='center', fontsize=7.5, color="#c8e6c9")
        ax4.set_xlabel("Avg Revenue (₹)", color="#a5d6a7")
        ax4.tick_params(colors="#a5d6a7", labelsize=8)
        for spine in ax4.spines.values(): spine.set_edgecolor((0.506, 0.780, 0.518, 0.1))
        plt.tight_layout()
        st.pyplot(fig4)

    with d2:
        st.markdown("#### Revenue by Crop Type")
        crop_avg = df_data.groupby("Crop")["Expected_Revenue"].mean().sort_values(ascending=False)
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        fig5.patch.set_facecolor("#0f2010")
        ax5.set_facecolor("#0f2010")
        crop_colors = ["#ffb74d","#81c784","#64b5f6","#f48fb1","#ce93d8","#80cbc4"]
        bars5 = ax5.bar(range(len(crop_avg)), crop_avg.values, color=crop_colors, edgecolor="none", width=0.6)
        ax5.set_xticks(range(len(crop_avg)))
        ax5.set_xticklabels([c.split("(")[0].strip() for c in crop_avg.index], rotation=30, ha='right', color="#a5d6a7", fontsize=8)
        ax5.tick_params(colors="#a5d6a7")
        ax5.set_ylabel("Avg Revenue (₹)", color="#a5d6a7")
        for bar, val in zip(bars5, crop_avg.values):
            ax5.text(bar.get_x()+bar.get_width()/2, val+500, f"₹{int(val/1000)}k",
                     ha='center', fontsize=8, color="#c8e6c9")
        for spine in ax5.spines.values(): spine.set_edgecolor((0.506, 0.780, 0.518, 0.1))
        plt.tight_layout()
        st.pyplot(fig5)

        st.markdown("#### Revenue by Water Source")
        water_avg = df_data.groupby("Water_Source")["Expected_Revenue"].mean().sort_values(ascending=False)
        fig6, ax6 = plt.subplots(figsize=(6, 3))
        fig6.patch.set_facecolor("#0f2010")
        ax6.set_facecolor("#0f2010")
        ax6.bar(range(len(water_avg)), water_avg.values, color="#64b5f6", edgecolor="none", width=0.5)
        ax6.set_xticks(range(len(water_avg)))
        ax6.set_xticklabels([w.split("(")[0].strip() for w in water_avg.index], rotation=15, ha='right', color="#a5d6a7", fontsize=8)
        ax6.tick_params(colors="#a5d6a7")
        ax6.set_ylabel("Avg Revenue (₹)", color="#a5d6a7")
        for spine in ax6.spines.values(): spine.set_edgecolor((0.506, 0.780, 0.518, 0.1))
        plt.tight_layout()
        st.pyplot(fig6)

# ═══════════════════════════════════════════════════
# TAB 4 — DATASET
# ═══════════════════════════════════════════════════
with tab4:
    st.markdown("### 🗃️ Training Dataset (Tamil Nadu Farm Records)")
    st.markdown(f"**{len(df_data)} synthetic records** generated based on real TN agricultural patterns.")
    st.dataframe(df_data.head(30), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Crop Distribution**")
        st.bar_chart(df_data["Crop"].value_counts().rename(lambda x: x.split("(")[0].strip()))
    with col_b:
        st.markdown("**District Distribution**")
        st.bar_chart(df_data["District"].value_counts())

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#4caf50; font-size:0.85rem; padding:1rem 0; font-family:'Source Sans 3',sans-serif; letter-spacing:0.05em;">
  🌾 Built for Tamil Nadu Farmers &nbsp;·&nbsp; Python · scikit-learn · Streamlit &nbsp;·&nbsp; MCA Project &nbsp;·&nbsp; 2024
</div>
""", unsafe_allow_html=True)
