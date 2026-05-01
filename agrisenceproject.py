import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from io import BytesIO

# ─── Optional imports with graceful fallback ───────────────────────────────────
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil&display=swap');

* { font-family: 'Segoe UI', sans-serif; }

.main { background-color: #f8fdf4; }

.hero-banner {
    background: linear-gradient(135deg, #1a5c2a 0%, #2d8a47 50%, #4caf50 100%);
    padding: 2rem; border-radius: 16px; color: white;
    text-align: center; margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(26,92,42,0.3);
}
.hero-banner h1 { font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.hero-banner p { font-size: 1.1rem; margin: 0.5rem 0 0; opacity: 0.9; }

.metric-card {
    background: white; border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    border-left: 4px solid #2d8a47; text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #1a5c2a; }
.metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
.metric-card .icon { font-size: 1.5rem; }

.crop-card {
    background: white; border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;
}
.crop-card h2 { color: #1a5c2a; font-size: 1.8rem; margin: 0; }

.irrigation-card {
    border-radius: 16px; padding: 1.5rem;
    text-align: center; margin-bottom: 1rem;
}
.water-now { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; }
.water-soon { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; }
.wait { background: linear-gradient(135deg, #059669, #10b981); color: white; }
.irrigation-card h2 { font-size: 2rem; margin: 0; }
.irrigation-card p { margin: 0.5rem 0 0; opacity: 0.9; }

.soil-card {
    background: white; border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 0.8rem;
}
.ph-optimal { border-left: 5px solid #16a34a; }
.ph-acidic { border-left: 5px solid #dc2626; }
.ph-alkaline { border-left: 5px solid #d97706; }

.profit-card {
    background: white; border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08); text-align: center;
    border-top: 4px solid #2d8a47; position: relative;
}
.profit-card .crop-name { font-size: 1.1rem; font-weight: 700; color: #1a5c2a; }
.profit-card .profit-amt { font-size: 1.6rem; font-weight: 800; color: #166534; }
.profit-card .per-acre { font-size: 0.85rem; color: #666; }
.star-badge {
    position: absolute; top: -12px; right: 10px;
    background: #fbbf24; color: #78350f;
    padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700;
}

.explanation-box {
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 10px; padding: 1rem; margin: 1rem 0;
    font-size: 0.95rem; color: #166534;
}
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #1a5c2a;
    margin: 1rem 0 0.5rem; border-bottom: 2px solid #bbf7d0; padding-bottom: 0.3rem;
}
.stProgress > div > div { background-color: #2d8a47 !important; }

.voice-box {
    background: #f0f9ff; border: 1px solid #7dd3fc;
    border-radius: 10px; padding: 1rem; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Crop Images ──────────────────────────────────────────────────────────────
CROP_IMAGES = {
    'Rice':      'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/A_green_ear_of_rice_01.jpg/400px-A_green_ear_of_rice_01.jpg',
    'Wheat':     'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Vehn%C3%A4pelto_6.jpg/400px-Vehn%C3%A4pelto_6.jpg',
    'Sugarcane': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Sugarcane_field.jpg/400px-Sugarcane_field.jpg',
    'Cotton':    'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/CottonPlant.JPG/400px-CottonPlant.JPG',
    'Groundnut': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Peanut_clusters_on_a_white_background.jpg/400px-Peanut_clusters_on_a_white_background.jpg'
}

CROP_EMOJIS = {'Rice': '🌾', 'Wheat': '🌿', 'Sugarcane': '🎋', 'Cotton': '🌸', 'Groundnut': '🥜'}

CROP_INFO = {
    'Rice':      {'season': 'Kharif (Jun-Nov)', 'water': 'High (150-250mm)', 'days': '90-120 days', 'temp': '25-35°C'},
    'Wheat':     {'season': 'Rabi (Nov-Apr)',  'water': 'Moderate (50-150mm)', 'days': '100-130 days', 'temp': '15-25°C'},
    'Sugarcane': {'season': 'Year-round',      'water': 'High (100-250mm)', 'days': '270-365 days', 'temp': '25-35°C'},
    'Cotton':    {'season': 'Kharif (Jun-Oct)', 'water': 'Low-Med (50-150mm)', 'days': '150-180 days', 'temp': '25-35°C'},
    'Groundnut': {'season': 'Kharif (Jun-Sep)', 'water': 'Low (40-120mm)', 'days': '90-120 days', 'temp': '25-35°C'},
}

PROFIT_TABLE = {
    'Rice':      {'Tamil Nadu': 42000, 'Punjab': 48000, 'Maharashtra': 38000, 'Andhra Pradesh': 45000, 'Gujarat': 35000},
    'Wheat':     {'Tamil Nadu': 28000, 'Punjab': 52000, 'Maharashtra': 30000, 'Andhra Pradesh': 27000, 'Gujarat': 32000},
    'Sugarcane': {'Tamil Nadu': 75000, 'Punjab': 60000, 'Maharashtra': 80000, 'Andhra Pradesh': 70000, 'Gujarat': 65000},
    'Cotton':    {'Tamil Nadu': 48000, 'Punjab': 40000, 'Maharashtra': 55000, 'Andhra Pradesh': 50000, 'Gujarat': 58000},
    'Groundnut': {'Tamil Nadu': 35000, 'Punjab': 30000, 'Maharashtra': 36000, 'Andhra Pradesh': 38000, 'Gujarat': 40000},
}

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        if JOBLIB_AVAILABLE and os.path.exists('agrisense_rf_model.pkl') and os.path.exists('agrisense_label_encoder.pkl'):
            model = joblib.load('agrisense_rf_model.pkl')
            le = joblib.load('agrisense_label_encoder.pkl')
            return model, le, True
    except Exception:
        pass
    return None, None, False

model, label_encoder, MODEL_LOADED = load_model()

# ─── Rule-Based Prediction Fallback ──────────────────────────────────────────
def rule_based_predict(temp, humidity, rainfall, soil_moisture, ph, N, P, K):
    moisture_pct = (soil_moisture / 1023.0) * 100
    scores = {}
    # Rice
    scores['Rice'] = (
        max(0, 1 - abs(temp - 30) / 15) * 0.25 +
        max(0, 1 - abs(humidity - 80) / 30) * 0.25 +
        max(0, 1 - abs(rainfall - 200) / 150) * 0.25 +
        max(0, 1 - abs(moisture_pct - 70) / 40) * 0.15 +
        max(0, 1 - abs(ph - 6.0) / 2.0) * 0.10
    )
    # Wheat
    scores['Wheat'] = (
        max(0, 1 - abs(temp - 20) / 15) * 0.25 +
        max(0, 1 - abs(humidity - 60) / 30) * 0.25 +
        max(0, 1 - abs(rainfall - 100) / 100) * 0.25 +
        max(0, 1 - abs(moisture_pct - 50) / 40) * 0.15 +
        max(0, 1 - abs(ph - 6.5) / 2.0) * 0.10
    )
    # Sugarcane
    scores['Sugarcane'] = (
        max(0, 1 - abs(temp - 30) / 10) * 0.20 +
        max(0, 1 - abs(humidity - 75) / 25) * 0.25 +
        max(0, 1 - abs(rainfall - 175) / 120) * 0.25 +
        max(0, 1 - abs(moisture_pct - 65) / 35) * 0.15 +
        max(0, 1 - abs(N - 110) / 60) * 0.15
    )
    # Cotton
    scores['Cotton'] = (
        max(0, 1 - abs(temp - 30) / 10) * 0.25 +
        max(0, 1 - abs(humidity - 65) / 25) * 0.25 +
        max(0, 1 - abs(rainfall - 100) / 80) * 0.25 +
        max(0, 1 - abs(moisture_pct - 45) / 30) * 0.15 +
        max(0, 1 - abs(ph - 7.0) / 2.0) * 0.10
    )
    # Groundnut
    scores['Groundnut'] = (
        max(0, 1 - abs(temp - 30) / 10) * 0.25 +
        max(0, 1 - abs(humidity - 60) / 25) * 0.25 +
        max(0, 1 - abs(rainfall - 80) / 70) * 0.25 +
        max(0, 1 - abs(moisture_pct - 40) / 30) * 0.15 +
        max(0, 1 - abs(ph - 6.5) / 2.0) * 0.10
    )
    total = sum(scores.values()) or 1e-9
    probs = {k: v / total for k, v in scores.items()}
    best = max(probs, key=probs.get)
    return best, probs[best] * 100, probs

def predict_crop(temp, humidity, rainfall, soil_moisture, ph, N, P, K):
    if MODEL_LOADED:
        try:
            features = np.array([[temp, humidity, rainfall, soil_moisture, ph, N, P, K]])
            probs_arr = model.predict_proba(features)[0]
            classes = label_encoder.classes_
            probs = dict(zip(classes, probs_arr))
            best = max(probs, key=probs.get)
            return best, probs[best] * 100, probs
        except Exception:
            pass
    return rule_based_predict(temp, humidity, rainfall, soil_moisture, ph, N, P, K)

def get_explanation(crop, temp, humidity, rainfall, soil_moisture, ph, N, P, K):
    moisture_pct = (soil_moisture / 1023.0) * 100
    optimal = {
        'Rice':      {'temp': '25-35°C', 'reason': 'high moisture and rainfall support paddy cultivation'},
        'Wheat':     {'temp': '15-25°C', 'reason': 'cool temperatures favor grain formation'},
        'Sugarcane': {'temp': '25-35°C', 'reason': 'high humidity and warmth boost sugar content'},
        'Cotton':    {'temp': '25-35°C', 'reason': 'moderate rainfall prevents waterlogging'},
        'Groundnut': {'temp': '25-35°C', 'reason': 'well-drained soil with low-moderate rainfall is ideal'},
    }
    info = optimal.get(crop, {'temp': '25-35°C', 'reason': 'conditions match crop requirements'})
    return (f"{crop} is recommended because temperature {temp}°C suits its optimal range ({info['temp']}), "
            f"humidity {humidity}% and rainfall {rainfall}mm are appropriate, "
            f"soil moisture {moisture_pct:.0f}% is adequate, and {info['reason']}.")

def irrigation_decision(soil_moisture, temp):
    moisture_pct = (soil_moisture / 1023.0) * 100
    if moisture_pct < 25 or temp > 38:
        return "WATER NOW", "High", 15, "water-now"
    elif moisture_pct < 40:
        return "WATER SOON", "Medium", 8, "water-soon"
    else:
        return "WAIT", "Low", 0, "wait"

def safe_image(url, caption="", width=None):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            img_args = {"use_container_width": True}
            if width:
                img_args = {"width": width}
            st.image(BytesIO(r.content), caption=caption, **img_args)
            return True
    except Exception:
        pass
    return False

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgriSense AI")
    st.markdown("---")

    page = st.radio("📋 Navigation", [
        "🏠 Dashboard", "🌱 Crop AI", "🪨 Soil Health",
        "💧 Irrigation", "💰 Profit Calculator"
    ])

    st.markdown("---")
    st.markdown("### 📡 Sensor Inputs")

    temp      = st.slider("🌡️ Temperature (°C)", 10, 50, 32)
    humidity  = st.slider("💧 Humidity (%)", 20, 100, 65)
    rainfall  = st.slider("🌧️ Rainfall (mm)", 0, 300, 120)
    soil_mois = st.slider("🌊 Soil Moisture (0-1023)", 0, 1023, 380)
    soil_ph   = st.slider("⚗️ Soil pH", 4.0, 9.0, 6.5, step=0.1)
    nitrogen  = st.slider("🧪 Nitrogen (kg/ha)", 0, 200, 90)
    phosphorus= st.slider("🧪 Phosphorus (kg/ha)", 0, 150, 55)
    potassium = st.slider("🧪 Potassium (kg/ha)", 0, 200, 50)

    st.markdown("---")
    soil_type = st.selectbox("🪨 Soil Type", ["Loamy", "Sandy", "Clay", "Black", "Laterite"])

    st.markdown("---")
    st.markdown("### 🎤 Voice Input")
    voice_lang = st.selectbox("Language", ["English", "Tamil (தமிழ்)"])

    if st.button("🎤 Speak Now"):
        if not SR_AVAILABLE:
            st.error("Install SpeechRecognition: pip install SpeechRecognition pyaudio")
        else:
            try:
                recognizer = sr.Recognizer()
                with sr.Microphone() as mic:
                    st.info("🎙️ Listening for 5 seconds...")
                    audio = recognizer.listen(mic, timeout=5, phrase_time_limit=5)
                lang_code = "ta-IN" if "Tamil" in voice_lang else "en-IN"
                text = recognizer.recognize_google(audio, language=lang_code)
                st.success(f"Heard: {text}")
                import re
                nums = re.findall(r'\d+(?:\.\d+)?', text)
                if nums and len(nums) >= 1:
                    st.session_state['voice_temp'] = float(nums[0])
                    st.info(f"Set temperature to {nums[0]}°C. Adjust other sliders manually.")
                else:
                    st.warning("Could not extract numbers from speech.")
            except sr.WaitTimeoutError:
                st.warning("No speech detected.")
            except sr.UnknownValueError:
                st.warning("Could not understand audio.")
            except OSError:
                st.error("Microphone not found or not accessible.")
            except Exception as e:
                st.error(f"Voice error: {e}")

    st.markdown("---")
    status_color = "🟢" if MODEL_LOADED else "🟡"
    mode_text = "ML Model Active" if MODEL_LOADED else "Rule-Based Mode"
    st.markdown(f"{status_color} **{mode_text}**")

# ─── Compute predictions ──────────────────────────────────────────────────────
crop, confidence, all_probs = predict_crop(temp, humidity, rainfall, soil_mois, soil_ph, nitrogen, phosphorus, potassium)
explanation = get_explanation(crop, temp, humidity, rainfall, soil_mois, soil_ph, nitrogen, phosphorus, potassium)
irr_decision, irr_urgency, irr_duration, irr_class = irrigation_decision(soil_mois, temp)
moisture_pct = (soil_mois / 1023.0) * 100

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <h1>🌾 AgriSense AI</h1>
        <p>Smart Agriculture Intelligence System — Tamil Nadu Edition</p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="icon">🌡️</div>
            <div class="value">{temp}°C</div>
            <div class="label">Temperature</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="icon">💧</div>
            <div class="value">{humidity}%</div>
            <div class="label">Humidity</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="icon">🌊</div>
            <div class="value">{moisture_pct:.0f}%</div>
            <div class="label">Soil Moisture</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="icon">🌧️</div>
            <div class="value">{rainfall}mm</div>
            <div class="label">Rainfall</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="crop-card">', unsafe_allow_html=True)
        st.markdown(f"### {CROP_EMOJIS.get(crop, '🌱')} Recommended Crop")
        st.markdown(f"<h2>{crop}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Confidence: {confidence:.1f}%**")
        st.progress(confidence / 100)
        img_shown = safe_image(CROP_IMAGES.get(crop, ''), f"{crop} field")
        if not img_shown:
            st.markdown(f"<div style='font-size:5rem;text-align:center'>{CROP_EMOJIS.get(crop,'🌱')}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        irr_label_map = {"water-now": "water-now", "water-soon": "water-soon", "wait": "wait"}
        st.markdown(f"""<div class="irrigation-card {irr_class}">
            <h2>💧 {irr_decision}</h2>
            <p>Urgency: <strong>{irr_urgency}</strong></p>
            <p>Duration: <strong>{irr_duration} minutes</strong></p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="soil-card">
            <b>🌱 Soil Type:</b> {soil_type}<br>
            <b>⚗️ pH:</b> {soil_ph} — {"🟢 Optimal" if 6.0 <= soil_ph <= 7.5 else ("🔴 Acidic" if soil_ph < 6.0 else "🟠 Alkaline")}<br>
            <b>🧪 NPK:</b> N:{nitrogen} | P:{phosphorus} | K:{potassium}
        </div>""", unsafe_allow_html=True)

        # Probability mini-chart
        st.markdown("**All Crop Probabilities:**")
        for c_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            pct = prob * 100 if prob <= 1 else prob
            st.progress(pct / 100, text=f"{CROP_EMOJIS.get(c_name,'🌱')} {c_name}: {pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CROP AI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌱 Crop AI":
    st.markdown("## 🌱 Crop AI Recommendation")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown(f"### {CROP_EMOJIS.get(crop, '🌱')} {crop}")
        img_shown = safe_image(CROP_IMAGES.get(crop, ''), f"{crop}")
        if not img_shown:
            st.markdown(f"<div style='font-size:6rem;text-align:center'>{CROP_EMOJIS.get(crop,'🌱')}</div>", unsafe_allow_html=True)

        info = CROP_INFO.get(crop, {})
        st.markdown("---")
        st.markdown("**📅 Growing Conditions**")
        df_info = pd.DataFrame({
            'Parameter': ['Season', 'Water Requirement', 'Days to Harvest', 'Optimal Temperature'],
            'Value': [info.get('season', '-'), info.get('water', '-'), info.get('days', '-'), info.get('temp', '-')]
        })
        st.dataframe(df_info, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 💡 Why This Crop?")
        st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)

        st.markdown("### 📊 All Crop Probabilities")
        for c_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            pct = prob * 100 if prob <= 1 else prob
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**{CROP_EMOJIS.get(c_name,'🌱')} {c_name}**")
                st.progress(pct / 100)
            with col_b:
                st.markdown(f"<br><b>{pct:.1f}%</b>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SOIL HEALTH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🪨 Soil Health":
    st.markdown("## 🪨 Soil Health Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌊 Soil Moisture")
        st.markdown(f"<div class='soil-card'><div style='font-size:2rem;font-weight:800;color:#1a5c2a'>{moisture_pct:.0f}%</div>", unsafe_allow_html=True)
        st.progress(moisture_pct / 100)
        moist_status = "🟢 Well Moisturized" if 30 <= moisture_pct <= 70 else ("🔴 Too Dry — Irrigate" if moisture_pct < 30 else "🔵 Over-Saturated")
        st.markdown(f"**Status:** {moist_status}</div>", unsafe_allow_html=True)

        st.markdown("### ⚗️ Soil pH Status")
        if soil_ph < 6.0:
            ph_class, ph_label, ph_advice = "ph-acidic", "🔴 Acidic", "Add agricultural lime to raise pH"
        elif soil_ph > 7.5:
            ph_class, ph_label, ph_advice = "ph-alkaline", "🟠 Alkaline", "Add elemental sulfur to lower pH"
        else:
            ph_class, ph_label, ph_advice = "ph-optimal", "🟢 Optimal", "pH is perfect for most crops"

        st.markdown(f"""<div class="soil-card {ph_class}">
            <div style='font-size:1.8rem;font-weight:800'>{soil_ph}</div>
            <div style='font-size:1.1rem'>{ph_label}</div>
            <div style='color:#555;margin-top:0.3rem'>💡 {ph_advice}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🧪 NPK Nutrient Levels")

        for nutrient, val, max_val, unit, advice in [
            ("Nitrogen (N)", nitrogen, 200, "kg/ha", "Boosts leaf growth" if nitrogen < 60 else ("Optimal for most crops" if nitrogen <= 120 else "High — reduce chemical fertilizer")),
            ("Phosphorus (P)", phosphorus, 150, "kg/ha", "Improve with bone meal" if phosphorus < 30 else ("Good for root development" if phosphorus <= 80 else "High — use sparingly")),
            ("Potassium (K)", potassium, 200, "kg/ha", "Add potash fertilizer" if potassium < 50 else ("Good for fruit quality" if potassium <= 120 else "High — balance with N and P")),
        ]:
            pct = min(val / max_val, 1.0)
            color = "#dc2626" if pct < 0.3 else ("#16a34a" if pct <= 0.7 else "#d97706")
            st.markdown(f"**{nutrient}: {val} {unit}**")
            st.progress(pct)
            st.caption(f"💡 {advice}")

        st.markdown("### 🪨 Soil Profile")
        soil_profiles = {
            "Loamy": "Best for most crops. Good drainage and nutrient retention.",
            "Sandy": "Quick-draining. Add organic matter to improve water retention.",
            "Clay":  "Heavy, slow-draining. Good for rice. Add sand to improve aeration.",
            "Black": "Rich in calcium and magnesium. Ideal for cotton and sugarcane.",
            "Laterite": "Iron-rich, acidic. Good for tea and cashew. Needs liming."
        }
        st.info(f"**{soil_type}:** {soil_profiles.get(soil_type, 'Good general-purpose soil.')}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: IRRIGATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💧 Irrigation":
    st.markdown("## 💧 Irrigation Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""<div class="irrigation-card {irr_class}">
            <h2>💧 {irr_decision}</h2>
            <p><b>Urgency Level:</b> {irr_urgency}</p>
            <p><b>Recommended Duration:</b> {irr_duration} minutes</p>
            <p>Soil Moisture: {moisture_pct:.0f}% | Temperature: {temp}°C</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 🔧 Manual Pump Control")
        c1, c2 = st.columns(2)
        if c1.button("🟢 Pump ON", use_container_width=True):
            st.success("✅ Pump turned ON (simulated)")
        if c2.button("🔴 Pump OFF", use_container_width=True):
            st.warning("⛔ Pump turned OFF (simulated)")

        st.markdown("### 🌧️ Rainfall Forecast (7-Day Mock)")
        forecast = pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Rain (mm)': [12, 0, 5, 25, 8, 0, 15],
            'Status': ['Light', 'Clear', 'Light', 'Moderate', 'Light', 'Clear', 'Light']
        })
        st.dataframe(forecast, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### 📅 Weekly Watering Schedule")
        schedule = pd.DataFrame({
            'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'Time': ['06:00 AM', '—', '06:00 AM', '—', '06:00 AM', '06:30 AM', '—'],
            'Duration': ['15 min', 'Skip', '12 min', 'Skip', '15 min', '10 min', 'Skip'],
            'Status': ['✅', '⏭️', '✅', '⏭️', '✅', '✅', '⏭️']
        })
        st.dataframe(schedule, hide_index=True, use_container_width=True)

        st.markdown("### 📱 SMS Alerts (Twilio)")
        with st.expander("Configure SMS Alerts"):
            twilio_sid = st.text_input("Twilio Account SID", type="password", placeholder="ACxxxxxxxx")
            twilio_token = st.text_input("Auth Token", type="password", placeholder="Your token")
            phone_number = st.text_input("Your Phone Number", placeholder="+91 9876543210")
            if st.button("💾 Save & Enable Alerts"):
                if twilio_sid and twilio_token and phone_number:
                    st.success("✅ Twilio configured! SMS alerts enabled (simulated).")
                else:
                    st.warning("Fill all fields to enable SMS.")
            st.caption("*SMS will be sent when irrigation is needed urgently.*")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: PROFIT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Profit Calculator":
    st.markdown("## 💰 Crop Profit Calculator")

    col1, col2 = st.columns([1, 2])
    with col1:
        farm_size = st.slider("🌾 Farm Size (acres)", 1, 100, 10)
        region = st.selectbox("📍 Region", ["Tamil Nadu", "Punjab", "Maharashtra", "Andhra Pradesh", "Gujarat"])

    st.markdown(f"**Showing profits for {farm_size} acres in {region}:**")

    profits = {c: PROFIT_TABLE[c][region] * farm_size for c in PROFIT_TABLE}
    best_crop = max(profits, key=profits.get)

    cols = st.columns(5)
    crop_list = list(PROFIT_TABLE.keys())

    for i, c_name in enumerate(crop_list):
        profit = profits[c_name]
        per_acre = PROFIT_TABLE[c_name][region]
        is_best = (c_name == best_crop)

        with cols[i]:
            img_shown = safe_image(CROP_IMAGES.get(c_name, ''), width=150)
            if not img_shown:
                st.markdown(f"<div style='font-size:3rem;text-align:center'>{CROP_EMOJIS.get(c_name,'🌱')}</div>", unsafe_allow_html=True)

            border_color = "#fbbf24" if is_best else "#2d8a47"
            badge = '<span class="star-badge">⭐ BEST</span>' if is_best else ""
            st.markdown(f"""
            <div class="profit-card" style="border-top-color:{border_color}">
                {badge}
                <div class="crop-name">{CROP_EMOJIS.get(c_name,'🌱')} {c_name}</div>
                <div class="profit-amt">₹{profit:,.0f}</div>
                <div class="per-acre">₹{per_acre:,}/acre</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📝 **Note:** Profits are estimates based on average market prices. Actual returns depend on weather, pest control, and market conditions.")

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            x=list(profits.keys()), y=list(profits.values()),
            labels={'x': 'Crop', 'y': 'Total Profit (₹)'},
            title=f"Profit Comparison — {farm_size} acres in {region}",
            color=list(profits.values()),
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
