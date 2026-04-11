import streamlit as st
import sys
import os

st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Auto-retrain if models are incompatible ───────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE, "models")
FLAG_FILE   = os.path.join(MODELS_PATH, ".trained")

def models_need_retraining():
    if not os.path.exists(FLAG_FILE):
        return True
    try:
        import joblib
        import sklearn
        flag_version = open(FLAG_FILE).read().strip()
        return flag_version != sklearn.__version__
    except:
        return True

if models_need_retraining():
    with st.spinner("🔄 Setting up models for first run — please wait 1-2 minutes..."):
        try:
            import retrain
            import sklearn
            os.makedirs(MODELS_PATH, exist_ok=True)
            open(FLAG_FILE, 'w').write(sklearn.__version__)
            st.success("✅ Models ready!")
            st.rerun()
        except Exception as e:
            st.error(f"Setup failed: {e}")
            st.stop()

sys.path.insert(0, BASE)
from theme import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:2px;'>
    <span style='font-size:1.6em; font-weight:800; color:#f1f5f9;'>🛡️ ChurnShield</span>
</div>
<p style='font-size:0.85em; color:#64748b; margin-top:0; margin-bottom:0;'>
    E-Commerce Customer Retention Intelligence Platform
</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────
st.markdown("<div class='section-title'>Modules</div>", unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)

cards = [
    ("📊","Dashboard",        "Live overview of customers & churn risk",           "#3b82f6"),
    ("👤","Single Customer",  "Predict churn & get personalised retention offer",  "#a78bfa"),
    ("📂","Bulk Analysis",    "Upload CSV — predict churn for all customers",      "#22c55e"),
    ("🎯","Retention Engine", "Auto-ranked at-risk customers with smart offers",   "#f59e0b"),
    ("🎮","Simulation",       "What-if sliders — see how actions reduce churn",    "#f472b6"),
]

for col,(icon,title,desc,color) in zip([c1,c2,c3,c4,c5], cards):
    col.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748;
                border-top:3px solid {color};
                border-radius:8px; padding:18px 14px;
                text-align:center; height:155px;'>
        <div style='font-size:1.7em; margin-bottom:8px;'>{icon}</div>
        <div style='font-size:0.9em; font-weight:700;
                    color:{color}; margin-bottom:6px;'>{title}</div>
        <div style='font-size:0.78em; color:#64748b; line-height:1.4;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── How to use ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>How to Use</div>", unsafe_allow_html=True)
s1,s2,s3,s4,s5 = st.columns(5)

steps = [
    ("01","Dashboard",        "Start here for a full overview of your customer base",       "#3b82f6"),
    ("02","Single Customer",  "Predict churn risk for any individual customer",             "#a78bfa"),
    ("03","Bulk Analysis",    "Upload your customer CSV for mass predictions",              "#22c55e"),
    ("04","Retention Engine", "See ranked at-risk customers with personalised offers",      "#f59e0b"),
    ("05","Simulation",       "Test what actions reduce a customer's churn probability",    "#f472b6"),
]

for col,(num,title,desc,color) in zip([s1,s2,s3,s4,s5], steps):
    col.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748;
                border-radius:8px; padding:14px;'>
        <div style='font-size:0.68em; color:{color}; font-weight:700;
                    letter-spacing:0.08em; margin-bottom:4px;'>STEP {num}</div>
        <div style='font-size:0.88em; font-weight:600;
                    color:#f1f5f9; margin-bottom:4px;'>{title}</div>
        <div style='font-size:0.78em; color:#64748b; line-height:1.4;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top:24px;'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#2d3748; font-size:0.78em; margin:8px 0;'>
    🛡️ ChurnShield &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp;
    Python · Streamlit · XGBoost · Scikit-learn · Plotly
</p>
""", unsafe_allow_html=True)
