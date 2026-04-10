import streamlit as st
import sys, os
from datetime import date

st.set_page_config(page_title="About – ChurnShield", page_icon="ℹ️", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <span class='page-title'>ℹ️ About ChurnShield</span>
</div>
<p class='page-subtitle'>Project documentation, dataset details and technical overview</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Project overview ──────────────────────────────────────────────
st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#1a1d24; border:1px solid #2d3748; border-radius:8px; padding:20px;'>
    <p style='color:#e2e8f0; font-size:0.92em; line-height:1.8; margin:0;'>
        <b style='color:#f1f5f9;'>ChurnShield</b> is an end-to-end E-Commerce Customer
        Retention Intelligence Platform built as a Final Year Project. It uses machine
        learning to predict which customers are likely to leave a platform, estimates
        their lifetime value, segments them into behaviour groups, and automatically
        generates personalised retention offers — all through an interactive web dashboard.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

# ── Pages guide ───────────────────────────────────────────────────
st.markdown("<div class='section-title'>Pages Guide</div>", unsafe_allow_html=True)

pages = [
    ("📊","Dashboard",
     "Live overview of your entire customer base. Shows churn distribution, customer segments, CLV by segment, churn rate by location and satisfaction score. Supports uploading any custom CSV to update all charts dynamically.",
     "#3b82f6"),
    ("👤","Single Customer",
     "Enter any individual customer's details to get an instant churn prediction, probability score, CLV estimate, segment classification and a personalised retention offer tailored to their profile.",
     "#a78bfa"),
    ("📂","Bulk Analysis",
     "Upload any CSV file with customer data. ChurnShield automatically handles missing columns using smart defaults and runs predictions for all customers at once. Download full reports or at-risk only reports.",
     "#22c55e"),
    ("🎯","Retention Engine",
     "Shows all at-risk customers ranked by revenue at risk. Each customer gets an urgency badge (Critical/High/Medium/Low) and a personalised offer card. Fully filterable by segment, churn probability and CLV.",
     "#f59e0b"),
    ("🎮","Simulation",
     "What-if analysis tool. Set a customer's base profile and simulate business interventions like giving a discount or improving satisfaction. See live churn probability change and a sensitivity analysis showing which action reduces churn the most.",
     "#f472b6"),
    ("📈","Model Performance",
     "Shows the churn prediction model's evaluation metrics — Accuracy, Precision, Recall, F1 Score and ROC-AUC. Includes a confusion matrix and feature importance chart showing which factors influence churn predictions the most.",
     "#ef4444"),
]

for icon,title,desc,color in pages:
    st.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748;
                border-left:3px solid {color}; border-radius:8px;
                padding:16px 18px; margin-bottom:10px;'>
        <div style='font-size:0.95em; font-weight:700; color:{color}; margin-bottom:6px;'>
            {icon} {title}
        </div>
        <div style='font-size:0.85em; color:#94a3b8; line-height:1.6;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Dataset + Models side by side ────────────────────────────────
d1, d2 = st.columns(2)

with d1:
    st.markdown("<div class='section-title'>Dataset Details</div>", unsafe_allow_html=True)
    dataset_info = [
        ("Total Records",       "5,000 customers"),
        ("Features",            "18 input features + 1 target"),
        ("Target Variable",     "ChurnRisk (0 = Return, 1 = Churn)"),
        ("Class Balance",       "50% churn / 50% return"),
        ("Data Type",           "Synthetic — generated for training"),
        ("Locations",           "10 Indian cities"),
        ("Categories",          "8 product categories"),
        ("Numeric Features",    "Age, Spent, Frequency, Sessions..."),
        ("Categorical Features","Gender, Location, Category, App Usage"),
    ]
    for label, val in dataset_info:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    padding:8px 0; border-bottom:1px solid #2d374855;
                    font-size:0.85em;'>
            <span style='color:#64748b;'>{label}</span>
            <span style='color:#e2e8f0; font-weight:500;'>{val}</span>
        </div>""", unsafe_allow_html=True)

with d2:
    st.markdown("<div class='section-title'>ML Models Used</div>", unsafe_allow_html=True)
    models_info = [
        ("Churn Prediction",    "XGBoost / Random Forest (best auto-selected)"),
        ("Accuracy",            "97.34%"),
        ("ROC-AUC",             "0.9946"),
        ("CLV Prediction",      "Gradient Boosting Regressor"),
        ("Segmentation",        "K-Means Clustering (K=4)"),
        ("Segments",            "Champions · Loyal · At-Risk · Lost"),
        ("RFM Features",        "Recency · Frequency · Monetary"),
        ("Scaler",              "StandardScaler (Z-score normalisation)"),
        ("Encoders",            "LabelEncoder for all categoricals"),
    ]
    for label, val in models_info:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    padding:8px 0; border-bottom:1px solid #2d374855;
                    font-size:0.85em;'>
            <span style='color:#64748b;'>{label}</span>
            <span style='color:#e2e8f0; font-weight:500;'>{val}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Tech Stack</div>", unsafe_allow_html=True)
t1,t2,t3,t4,t5,t6 = st.columns(6)

for col,tech,desc,color in [
    (t1,"Python",      "Core language",        "#3b82f6"),
    (t2,"Streamlit",   "Web dashboard",        "#ef4444"),
    (t3,"XGBoost",     "Churn model",          "#f59e0b"),
    (t4,"Scikit-learn","ML pipeline",          "#22c55e"),
    (t5,"Plotly",      "Interactive charts",   "#a78bfa"),
    (t6,"Pandas",      "Data processing",      "#f472b6"),
]:
    col.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748;
                border-top:3px solid {color}; border-radius:8px;
                padding:14px; text-align:center;'>
        <div style='font-size:0.92em; font-weight:700; color:{color};'>{tech}</div>
        <div style='font-size:0.75em; color:#64748b; margin-top:4px;'>{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── FAQ ───────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Frequently Asked Questions</div>",
            unsafe_allow_html=True)

faqs = [
    ("Why is the dataset synthetic?",
     "Real customer data has strict privacy regulations and is not publicly available. A synthetic dataset with realistic distributions was generated to train and demonstrate the model. The model generalises well to any real e-commerce CSV with the same feature structure."),
    ("Why is accuracy so high (97%)?",
     "The model is trained and tested on synthetic data with clear patterns. In production with real noisy data, accuracy would typically be 80-90%. The high score validates that the model has learned the underlying patterns correctly."),
    ("Can I use this with my own data?",
     "Yes. Use the Bulk Analysis page to upload any CSV. Missing columns are automatically filled with smart defaults so predictions still work even with partial data."),
    ("Why XGBoost?",
     "Both Random Forest and XGBoost were trained and compared. XGBoost uses gradient boosting — it learns from previous mistakes iteratively — making it stronger than Random Forest for this type of tabular classification problem."),
    ("What is CLV?",
     "Customer Lifetime Value is the total revenue a business can expect from a customer over their entire relationship. It helps prioritise which customers are worth spending more to retain."),
    ("What is RFM?",
     "RFM stands for Recency (how recently they bought), Frequency (how often they buy) and Monetary (how much they spend). These three together are the most powerful indicators of customer value and are used for segmentation."),
]

for q, a in faqs:
    with st.expander(f"❓ {q}"):
        st.markdown(f"<p style='color:#94a3b8; font-size:0.88em; line-height:1.7;'>{a}</p>",
                    unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<p style='text-align:center; color:#334155; font-size:0.78em; margin:8px 0;'>
    🛡️ ChurnShield &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp;
    Built with Python · Streamlit · XGBoost · Scikit-learn · Plotly
    &nbsp;·&nbsp; {date.today().strftime("%B %Y")}
</p>
""", unsafe_allow_html=True)
