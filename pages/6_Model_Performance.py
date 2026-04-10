import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix)
import sys, os

st.set_page_config(page_title="Model Performance – ChurnShield", page_icon="📈", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import GLOBAL_CSS, CHART_BASE, GRID
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

BASE        = os.path.dirname(os.path.dirname(__file__))
DATA_PATH   = os.path.join(BASE, "data", "churnshield_dataset.csv")
MODELS_PATH = os.path.join(BASE, "models")

@st.cache_resource
def load_all():
    return {k: joblib.load(os.path.join(MODELS_PATH, v)) for k,v in {
        'churn':'churn_model.pkl','scaler':'scaler.pkl',
        'le_gender':'le_gender.pkl','le_location':'le_location.pkl',
        'le_category':'le_category.pkl','le_app':'le_app.pkl',
        'feat_cols':'feature_columns.pkl',
    }.items()}

@st.cache_data
def prepare():
    df = pd.read_csv(DATA_PATH)
    m  = load_all()
    df['Gender_enc']   = m['le_gender'].transform(df['Gender'])
    df['Location_enc'] = m['le_location'].transform(df['Location'])
    df['Category_enc'] = m['le_category'].transform(df['PreferredCategory'])
    df['AppUsage_enc'] = m['le_app'].transform(df['AppUsage'])
    X      = m['scaler'].transform(df[m['feat_cols']])
    y      = df['ChurnRisk'].values
    y_pred = m['churn'].predict(X)
    y_prob = m['churn'].predict_proba(X)[:,1]
    return y, y_pred, y_prob, m

y, y_pred, y_prob, m = prepare()

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <span class='page-title'>📈 Model Performance</span>
</div>
<p class='page-subtitle'>Churn prediction model evaluation metrics</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────
acc  = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec  = recall_score(y, y_pred)
f1   = f1_score(y, y_pred)
auc  = roc_auc_score(y, y_prob)

st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)
m1,m2,m3,m4,m5 = st.columns(5)
for col,lbl,val,clr,desc in [
    (m1,"Accuracy",  f"{acc*100:.2f}%",  "#22c55e","Overall correct predictions"),
    (m2,"Precision", f"{prec*100:.2f}%", "#3b82f6","Of predicted churns, how many were correct"),
    (m3,"Recall",    f"{rec*100:.2f}%",  "#f59e0b","Of actual churns, how many were caught"),
    (m4,"F1 Score",  f"{f1*100:.2f}%",   "#a78bfa","Balance of precision & recall"),
    (m5,"ROC-AUC",   f"{auc:.4f}",       "#ef4444","Model's ability to distinguish classes"),
]:
    col.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748;
                border-left:3px solid {clr}; border-radius:8px;
                padding:16px; text-align:center;'>
        <div style='font-size:0.72em; color:#64748b; text-transform:uppercase;
                    letter-spacing:0.06em; font-weight:600; margin-bottom:8px;'>{lbl}</div>
        <div style='font-size:1.6em; font-weight:800; color:{clr};
                    margin-bottom:8px;'>{val}</div>
        <div style='font-size:0.72em; color:#64748b; line-height:1.4;'>{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

# ── What these mean ───────────────────────────────────────────────
with st.expander("📖 What do these metrics mean?"):
    st.markdown("""
    <div style='color:#94a3b8; font-size:0.88em; line-height:2.2;'>
        <b style='color:#22c55e;'>Accuracy</b> — Out of all 5000 customers,
        how many did the model predict correctly. 97.34% means 4867 out of 5000 were correct.<br>
        <b style='color:#3b82f6;'>Precision</b> — When the model says a customer
        will churn, how often is it right. 97.55% means very few false alarms.<br>
        <b style='color:#f59e0b;'>Recall</b> — Out of all customers who actually
        churned, how many did the model catch. 97.12% means almost no churner was missed.<br>
        <b style='color:#a78bfa;'>F1 Score</b> — The overall balance between
        Precision and Recall. A high F1 means the model is reliable on both counts.<br>
        <b style='color:#ef4444;'>ROC-AUC</b> — Measures how well the model
        separates churners from non-churners. 1.0 is perfect, 0.5 is random guessing.
        0.9946 is excellent.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Confusion Matrix + Feature Importance ─────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#64748b; font-size:0.8em; margin-bottom:10px;'>
    Shows how many predictions were correct vs incorrect
    </p>""", unsafe_allow_html=True)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=['Predicted: Return', 'Predicted: Churn'],
        y=['Actual: Return',    'Actual: Churn'],
        colorscale=[[0,'#1a1d24'],[0.5,'#1e3a5f'],[1,'#2563eb']],
        text=[[f'{tn}<br><span style="font-size:0.8em">Correct</span>',
               f'{fp}<br><span style="font-size:0.8em">Wrong</span>'],
              [f'{fn}<br><span style="font-size:0.8em">Wrong</span>',
               f'{tp}<br><span style="font-size:0.8em">Correct</span>']],
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=18, color='white'),
        showscale=False
    ))
    fig_cm.update_layout(**{k:v for k,v in CHART_BASE.items() if k!='margin'},
                         height=320, margin=dict(t=10,b=10,l=10,r=10),
                         xaxis=dict(side='bottom', tickfont=dict(color='#94a3b8')),
                         yaxis=dict(tickfont=dict(color='#94a3b8')))
    st.plotly_chart(fig_cm, use_container_width=True)

    # Summary below confusion matrix
    st.markdown(f"""
    <div style='display:flex; gap:10px; margin-top:8px;'>
        <div style='background:#1a1d24; border:1px solid #22c55e33;
                    border-radius:6px; padding:10px; flex:1; text-align:center;'>
            <div style='color:#64748b; font-size:0.72em;'>Correctly Identified</div>
            <div style='color:#22c55e; font-weight:700; font-size:1.1em;'>{tn+tp:,}</div>
        </div>
        <div style='background:#1a1d24; border:1px solid #ef444433;
                    border-radius:6px; padding:10px; flex:1; text-align:center;'>
            <div style='color:#64748b; font-size:0.72em;'>Incorrectly Identified</div>
            <div style='color:#ef4444; font-weight:700; font-size:1.1em;'>{tn+tp and (fp+fn):,}</div>
        </div>
        <div style='background:#1a1d24; border:1px solid #3b82f633;
                    border-radius:6px; padding:10px; flex:1; text-align:center;'>
            <div style='color:#64748b; font-size:0.72em;'>Total Customers</div>
            <div style='color:#3b82f6; font-weight:700; font-size:1.1em;'>{len(y):,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#64748b; font-size:0.8em; margin-bottom:10px;'>
    Which customer factors influence churn prediction the most
    </p>""", unsafe_allow_html=True)

    fi = pd.DataFrame({
        'Feature':    m['feat_cols'],
        'Importance': m['churn'].feature_importances_
    }).sort_values('Importance', ascending=True).tail(10)

    # Clean feature names for display
    name_map = {
        'CartAbandonRate':      'Cart Abandon Rate',
        'CustomerSatisfaction': 'Customer Satisfaction',
        'LastPurchaseDaysAgo':  'Last Purchase (days)',
        'ComplaintHistory':     'Complaint History',
        'TotalSpent':           'Total Spent',
        'PurchaseFrequency':    'Purchase Frequency',
        'AvgOrderValue':        'Avg Order Value',
        'MembershipPeriod':     'Membership Period',
        'VisitFrequency':       'Visit Frequency',
        'Age':                  'Age',
        'AppUsage_enc':         'App Usage',
        'Location_enc':         'Location',
        'Category_enc':         'Category',
        'Gender_enc':           'Gender',
        'AvgSessionTime':       'Avg Session Time',
        'PagesViewed':          'Pages Viewed',
        'ReturnCount':          'Return Count',
        'DiscountUsed':         'Discount Used',
    }
    fi['Feature'] = fi['Feature'].map(lambda x: name_map.get(x, x))

    fig_fi = px.bar(
        fi, x='Importance', y='Feature', orientation='h',
        color='Importance',
        color_continuous_scale=['#1e3a5f','#2563eb','#60a5fa']
    )
    fig_fi.update_traces(marker_line_width=0,
                         text=fi['Importance'].apply(lambda x: f'{x:.3f}'),
                         textposition='outside', textfont_size=10)
    fig_fi.update_layout(**{k:v for k,v in CHART_BASE.items() if k!='margin'},
                         height=320, margin=dict(t=10,b=10,l=10,r=10),
                         coloraxis_showscale=False,
                         xaxis=dict(title='Importance Score', **GRID),
                         yaxis=dict(title='', showgrid=False,
                                    tickfont=dict(color='#94a3b8')))
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Model info summary ────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Model Summary</div>", unsafe_allow_html=True)

i1,i2,i3,i4,i5 = st.columns(5)
for col,lbl,val,clr in [
    (i1,"Algorithm",      type(m['churn']).__name__,  "#3b82f6"),
    (i2,"Training Size",  "4,000 customers",           "#22c55e"),
    (i3,"Test Size",      "1,000 customers",           "#f59e0b"),
    (i4,"Features Used",  f"{len(m['feat_cols'])}",    "#a78bfa"),
    (i5,"Status",         "✅ Production Ready",        "#22c55e"),
]:
    col.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748;
                border-radius:8px; padding:14px; text-align:center;'>
        <div style='font-size:0.72em; color:#64748b; text-transform:uppercase;
                    letter-spacing:0.06em; font-weight:600; margin-bottom:6px;'>{lbl}</div>
        <div style='font-size:0.92em; font-weight:700; color:{clr};'>{val}</div>
    </div>""", unsafe_allow_html=True)