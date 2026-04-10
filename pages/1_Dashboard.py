import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys, os

st.set_page_config(page_title="Dashboard – ChurnShield", page_icon="📊", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import GLOBAL_CSS, CHART_BASE, GRID, SEG_COLORS, GREEN_RED
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

BASE        = os.path.dirname(os.path.dirname(__file__))
DATA_PATH   = os.path.join(BASE, "data", "churnshield_dataset.csv")
MODELS_PATH = os.path.join(BASE, "models")

ALL_COLS = {
    'Gender':'Male','Age':38,'Location':'Mumbai','MembershipPeriod':24,
    'PreferredCategory':'Electronics','VisitFrequency':10,'AvgSessionTime':20.0,
    'PagesViewed':8,'PurchaseFrequency':5.0,'AvgOrderValue':2000,'TotalSpent':25000,
    'LastPurchaseDaysAgo':45,'DiscountUsed':0,'CustomerSatisfaction':3,
    'ReturnCount':3,'ComplaintHistory':1,'AppUsage':'Medium','CartAbandonRate':0.4,
}

@st.cache_data
def load_default(): return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    return {k: joblib.load(os.path.join(MODELS_PATH, v)) for k,v in {
        'churn':'churn_model.pkl','scaler':'scaler.pkl',
        'clv':'clv_model.pkl','clv_scaler':'clv_scaler.pkl',
        'kmeans':'kmeans_model.pkl','rfm_scaler':'rfm_scaler.pkl',
        'segment_map':'segment_map.pkl','le_gender':'le_gender.pkl',
        'le_location':'le_location.pkl','le_category':'le_category.pkl',
        'le_app':'le_app.pkl','feat_cols':'feature_columns.pkl',
        'clv_feat':'clv_feature_columns.pkl'
    }.items()}

m = load_models()

def smart_enrich(df_raw):
    df = df_raw.copy()
    # fill missing cols
    for col, default in ALL_COLS.items():
        if col not in df.columns:
            df[col] = default

    # encode safely
    for col, enc_key in [('Gender','le_gender'),('Location','le_location'),
                          ('PreferredCategory','le_category'),('AppUsage','le_app')]:
        enc   = m[enc_key]
        known = set(enc.classes_)
        df[col] = df[col].astype(str).apply(
            lambda x: x if x in known else list(enc.classes_)[0])

    df['Gender_enc']   = m['le_gender'].transform(df['Gender'])
    df['Location_enc'] = m['le_location'].transform(df['Location'])
    df['Category_enc'] = m['le_category'].transform(df['PreferredCategory'])
    df['AppUsage_enc'] = m['le_app'].transform(df['AppUsage'])

    X = m['scaler'].transform(df[m['feat_cols']])
    df['ChurnProb'] = m['churn'].predict_proba(X)[:,1]
    df['ChurnPred'] = m['churn'].predict(X)

    df['PredCLV'] = m['clv'].predict(m['clv_scaler'].transform(df[m['clv_feat']]))

    clv_vals = df['CLV'].values if 'CLV' in df.columns else df['PredCLV'].values
    rfm = m['rfm_scaler'].transform(np.column_stack([
        df['LastPurchaseDaysAgo'].values, df['PurchaseFrequency'].values,
        df['TotalSpent'].values, df['CustomerSatisfaction'].values, clv_vals
    ]))
    df['Cluster'] = m['kmeans'].predict(rfm)
    df['Segment'] = df['Cluster'].map(m['segment_map'])
    return df

# ── Session state for uploaded data ──────────────────────────────
if 'bulk_data' not in st.session_state:
    st.session_state['bulk_data'] = None
if 'bulk_name' not in st.session_state:
    st.session_state['bulk_name'] = None

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <span class='page-title'>📊 Customer Dashboard</span>
</div>
<p class='page-subtitle'>Live overview of your customer base</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Data source selector ──────────────────────────────────────────
st.markdown("<div class='filter-label'>Data Source</div>", unsafe_allow_html=True)
d1, d2, d3 = st.columns([2, 2, 4])

data_source = d1.selectbox(
    "Source", ["Default Dataset", "Upload CSV"],
    label_visibility="collapsed"
)

if data_source == "Upload CSV":
    uploaded = d2.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
    if uploaded:
        raw = pd.read_csv(uploaded)
        with st.spinner("Enriching uploaded data..."):
            enriched = smart_enrich(raw)
        st.session_state['bulk_data'] = enriched
        st.session_state['bulk_name'] = uploaded.name
        st.success(f"✅ Using: **{uploaded.name}** — {len(raw):,} customers")
    elif st.session_state['bulk_data'] is not None:
        st.info(f"📂 Using previously uploaded: **{st.session_state['bulk_name']}**")
else:
    st.session_state['bulk_data'] = None
    st.session_state['bulk_name'] = None

# ── Load correct data ─────────────────────────────────────────────
if st.session_state['bulk_data'] is not None:
    data      = st.session_state['bulk_data']
    is_custom = True
else:
    with st.spinner("Loading default dataset..."):
        raw_default = load_default()
        data        = smart_enrich(raw_default)
    is_custom = False

st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

# ── Horizontal filters ────────────────────────────────────────────
st.markdown("<div class='filter-label'>Filters</div>", unsafe_allow_html=True)
f1,f2,f3,f4,_ = st.columns([2,2,2,2,4])

loc_opts = ["All"] + sorted(data['Location'].unique().tolist()) \
           if 'Location' in data.columns else ["All"]
cat_opts = ["All"] + sorted(data['PreferredCategory'].unique().tolist()) \
           if 'PreferredCategory' in data.columns else ["All"]
gen_opts = ["All"] + sorted(data['Gender'].unique().tolist()) \
           if 'Gender' in data.columns else ["All"]

sel_loc = f1.selectbox("Location", loc_opts,                                label_visibility="collapsed")
sel_cat = f2.selectbox("Category", cat_opts,                                label_visibility="collapsed")
sel_seg = f3.selectbox("Segment",  ["All","Champions","Loyal","At-Risk","Lost"], label_visibility="collapsed")
sel_gen = f4.selectbox("Gender",   gen_opts,                                label_visibility="collapsed")

filtered = data.copy()
if sel_loc != "All" and 'Location'          in filtered.columns: filtered = filtered[filtered['Location']          == sel_loc]
if sel_cat != "All" and 'PreferredCategory' in filtered.columns: filtered = filtered[filtered['PreferredCategory'] == sel_cat]
if sel_seg != "All":                                              filtered = filtered[filtered['Segment']           == sel_seg]
if sel_gen != "All" and 'Gender'            in filtered.columns: filtered = filtered[filtered['Gender']            == sel_gen]

st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)

# ── Data source badge ─────────────────────────────────────────────
badge_color = "#3b82f6" if is_custom else "#22c55e"
badge_label = f"📂 {st.session_state['bulk_name']}" if is_custom else "📊 Default Dataset (5,000 customers)"
st.markdown(f"""
<div style='background:#1a1d24; border:1px solid {badge_color}33;
            border-left:3px solid {badge_color}; border-radius:6px;
            padding:8px 14px; margin-bottom:14px; font-size:0.82em;
            color:#94a3b8; display:inline-block;'>
    {badge_label} &nbsp;·&nbsp; Showing {len(filtered):,} customers
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────
total       = len(filtered)
churn_count = int(filtered['ChurnPred'].sum())
churn_pct   = churn_count/total*100 if total else 0
rev_risk    = filtered[filtered['ChurnPred']==1]['PredCLV'].sum()
avg_clv     = filtered['PredCLV'].mean()
champions   = len(filtered[filtered['Segment']=='Champions'])

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Customers",  f"{total:,}")
k2.metric("At Churn Risk",    f"{churn_count:,}",       f"{churn_pct:.1f}%")
k3.metric("Revenue at Risk",  f"₹{rev_risk/100000:.1f}L")
k4.metric("Avg Customer CLV", f"₹{avg_clv:,.0f}")
k5.metric("Champions",        f"{champions:,}")

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── Charts row 1 ──────────────────────────────────────────────────
c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("<div class='section-title'>Churn Risk Split</div>", unsafe_allow_html=True)
    cd = filtered['ChurnPred'].value_counts().reset_index()
    cd.columns = ['Status','Count']
    cd['Status'] = cd['Status'].map({0:'Will Return',1:'Will Churn'})
    fig1 = px.pie(cd, names='Status', values='Count', hole=0.58,
                  color='Status',
                  color_discrete_map={'Will Return':'#22c55e','Will Churn':'#ef4444'})
    fig1.update_traces(textinfo='percent+label', textfont_size=11,
                       marker=dict(line=dict(color='#13151a',width=2)))
    fig1.update_layout(**CHART_BASE, height=230, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("<div class='section-title'>Customer Segments</div>", unsafe_allow_html=True)
    sd = filtered['Segment'].value_counts().reset_index()
    sd.columns = ['Segment','Count']
    fig2 = px.pie(sd, names='Segment', values='Count', hole=0.58,
                  color='Segment', color_discrete_map=SEG_COLORS)
    fig2.update_traces(textinfo='percent+label', textfont_size=11,
                       marker=dict(line=dict(color='#13151a',width=2)))
    fig2.update_layout(**CHART_BASE, height=230, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with c3:
    st.markdown("<div class='section-title'>Avg CLV by Segment</div>", unsafe_allow_html=True)
    cs = filtered.groupby('Segment')['PredCLV'].mean().reindex(
        ['Champions','Loyal','At-Risk','Lost']).dropna().reset_index()
    cs.columns = ['Segment','AvgCLV']
    fig3 = px.bar(cs, x='Segment', y='AvgCLV', color='Segment',
                  color_discrete_map=SEG_COLORS,
                  text=cs['AvgCLV'].apply(lambda x: f'₹{x/1000:.0f}K'))
    fig3.update_traces(textposition='outside', textfont_size=10, marker_line_width=0)
    fig3.update_layout(**CHART_BASE, height=230, showlegend=False,
                       xaxis=dict(title='', **GRID), yaxis=dict(title='', **GRID))
    st.plotly_chart(fig3, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────
c4,c5 = st.columns(2)

with c4:
    if 'Location' in filtered.columns:
        st.markdown("<div class='section-title'>Churn Rate by Location</div>",
                    unsafe_allow_html=True)
        lc = filtered.groupby('Location')['ChurnPred'].mean().reset_index()
        lc.columns = ['Location','ChurnRate']
        lc = lc.sort_values('ChurnRate', ascending=True)
        fig4 = px.bar(lc, x='ChurnRate', y='Location', orientation='h',
                      color='ChurnRate', color_continuous_scale=GREEN_RED,
                      text=lc['ChurnRate'].apply(lambda x: f'{x*100:.1f}%'))
        fig4.update_traces(textposition='outside', textfont_size=10, marker_line_width=0)
        fig4.update_layout(**CHART_BASE, height=280, coloraxis_showscale=False,
                           xaxis=dict(title='Churn Rate', **GRID),
                           yaxis=dict(title='', **GRID))
        st.plotly_chart(fig4, use_container_width=True)

with c5:
    if 'CustomerSatisfaction' in filtered.columns:
        st.markdown("<div class='section-title'>Satisfaction vs Churn Rate</div>",
                    unsafe_allow_html=True)
        sc = filtered.groupby('CustomerSatisfaction')['ChurnPred'].mean().reset_index()
        sc.columns = ['Satisfaction','ChurnRate']
        fig5 = px.line(sc, x='Satisfaction', y='ChurnRate', markers=True,
                       color_discrete_sequence=['#3b82f6'])
        fig5.update_traces(line_width=2.5, marker_size=7)
        fig5.update_layout(**CHART_BASE, height=280,
                           xaxis=dict(title='Satisfaction (1–5)', **GRID),
                           yaxis=dict(title='Churn Rate', **GRID))
        st.plotly_chart(fig5, use_container_width=True)

# ── Table + Download ──────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Customer Records</div>", unsafe_allow_html=True)

show = [c for c in ['CustomerID','Gender','Age','Location','PreferredCategory',
                     'TotalSpent','CustomerSatisfaction','Segment','ChurnProb','PredCLV']
        if c in filtered.columns]
disp = filtered[show].copy()
if 'ChurnProb'  in disp: disp['ChurnProb']  = disp['ChurnProb'].apply(lambda x: f"{x*100:.1f}%")
if 'PredCLV'   in disp: disp['PredCLV']    = disp['PredCLV'].apply(lambda x: f"₹{x:,.0f}")
if 'TotalSpent' in disp: disp['TotalSpent'] = disp['TotalSpent'].apply(lambda x: f"₹{x:,.0f}")

dl, info = st.columns([1,4])
with dl:
    st.download_button("⬇️ Download CSV",
                       data=filtered[show].to_csv(index=False).encode(),
                       file_name="dashboard_export.csv", mime="text/csv",
                       use_container_width=True)
with info:
    st.markdown(f"<p style='color:#64748b; font-size:0.8em; margin-top:8px;'>"
                f"Showing top 100 of {total:,} customers</p>",
                unsafe_allow_html=True)

st.dataframe(disp.head(100), use_container_width=True, hide_index=True)

from datetime import datetime
st.markdown(
    f"<p style='color:#2d3748; font-size:0.75em; text-align:right; margin-top:6px;'>"
    f"Last refreshed: {datetime.now().strftime('%d %B %Y, %I:%M %p')}</p>",
    unsafe_allow_html=True
)
