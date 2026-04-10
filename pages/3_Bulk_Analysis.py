import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sys, os

st.set_page_config(page_title="Bulk Analysis – ChurnShield", page_icon="📂", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import GLOBAL_CSS, CHART_BASE, GRID, SEG_COLORS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

BASE        = os.path.dirname(os.path.dirname(__file__))
MODELS_PATH = os.path.join(BASE, "models")

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

# ── Column definitions ────────────────────────────────────────────
ALL_COLS = {
    'Gender':               ('categorical', 'le_gender',   'Male'),
    'Age':                  ('numeric',     None,          38),
    'Location':             ('categorical', 'le_location', 'Mumbai'),
    'MembershipPeriod':     ('numeric',     None,          24),
    'PreferredCategory':    ('categorical', 'le_category', 'Electronics'),
    'VisitFrequency':       ('numeric',     None,          10),
    'AvgSessionTime':       ('numeric',     None,          20.0),
    'PagesViewed':          ('numeric',     None,          8),
    'PurchaseFrequency':    ('numeric',     None,          5.0),
    'AvgOrderValue':        ('numeric',     None,          2000),
    'TotalSpent':           ('numeric',     None,          25000),
    'LastPurchaseDaysAgo':  ('numeric',     None,          45),
    'DiscountUsed':         ('numeric',     None,          0),
    'CustomerSatisfaction': ('numeric',     None,          3),
    'ReturnCount':          ('numeric',     None,          3),
    'ComplaintHistory':     ('numeric',     None,          1),
    'AppUsage':             ('categorical', 'le_app',      'Medium'),
    'CartAbandonRate':      ('numeric',     None,          0.4),
}

FEAT_COLS = m['feat_cols']   # churn model features
CLV_FEAT  = m['clv_feat']    # clv model features

def get_offer_short(segment, churn_prob, category):
    if segment=="Lost" and churn_prob>0.7:      return "30% discount + Free shipping + VIP upgrade"
    elif segment=="Lost":                        return f"20% coupon + Free shipping + {category} picks"
    elif segment=="At-Risk" and churn_prob>0.7: return "25% off + Double points + Express delivery"
    elif segment=="At-Risk":                     return f"15% off + {category} top picks + Flash sale"
    elif segment=="Loyal":                       return "Gold membership + Exclusive discounts"
    else:                                        return "Champions Club invite + Gift voucher ₹2000"

def smart_predict(df_raw):
    df       = df_raw.copy()
    present  = set(df.columns)
    missing  = [c for c in ALL_COLS if c not in present]
    filled   = []

    # ── Fill missing columns with smart defaults ──────────────────
    for col in missing:
        kind, encoder_key, default = ALL_COLS[col]
        df[col] = default
        filled.append(col)

    # ── Encode categoricals safely ────────────────────────────────
    for col, (kind, encoder_key, default) in ALL_COLS.items():
        if kind == 'categorical' and encoder_key:
            enc        = m[encoder_key]
            known      = set(enc.classes_)
            df[col]    = df[col].astype(str)
            # replace unknown values with the default
            df[col]    = df[col].apply(lambda x: x if x in known else default)
            df[f'{col}_enc'] = enc.transform(df[col])

    # Map encoded col names to match feat_cols
    enc_map = {
        'Gender':            'Gender_enc',
        'Location':          'Location_enc',
        'PreferredCategory': 'Category_enc',
        'AppUsage':          'AppUsage_enc',
    }
    df['Gender_enc']   = m['le_gender'].transform(df['Gender'])
    df['Location_enc'] = m['le_location'].transform(df['Location'])
    df['Category_enc'] = m['le_category'].transform(df['PreferredCategory'])
    df['AppUsage_enc'] = m['le_app'].transform(df['AppUsage'])

    # ── Churn prediction ──────────────────────────────────────────
    can_predict_churn = all(c in df.columns or
                            c in [v for v in ['Gender_enc','Location_enc',
                                              'Category_enc','AppUsage_enc']]
                            for c in FEAT_COLS)
    if can_predict_churn:
        X = m['scaler'].transform(df[FEAT_COLS])
        df['ChurnProbability'] = (m['churn'].predict_proba(X)[:,1] * 100).round(2)
        df['ChurnRisk']        = m['churn'].predict(X)
        df['ChurnStatus']      = df['ChurnRisk'].map({0:'Will Return',1:'Will Churn'})
    else:
        df['ChurnProbability'] = np.nan
        df['ChurnRisk']        = np.nan
        df['ChurnStatus']      = 'Unknown'

    # ── CLV prediction ────────────────────────────────────────────
    if all(c in df.columns for c in CLV_FEAT):
        df['PredCLV'] = m['clv'].predict(
            m['clv_scaler'].transform(df[CLV_FEAT])
        ).round(2)
    else:
        df['PredCLV'] = np.nan

    # ── Segmentation ──────────────────────────────────────────────
    rfm_cols_needed = ['LastPurchaseDaysAgo','PurchaseFrequency',
                       'TotalSpent','CustomerSatisfaction']
    clv_for_rfm     = df['CLV'].values if 'CLV' in df.columns else (
                      df['PredCLV'].values if df['PredCLV'].notna().all() else None)

    if all(c in df.columns for c in rfm_cols_needed) and clv_for_rfm is not None:
        rfm = m['rfm_scaler'].transform(np.column_stack([
            df['LastPurchaseDaysAgo'].values,
            df['PurchaseFrequency'].values,
            df['TotalSpent'].values,
            df['CustomerSatisfaction'].values,
            clv_for_rfm
        ]))
        df['Cluster'] = m['kmeans'].predict(rfm)
        df['Segment'] = df['Cluster'].map(m['segment_map'])
    else:
        df['Segment'] = 'Unknown'

    # ── Retention offer ───────────────────────────────────────────
    df['RetentionOffer'] = df.apply(
        lambda r: get_offer_short(
            r['Segment'],
            r['ChurnProbability']/100 if pd.notna(r['ChurnProbability']) else 0,
            r.get('PreferredCategory','General')
        ), axis=1
    )

    return df, filled

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <span class='page-title'>📂 Bulk Customer Analysis</span>
</div>
<p class='page-subtitle'>Upload any customer CSV — works even with partial or missing columns</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Column info ───────────────────────────────────────────────────
with st.expander("📋 Supported Columns & Smart Defaults"):
    st.markdown("""
    <p style='color:#94a3b8; font-size:0.88em; margin-bottom:10px;'>
    ChurnShield works with <b>any subset</b> of these columns.
    Missing columns are automatically filled with smart defaults so predictions still work.
    </p>""", unsafe_allow_html=True)

    defaults_df = pd.DataFrame([
        {'Column': col, 'Type': info[0].capitalize(),
         'Default if Missing': str(info[2]),
         'Required for': 'Churn' if col.replace('_enc','') in FEAT_COLS else
                         'CLV' if col in CLV_FEAT else 'Segmentation'}
        for col, info in ALL_COLS.items()
    ])
    st.dataframe(defaults_df, use_container_width=True, hide_index=True)
    st.info("💡 The more columns you provide, the more accurate the predictions.")

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

uploaded = st.file_uploader("📁 Drop your CSV file here", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        present_cols = set(df_raw.columns)
        all_cols_set = set(ALL_COLS.keys())
        missing_cols = all_cols_set - present_cols
        extra_cols   = present_cols - all_cols_set - {'CustomerID','CLV','ChurnRisk'}

        # ── Upload summary ────────────────────────────────────────
        st.success(f"✅ File uploaded — {len(df_raw):,} customers, {len(df_raw.columns)} columns found")

        if missing_cols:
            st.warning(f"⚠️ {len(missing_cols)} columns missing — filled with smart defaults: "
                       f"`{'`, `'.join(sorted(missing_cols))}`")
        if not missing_cols:
            st.info("✅ All columns present — full prediction accuracy!")

        # ── Run predictions ───────────────────────────────────────
        with st.spinner("Running smart predictions..."):
            result, filled = smart_predict(df_raw)

        # ── Prediction quality indicator ──────────────────────────
        completeness = round((1 - len(missing_cols)/len(ALL_COLS))*100)
        q_color      = "#22c55e" if completeness>=80 else "#f59e0b" if completeness>=50 else "#ef4444"
        st.markdown(f"""
        <div style='background:#1a1d24; border:1px solid #2d3748;
                    border-left:3px solid {q_color}; border-radius:8px;
                    padding:12px 16px; margin:10px 0; font-size:0.88em;'>
            <b style='color:{q_color};'>Data Completeness: {completeness}%</b>
            <span style='color:#64748b; margin-left:10px;'>
            {len(all_cols_set)-len(missing_cols)} of {len(all_cols_set)} columns provided ·
            {"Full predictions" if completeness==100 else
             "Partial predictions — some values estimated from defaults"}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Analysis Summary</div>", unsafe_allow_html=True)

        # ── KPIs ──────────────────────────────────────────────────
        total        = len(result)
        churn_known  = result[result['ChurnStatus'] != 'Unknown']
        churn_count  = int(churn_known['ChurnRisk'].sum()) if len(churn_known) else 0
        safe_count   = len(churn_known) - churn_count
        churn_pct    = churn_count/len(churn_known)*100 if len(churn_known) else 0
        rev_risk     = result[result['ChurnRisk']==1]['PredCLV'].sum() \
                       if result['PredCLV'].notna().any() else 0
        avg_clv      = result['PredCLV'].mean() if result['PredCLV'].notna().any() else 0

        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total Customers", f"{total:,}")
        k2.metric("Will Churn",      f"{churn_count:,}", f"{churn_pct:.1f}%")
        k3.metric("Will Return",     f"{safe_count:,}")
        k4.metric("Revenue at Risk", f"₹{rev_risk/1000:.0f}K" if rev_risk else "N/A")
        k5.metric("Avg CLV",         f"₹{avg_clv:,.0f}" if avg_clv else "N/A")

        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────
        has_churn   = result['ChurnStatus'].ne('Unknown').any()
        has_segment = result['Segment'].ne('Unknown').any()
        has_clv     = result['PredCLV'].notna().any()

        chart_cols = [c for c,flag in [
            ('churn',   has_churn),
            ('segment', has_segment),
            ('prob',    has_churn)
        ] if flag]

        if chart_cols:
            cols = st.columns(len(chart_cols))
            idx  = 0

            if has_churn:
                with cols[idx]:
                    st.markdown("<div class='section-title'>Churn Distribution</div>",
                                unsafe_allow_html=True)
                    cd = result['ChurnStatus'].value_counts().reset_index()
                    cd.columns = ['Status','Count']
                    fig1 = px.pie(cd, names='Status', values='Count', hole=0.58,
                                  color='Status',
                                  color_discrete_map={'Will Return':'#22c55e',
                                                      'Will Churn':'#ef4444','Unknown':'#64748b'})
                    fig1.update_traces(textinfo='percent+label', textfont_size=11,
                                       marker=dict(line=dict(color='#13151a',width=2)))
                    fig1.update_layout(**CHART_BASE, height=220, showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)
                idx += 1

            if has_segment:
                with cols[idx]:
                    st.markdown("<div class='section-title'>Customer Segments</div>",
                                unsafe_allow_html=True)
                    sd = result['Segment'].value_counts().reset_index()
                    sd.columns = ['Segment','Count']
                    fig2 = px.pie(sd, names='Segment', values='Count', hole=0.58,
                                  color='Segment', color_discrete_map=SEG_COLORS)
                    fig2.update_traces(textinfo='percent+label', textfont_size=11,
                                       marker=dict(line=dict(color='#13151a',width=2)))
                    fig2.update_layout(**CHART_BASE, height=220, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                idx += 1

            if has_churn and idx < len(cols):
                with cols[idx]:
                    st.markdown("<div class='section-title'>Churn Probability Distribution</div>",
                                unsafe_allow_html=True)
                    valid_prob = result['ChurnProbability'].dropna()
                    fig3 = px.histogram(valid_prob, nbins=20,
                                        color_discrete_sequence=['#3b82f6'],
                                        labels={'value':'Churn Probability (%)'})
                    fig3.update_layout(**CHART_BASE, height=220,
                                       xaxis=dict(title='Churn Probability (%)', **GRID),
                                       yaxis=dict(title='Count', **GRID))
                    st.plotly_chart(fig3, use_container_width=True)

        # ── Top risk ──────────────────────────────────────────────
        if has_churn:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Top 10 Highest Risk Customers</div>",
                        unsafe_allow_html=True)
            top10 = result.dropna(subset=['ChurnProbability']).nlargest(10,'ChurnProbability')
            show  = [c for c in ['CustomerID','Age','Location','PreferredCategory',
                                  'TotalSpent','ChurnProbability','PredCLV',
                                  'Segment','RetentionOffer'] if c in top10.columns]
            disp  = top10[show].copy()
            disp['ChurnProbability'] = disp['ChurnProbability'].apply(lambda x: f"{x:.1f}%")
            if 'PredCLV' in disp and disp['PredCLV'].notna().any():
                disp['PredCLV'] = disp['PredCLV'].apply(
                    lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
            if 'TotalSpent' in disp:
                disp['TotalSpent'] = disp['TotalSpent'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── Full results ──────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Full Prediction Results</div>",
                    unsafe_allow_html=True)

        f1,f2,_ = st.columns([2,2,4])
        filter_seg   = f1.selectbox("Segment",      ["All","Champions","Loyal","At-Risk","Lost","Unknown"])
        filter_churn = f2.selectbox("Churn Status", ["All","Will Churn","Will Return","Unknown"])

        filt = result.copy()
        if filter_seg   != "All": filt = filt[filt['Segment']     == filter_seg]
        if filter_churn != "All": filt = filt[filt['ChurnStatus'] == filter_churn]

        show_all = [c for c in ['CustomerID','Gender','Age','Location','PreferredCategory',
                                 'TotalSpent','CustomerSatisfaction','ChurnProbability',
                                 'ChurnStatus','PredCLV','Segment','RetentionOffer']
                    if c in filt.columns]
        disp_all = filt[show_all].copy()
        if 'ChurnProbability' in disp_all:
            disp_all['ChurnProbability'] = disp_all['ChurnProbability'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        if 'PredCLV' in disp_all:
            disp_all['PredCLV'] = disp_all['PredCLV'].apply(
                lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
        if 'TotalSpent' in disp_all:
            disp_all['TotalSpent'] = disp_all['TotalSpent'].apply(
                lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
        st.dataframe(disp_all, use_container_width=True, hide_index=True)
        st.markdown(f"<p style='color:#64748b; font-size:0.8em;'>"
                    f"Showing {len(filt):,} of {total:,} customers</p>",
                    unsafe_allow_html=True)

        # ── Downloads ─────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Download Reports</div>", unsafe_allow_html=True)

        export_cols = [c for c in ['CustomerID','Gender','Age','Location','PreferredCategory',
                                    'TotalSpent','CustomerSatisfaction','ChurnProbability',
                                    'ChurnStatus','PredCLV','Segment','RetentionOffer']
                       if c in result.columns]
        d1,d2,_ = st.columns([2,2,4])
        d1.download_button("⬇️ Full Report (CSV)",
                           data=result[export_cols].to_csv(index=False).encode(),
                           file_name="churnshield_predictions.csv", mime="text/csv",
                           use_container_width=True)
        if has_churn:
            atrisk = result[result['ChurnRisk']==1][export_cols]
            d2.download_button("⬇️ At-Risk Only (CSV)",
                               data=atrisk.to_csv(index=False).encode(),
                               file_name="churnshield_atrisk.csv", mime="text/csv",
                               use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please check your CSV file and try again.")

else:
    st.markdown("""
    <div style='text-align:center; padding:50px 20px;
                background:#1a1d24; border:1px dashed #2d3748;
                border-radius:10px; margin-top:10px;'>
        <div style='font-size:2.5em; margin-bottom:12px;'>📂</div>
        <div style='font-size:1.05em; font-weight:600; color:#e2e8f0;'>
            Upload any customer CSV
        </div>
        <div style='color:#64748b; margin-top:8px; font-size:0.88em; line-height:1.6;'>
            Works with full or partial datasets<br>
            Missing columns are auto-filled with smart defaults<br>
            The more columns provided, the more accurate the predictions
        </div>
    </div>
    """, unsafe_allow_html=True)
