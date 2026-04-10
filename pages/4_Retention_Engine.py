import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sys, os

st.set_page_config(page_title="Retention Engine – ChurnShield", page_icon="🎯", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import GLOBAL_CSS, CHART_BASE, GRID, SEG_COLORS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

BASE        = os.path.dirname(os.path.dirname(__file__))
DATA_PATH   = os.path.join(BASE, "data", "churnshield_dataset.csv")
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

@st.cache_data
def load_and_predict():
    df = pd.read_csv(DATA_PATH)
    m  = load_models()
    df['Gender_enc']   = m['le_gender'].transform(df['Gender'])
    df['Location_enc'] = m['le_location'].transform(df['Location'])
    df['Category_enc'] = m['le_category'].transform(df['PreferredCategory'])
    df['AppUsage_enc'] = m['le_app'].transform(df['AppUsage'])
    X = m['scaler'].transform(df[m['feat_cols']])
    df['ChurnProb'] = m['churn'].predict_proba(X)[:,1]
    df['ChurnPred'] = m['churn'].predict(X)
    df['PredCLV']   = m['clv'].predict(m['clv_scaler'].transform(df[m['clv_feat']]))
    rfm = m['rfm_scaler'].transform(np.column_stack([
        df['LastPurchaseDaysAgo'].values, df['PurchaseFrequency'].values,
        df['TotalSpent'].values, df['CustomerSatisfaction'].values, df['CLV'].values
    ]))
    df['Cluster'] = m['kmeans'].predict(rfm)
    df['Segment'] = df['Cluster'].map(m['segment_map'])
    return df

m    = load_models()
data = load_and_predict()

def get_offer(segment, churn_prob, clv, category):
    if segment=="Lost" and clv>50000:
        return {"tag":"🎁 Premium Win-Back","color":"#ef4444","urgency":"CRITICAL",
                "offers":[f"30% discount on next 3 orders","Free priority shipping 3 months",
                           "VIP membership upgrade",f"Curated {category} deals"]}
    elif segment=="Lost":
        return {"tag":"💌 Win-Back Campaign","color":"#f97316","urgency":"HIGH",
                "offers":["20% discount (7 days)","Free shipping next order",
                           f"New {category} arrivals","200 loyalty points"]}
    elif segment=="At-Risk" and churn_prob>0.7:
        return {"tag":"⚠️ Urgent Retention","color":"#f59e0b","urgency":"HIGH",
                "offers":["25% discount this week","Double loyalty points",
                           "Free express delivery 1 month","Early sale access"]}
    elif segment=="At-Risk":
        return {"tag":"🔔 Gentle Nudge","color":"#f59e0b","urgency":"MEDIUM",
                "offers":["15% off next purchase",f"Top {category} picks",
                           "Loyalty points bonus","48-hr flash sale invite"]}
    elif segment=="Loyal":
        return {"tag":"⭐ Loyalty Reward","color":"#3b82f6","urgency":"LOW",
                "offers":["Free Gold membership 3 months","Member-only discounts",
                           "Early product launch access","Refer & earn ₹500"]}
    else:
        return {"tag":"🏆 Champions Perk","color":"#22c55e","urgency":"LOW",
                "offers":["Champions Club invitation","Personal shopping assistant",
                           "Limited edition early access","Gift voucher ₹2000"]}

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <span class='page-title'>🎯 Retention Engine</span>
</div>
<p class='page-subtitle'>Auto-ranked at-risk customers with personalised retention offers</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Horizontal filters ─────────────────────────────────────────────
st.markdown("<div class='filter-label'>Filters</div>", unsafe_allow_html=True)
f1,f2,f3,f4,f5,_ = st.columns([2,2,2,2,2,2])

seg_filter = f1.multiselect("Segment",   ["Lost","At-Risk","Loyal","Champions"],
                             default=["Lost","At-Risk"], label_visibility="collapsed")
min_prob   = f2.slider("Min Churn %",    0, 100, 50, label_visibility="collapsed")
min_clv    = f3.number_input("Min CLV",  0, 500000, 0, step=5000, label_visibility="collapsed")
sort_by    = f4.selectbox("Sort By",     ["Revenue at Risk","Churn Probability"],
                           label_visibility="collapsed")
top_n      = f5.slider("Top N",          10, 200, 50, label_visibility="collapsed")

at_risk  = data[data['ChurnPred']==1].copy()
filtered = at_risk.copy()
if seg_filter: filtered = filtered[filtered['Segment'].isin(seg_filter)]
filtered = filtered[filtered['ChurnProb'] >= min_prob/100]
filtered = filtered[filtered['PredCLV']   >= min_clv]
sort_col = 'PredCLV' if sort_by=="Revenue at Risk" else 'ChurnProb'
filtered = filtered.sort_values(sort_col, ascending=False).head(top_n)

st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────
total_risk = len(at_risk)
critical   = len(at_risk[at_risk['ChurnProb']>=0.7])
high_val   = len(at_risk[at_risk['PredCLV']>=at_risk['PredCLV'].quantile(0.75)])
rev_risk   = at_risk['PredCLV'].sum()
avg_prob   = at_risk['ChurnProb'].mean()*100

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total At-Risk",      f"{total_risk:,}")
k2.metric("Critical Risk",      f"{critical:,}")
k3.metric("High-Value At-Risk", f"{high_val:,}")
k4.metric("Revenue at Risk",    f"₹{rev_risk/100000:.1f}L")
k5.metric("Avg Churn Prob",     f"{avg_prob:.1f}%")

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────
c1,c2 = st.columns(2)

with c1:
    st.markdown("<div class='section-title'>Revenue at Risk by Segment</div>",
                unsafe_allow_html=True)
    rev_seg = at_risk.groupby('Segment')['PredCLV'].sum().reset_index()
    rev_seg.columns = ['Segment','Revenue']
    rev_seg = rev_seg.sort_values('Revenue', ascending=False)
    fig1 = px.bar(rev_seg, x='Segment', y='Revenue', color='Segment',
                  color_discrete_map=SEG_COLORS,
                  text=rev_seg['Revenue'].apply(lambda x: f'₹{x/1000:.0f}K'))
    fig1.update_traces(textposition='outside', textfont_size=10, marker_line_width=0)
    fig1.update_layout(**CHART_BASE, height=250, showlegend=False,
                       xaxis=dict(title='', **GRID), yaxis=dict(title='Revenue (₹)', **GRID))
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("<div class='section-title'>Churn Probability vs CLV</div>",
                unsafe_allow_html=True)
    fig2 = px.scatter(filtered.head(100), x='ChurnProb', y='PredCLV',
                      color='Segment', color_discrete_map=SEG_COLORS,
                      size='TotalSpent', opacity=0.7,
                      hover_data=['CustomerID','Location','PreferredCategory'])
    fig2.update_layout(**CHART_BASE, height=250,
                       xaxis=dict(title='Churn Probability', **GRID),
                       yaxis=dict(title='Predicted CLV (₹)', **GRID),
                       legend=dict(font=dict(color='#94a3b8')))
    st.plotly_chart(fig2, use_container_width=True)

# ── Customer offer cards ──────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>Top {min(top_n,len(filtered))} At-Risk Customers — Personalised Offers</div>",
            unsafe_allow_html=True)

urgency_color = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#f59e0b","LOW":"#22c55e"}

if len(filtered)==0:
    st.warning("No customers match your filter criteria. Adjust the filters above.")
else:
    rows = [filtered.iloc[i:i+3] for i in range(0, min(len(filtered),30), 3)]
    for row_df in rows:
        cols = st.columns(3)
        for col,(_, cust) in zip(cols, row_df.iterrows()):
            offer = get_offer(cust['Segment'],cust['ChurnProb'],
                              cust['PredCLV'],cust['PreferredCategory'])
            uc  = urgency_color.get(offer['urgency'],'#94a3b8')
            cid = cust.get('CustomerID', f"C{_}")
            col.markdown(f"""
            <div style='background:#1a1d24; border:1px solid #2d3748;
                        border-left:3px solid {offer["color"]};
                        border-radius:8px; padding:16px; margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                    <span style='font-weight:700; color:#f1f5f9; font-size:0.92em;'>{cid}</span>
                    <span style='font-size:0.7em; font-weight:700; color:{uc};
                                 background:{uc}18; padding:2px 8px;
                                 border-radius:12px;'>{offer["urgency"]}</span>
                </div>
                <div style='font-size:0.78em; color:#64748b; margin-bottom:10px;'>
                    {cust["Location"]} · {cust["PreferredCategory"]} · {cust["Segment"]}
                </div>
                <div style='display:flex; gap:8px; margin-bottom:10px;'>
                    <div style='background:#13151a; border-radius:6px; padding:6px 10px;
                                font-size:0.78em; text-align:center; flex:1;'>
                        <div style='color:#64748b;'>Risk</div>
                        <div style='color:{offer["color"]}; font-weight:700;'>{cust["ChurnProb"]*100:.0f}%</div>
                    </div>
                    <div style='background:#13151a; border-radius:6px; padding:6px 10px;
                                font-size:0.78em; text-align:center; flex:1;'>
                        <div style='color:#64748b;'>CLV</div>
                        <div style='color:#3b82f6; font-weight:700;'>₹{cust["PredCLV"]/1000:.0f}K</div>
                    </div>
                    <div style='background:#13151a; border-radius:6px; padding:6px 10px;
                                font-size:0.78em; text-align:center; flex:1;'>
                        <div style='color:#64748b;'>Sat.</div>
                        <div style='color:#f59e0b; font-weight:700;'>{'★'*int(cust["CustomerSatisfaction"])}</div>
                    </div>
                </div>
                <div style='font-size:0.82em; font-weight:700;
                            color:{offer["color"]}; margin-bottom:8px;'>{offer["tag"]}</div>
                {''.join([f"<div style='font-size:0.78em; color:#94a3b8; padding:2px 0;'>· {o}</div>" for o in offer["offers"]])}
            </div>""", unsafe_allow_html=True)

# ── Table + Download ──────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Full At-Risk Customer Table</div>",
            unsafe_allow_html=True)

tbl = filtered.copy()
tbl['ChurnProb'] = tbl['ChurnProb'].apply(lambda x: f"{x*100:.1f}%")
tbl['PredCLV']   = tbl['PredCLV'].apply(lambda x: f"₹{x:,.0f}")
tbl['TotalSpent']= tbl['TotalSpent'].apply(lambda x: f"₹{x:,.0f}")
show = [c for c in ['CustomerID','Age','Gender','Location','PreferredCategory',
                     'TotalSpent','CustomerSatisfaction','ChurnProb','PredCLV','Segment']
        if c in tbl.columns]
st.dataframe(tbl[show], use_container_width=True, hide_index=True)

export = filtered.copy()
export['ChurnProbability'] = (export['ChurnProb']*100).round(2)
export['RetentionOffer']   = export.apply(
    lambda r: get_offer(r['Segment'],r['ChurnProb'],r['PredCLV'],r['PreferredCategory'])['tag'],axis=1)
exp_cols = [c for c in ['CustomerID','Age','Gender','Location','PreferredCategory',
                         'TotalSpent','Segment','ChurnProbability','PredCLV','RetentionOffer']
            if c in export.columns]
st.download_button("⬇️ Download At-Risk Report",
                   data=export[exp_cols].to_csv(index=False).encode(),
                   file_name="retention_report.csv", mime="text/csv")
