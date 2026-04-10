import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import sys, os

st.set_page_config(page_title="Simulation – ChurnShield", page_icon="🎮", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import GLOBAL_CSS, CHART_BASE, GRID
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

def predict_churn(age,membership,visit_freq,avg_session,pages_viewed,
                  purchase_freq,avg_order,total_spent,last_purchase,disc,
                  satisfaction,return_count,complaint_hist,cart_abandon,
                  gender,location,category,app_usage):
    inp = pd.DataFrame([[age,membership,visit_freq,avg_session,pages_viewed,
                         purchase_freq,avg_order,total_spent,last_purchase,disc,
                         satisfaction,return_count,complaint_hist,cart_abandon,
                         m['le_gender'].transform([gender])[0],
                         m['le_location'].transform([location])[0],
                         m['le_category'].transform([category])[0],
                         m['le_app'].transform([app_usage])[0]]],
                        columns=m['feat_cols'])
    return round(m['churn'].predict_proba(m['scaler'].transform(inp))[0][1]*100, 2)

def predict_clv(age,membership,purchase_freq,avg_order,total_spent,
                satisfaction,return_count,visit_freq,disc):
    inp = pd.DataFrame([[age,membership,purchase_freq,avg_order,
                         total_spent,satisfaction,return_count,visit_freq,disc]],
                        columns=m['clv_feat'])
    return round(m['clv'].predict(m['clv_scaler'].transform(inp))[0], 2)

def gauge(prob, title):
    color = "#22c55e" if prob<35 else "#f59e0b" if prob<65 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob,
        number={'suffix':"%",'font':{'size':34,'color':'#f1f5f9'}},
        gauge={'axis':{'range':[0,100],'tickcolor':'#64748b','tickfont':{'color':'#64748b'}},
               'bar':{'color':color,'thickness':0.22},
               'bgcolor':'rgba(0,0,0,0)',
               'steps':[{'range':[0,35],'color':'rgba(34,197,94,0.08)'},
                         {'range':[35,65],'color':'rgba(245,158,11,0.08)'},
                         {'range':[65,100],'color':'rgba(239,68,68,0.08)'}],
               'threshold':{'line':{'color':color,'width':3},'thickness':0.7,'value':prob}},
        title={'text':title,'font':{'color':'#94a3b8','size':12}}
    ))
    base = {k:v for k,v in CHART_BASE.items() if k != 'margin'}
    fig.update_layout(**base, height=230, margin=dict(t=36,b=4,l=10,r=10))
    return fig

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <span class='page-title'>🎮 What-If Simulation</span>
</div>
<p class='page-subtitle'>Adjust customer parameters and instantly see how churn probability changes</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div style='background:#1a1d24; border:1px solid #2d3748; border-left:3px solid #3b82f6;
            border-radius:8px; padding:12px 16px; margin-bottom:18px;
            font-size:0.88em; color:#94a3b8;'>
    💡 <b style='color:#f1f5f9;'>How to use:</b>
    Set the base customer profile on the left →
    Adjust interventions on the right →
    See live churn probability change and sensitivity analysis below.
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2)

# ── Left — Base profile ───────────────────────────────────────────
with left:
    st.markdown("<div class='section-title'>Base Customer Profile</div>", unsafe_allow_html=True)
    r1,r2 = st.columns(2)
    with r1:
        gender        = st.selectbox("Gender",    ["Male","Female"])
        age           = st.number_input("Age",     18, 64, 35)
        location      = st.selectbox("Location",  sorted(m['le_location'].classes_))
        category      = st.selectbox("Category",  sorted(m['le_category'].classes_))
        app_usage     = st.selectbox("App Usage", ["Low","Medium","High"])
        membership    = st.number_input("Membership (months)", 1, 59, 12)
    with r2:
        visit_freq    = st.number_input("Visit Frequency",     1,   29,   5)
        avg_session   = st.number_input("Avg Session (min)",   2.0, 45.0, 12.0, step=0.5)
        pages_viewed  = st.number_input("Pages Viewed",        1,   19,   5)
        purchase_freq = st.number_input("Purchase Freq/month", 0.5, 15.0, 3.0,  step=0.5)
        avg_order     = st.number_input("Avg Order (₹)",       300, 8000, 1500, step=100)
        total_spent   = st.number_input("Total Spent (₹)",     300, 150000,15000,step=1000)

    r3,r4 = st.columns(2)
    with r3:
        last_purchase  = st.number_input("Last Purchase (days ago)", 1, 180, 60)
        disc_base      = st.selectbox("Discount Used",["No","Yes"])
        return_count   = st.number_input("Return Count", 0, 11, 2)
    with r4:
        satisfaction   = st.slider("Satisfaction",      1, 5,   2)
        complaint_hist = st.number_input("Complaints",  0, 4,   2)
        cart_abandon   = st.slider("Cart Abandon Rate", 0.0, 1.0, 0.65, step=0.05)

    disc_val  = 1 if disc_base=="Yes" else 0
    base_prob = predict_churn(age,membership,visit_freq,avg_session,pages_viewed,
                               purchase_freq,avg_order,total_spent,last_purchase,disc_val,
                               satisfaction,return_count,complaint_hist,cart_abandon,
                               gender,location,category,app_usage)
    base_clv  = predict_clv(age,membership,purchase_freq,avg_order,total_spent,
                             satisfaction,return_count,visit_freq,disc_val)

    st.markdown("<div class='section-title' style='margin-top:10px;'>Current Churn Risk</div>",
                unsafe_allow_html=True)
    st.plotly_chart(gauge(base_prob,"Current"), use_container_width=True)
    st.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748; border-radius:6px;
                padding:10px 14px; text-align:center; font-size:0.85em;'>
        <span style='color:#64748b;'>Current CLV: </span>
        <span style='color:#3b82f6; font-weight:700;'>₹{base_clv:,.0f}</span>
    </div>""", unsafe_allow_html=True)

# ── Right — Interventions ─────────────────────────────────────────
with right:
    st.markdown("<div class='section-title'>Simulate Interventions</div>", unsafe_allow_html=True)

    i1,i2 = st.columns(2)
    with i1:
        st.markdown("<div class='filter-label'>Offer Actions</div>", unsafe_allow_html=True)
        new_disc         = st.selectbox("Give Discount?",           ["No","Yes"], index=1, key="sd")
        new_satisfaction = st.slider("Improve Satisfaction to",     1, 5, min(satisfaction+1,5), key="ss")
        new_cart_abandon = st.slider("Reduce Cart Abandon Rate to", 0.0, 1.0,
                                     max(cart_abandon-0.2,0.0), step=0.05, key="sc")
    with i2:
        st.markdown("<div class='filter-label'>Engagement Actions</div>", unsafe_allow_html=True)
        new_visit_freq    = st.number_input("Increase Visit Freq to",    1, 29,  min(visit_freq+3,29),    key="sv")
        new_last_purchase = st.number_input("Days Since Last Purchase",  1, 180, max(last_purchase-15,1), key="sl")
        new_complaints    = st.number_input("Reduce Complaints to",      0, 4,   max(complaint_hist-1,0), key="sco")

    new_disc_val = 1 if new_disc=="Yes" else 0
    sim_prob = predict_churn(age,membership,new_visit_freq,avg_session,pages_viewed,
                              purchase_freq,avg_order,total_spent,new_last_purchase,new_disc_val,
                              new_satisfaction,return_count,new_complaints,new_cart_abandon,
                              gender,location,category,app_usage)
    sim_clv  = predict_clv(age,membership,purchase_freq,avg_order,total_spent,
                            new_satisfaction,return_count,new_visit_freq,new_disc_val)

    st.markdown("<div class='section-title' style='margin-top:10px;'>Churn Risk After Intervention</div>",
                unsafe_allow_html=True)
    st.plotly_chart(gauge(sim_prob,"After Intervention"), use_container_width=True)
    st.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748; border-radius:6px;
                padding:10px 14px; text-align:center; font-size:0.85em;'>
        <span style='color:#64748b;'>Projected CLV: </span>
        <span style='color:#3b82f6; font-weight:700;'>₹{sim_clv:,.0f}</span>
    </div>""", unsafe_allow_html=True)

# ── Impact summary ────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Intervention Impact Summary</div>", unsafe_allow_html=True)

diff      = base_prob - sim_prob
clv_diff  = sim_clv - base_clv
dc        = "#22c55e" if diff>0 else "#ef4444"
cc        = "#22c55e" if clv_diff>=0 else "#ef4444"
sim_color = "#22c55e" if sim_prob<35 else "#f59e0b" if sim_prob<65 else "#ef4444"

s1,s2,s3,s4 = st.columns(4)
for col,lbl,val,clr in [
    (s1,"Before Intervention",f"{base_prob:.1f}%","#ef4444"),
    (s2,"After Intervention", f"{sim_prob:.1f}%", sim_color),
    (s3,"Risk Reduction",     f"{'▼' if diff>0 else '▲'} {abs(diff):.1f}%", dc),
    (s4,"CLV Change",         f"{'+' if clv_diff>=0 else ''}₹{clv_diff:,.0f}", cc),
]:
    col.markdown(f"""
    <div style='background:#1a1d24; border:1px solid #2d3748;
                border-left:3px solid {clr}; border-radius:8px;
                padding:16px; text-align:center;'>
        <div style='font-size:0.72em; color:#64748b; text-transform:uppercase;
                    letter-spacing:0.06em; font-weight:600;'>{lbl}</div>
        <div style='font-size:1.5em; font-weight:700; color:{clr}; margin-top:4px;'>{val}</div>
    </div>""", unsafe_allow_html=True)

# ── Sensitivity chart ─────────────────────────────────────────────
st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Sensitivity Analysis — Impact of Each Action</div>",
            unsafe_allow_html=True)

tests = [
    ("Give Discount",           dict(disc=1)),
    ("Satisfaction → 5",        dict(satisfaction=5)),
    ("Satisfaction → 4",        dict(satisfaction=4)),
    ("Cart Abandon → 0.1",      dict(cart_abandon=0.1)),
    ("Visit Frequency +5",      dict(visit_freq=min(visit_freq+5,29))),
    ("Last Purchase → 7 days",  dict(last_purchase=7)),
    ("Complaints → 0",          dict(complaint_hist=0)),
    ("All Actions Combined",    dict(disc=1,satisfaction=5,cart_abandon=0.1,
                                      visit_freq=min(visit_freq+5,29),
                                      last_purchase=7,complaint_hist=0)),
]
rows = []
for label, ov in tests:
    p = predict_churn(age,membership,ov.get('visit_freq',visit_freq),avg_session,
                       pages_viewed,purchase_freq,avg_order,total_spent,
                       ov.get('last_purchase',last_purchase),ov.get('disc',disc_val),
                       ov.get('satisfaction',satisfaction),return_count,
                       ov.get('complaint_hist',complaint_hist),
                       ov.get('cart_abandon',cart_abandon),
                       gender,location,category,app_usage)
    rows.append({'Action':label,'ChurnProb':p,'Reduction':round(base_prob-p,2)})

sens_df = pd.DataFrame(rows).sort_values('Reduction', ascending=True)
fig_s   = px.bar(sens_df, x='Reduction', y='Action', orientation='h',
                  color='Reduction', color_continuous_scale=['#ef4444','#f59e0b','#22c55e'],
                  text=sens_df['Reduction'].apply(lambda x: f"{x:+.1f}%"))
fig_s.add_vline(x=0, line_dash="dash", line_color="#2d3748")
fig_s.update_traces(textposition='outside', textfont_size=10, marker_line_width=0)
fig_s.update_layout(**CHART_BASE, height=320, coloraxis_showscale=False,
                     xaxis=dict(title='Churn Risk Reduction (%)', **GRID),
                     yaxis=dict(title='', showgrid=False))
st.plotly_chart(fig_s, use_container_width=True)

best = sens_df.iloc[-1]
st.markdown(f"""
<div style='background:#1a1d24; border:1px solid #2d3748;
            border-left:3px solid #22c55e; border-radius:8px;
            padding:14px 18px; font-size:0.88em;'>
    <b style='color:#22c55e;'>Best Action:</b>
    <span style='color:#e2e8f0;'> {best["Action"]}</span>
    <span style='color:#64748b;'> reduces churn risk by </span>
    <b style='color:#22c55e;'>{best["Reduction"]:+.1f}%</b>
    <span style='color:#64748b;'>, bringing it down to </span>
    <b style='color:#f1f5f9;'>{best["ChurnProb"]:.1f}%</b>
</div>""", unsafe_allow_html=True)
