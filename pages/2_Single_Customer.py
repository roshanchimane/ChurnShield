import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys, os

st.set_page_config(page_title="Single Customer – ChurnShield", page_icon="👤", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from theme import GLOBAL_CSS, CHART_BASE, SEG_COLORS
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

def get_offer(segment, churn_prob, clv, category):
    if segment == "Lost" and clv > 50000:
        return {"title":"🎁 Premium Win-Back Offer",
                "offers":[f"30% discount on next 3 orders","Free priority shipping for 3 months",
                          "Exclusive VIP membership upgrade",f"Curated {category} deals"],
                "color":"#ef4444","message":"High-value lost customer. Aggressive win-back recommended."}
    elif segment == "Lost":
        return {"title":"💌 Win-Back Campaign",
                "offers":["20% discount coupon (valid 7 days)","Free shipping on next order",
                          f"New {category} arrivals handpicked","200 reactivation loyalty points"],
                "color":"#f97316","message":"Customer has churned. Send re-engagement email immediately."}
    elif segment == "At-Risk" and churn_prob > 0.7:
        return {"title":"⚠️ Urgent Retention Offer",
                "offers":["25% discount this week","Double loyalty points",
                          "Free express delivery for 1 month","Early access to upcoming sale"],
                "color":"#f59e0b","message":"High churn probability. Immediate intervention needed."}
    elif segment == "At-Risk":
        return {"title":"🔔 Re-Engagement Nudge",
                "offers":["15% off next purchase",f"Top {category} picks for you",
                          "Loyalty points expiry bonus","48-hour flash sale invite"],
                "color":"#f59e0b","message":"Customer is drifting. A gentle nudge should bring them back."}
    elif segment == "Loyal":
        return {"title":"⭐ Loyalty Reward",
                "offers":["Free Gold membership for 3 months","Member-only exclusive discounts",
                          "Early product launch access","Refer & earn ₹500"],
                "color":"#3b82f6","message":"Loyal customer. Reward and strengthen the relationship."}
    else:
        return {"title":"🏆 Champions Club Invite",
                "offers":["Champions Club invitation","Personal shopping assistant",
                          "Limited edition early access","Annual gift voucher ₹2000"],
                "color":"#22c55e","message":"Top-tier customer. Keep them engaged with exclusive perks."}

def gauge(prob):
    color = "#22c55e" if prob < 35 else "#f59e0b" if prob < 65 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={'suffix':"%",'font':{'size':38,'color':'#f1f5f9'}},
        gauge={
            'axis':{'range':[0,100],'tickcolor':'#64748b','tickfont':{'color':'#64748b'}},
            'bar':{'color':color,'thickness':0.22},
            'bgcolor':'rgba(0,0,0,0)',
            'steps':[{'range':[0,35],'color':'rgba(34,197,94,0.1)'},
                     {'range':[35,65],'color':'rgba(245,158,11,0.1)'},
                     {'range':[65,100],'color':'rgba(239,68,68,0.1)'}],
            'threshold':{'line':{'color':color,'width':3},'thickness':0.7,'value':prob}
        },
        title={'text':"Churn Risk Score",'font':{'color':'#94a3b8','size':13}}
    ))
    base = {k:v for k,v in CHART_BASE.items() if k != "margin"}
    fig.update_layout(**base, height=250, margin=dict(t=36,b=4,l=10,r=10))
    return fig

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <span class='page-title'>👤 Single Customer Analysis</span>
</div>
<p class='page-subtitle'>Enter customer details to predict churn risk and get a personalised retention offer</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Customer Details</div>", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown("<div class='filter-label'>Demographics</div>", unsafe_allow_html=True)
    gender    = st.selectbox("Gender",            ["Male","Female"])
    age       = st.number_input("Age",             18, 64, 35)
    location  = st.selectbox("Location",           sorted(m['le_location'].classes_))
    category  = st.selectbox("Preferred Category", sorted(m['le_category'].classes_))
    app_usage = st.selectbox("App Usage",          ["Low","Medium","High"])

with c2:
    st.markdown("<div class='filter-label'>Purchase Behaviour</div>", unsafe_allow_html=True)
    membership    = st.number_input("Membership Period (months)", 1,   59,    12)
    purchase_freq = st.number_input("Purchase Frequency / month", 0.5, 15.0,  3.0, step=0.5)
    avg_order     = st.number_input("Avg Order Value (₹)",        300, 8000, 1500, step=100)
    total_spent   = st.number_input("Total Spent (₹)",            300, 150000,15000,step=1000)
    last_purchase = st.number_input("Last Purchase (days ago)",   1,   180,   45)
    discount_used = st.selectbox("Used Discount?",                ["No","Yes"])

with c3:
    st.markdown("<div class='filter-label'>Engagement & Experience</div>", unsafe_allow_html=True)
    visit_freq     = st.number_input("Visit Frequency / month",   1,   29,   8)
    avg_session    = st.number_input("Avg Session Time (min)",    2.0, 45.0, 12.0, step=0.5)
    pages_viewed   = st.number_input("Pages Viewed per Visit",    1,   19,   5)
    satisfaction   = st.slider("Customer Satisfaction",           1,   5,    2)
    return_count   = st.number_input("Return Count",              0,   11,   2)
    complaint_hist = st.number_input("Complaint History",         0,   4,    2)
    cart_abandon   = st.slider("Cart Abandon Rate",               0.0, 1.0,  0.6, step=0.05)

st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)
_, btn, _ = st.columns([3,1,3])
predict_btn = btn.button("Analyse Customer", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────
if predict_btn:
    disc  = 1 if discount_used=="Yes" else 0
    g_enc = m['le_gender'].transform([gender])[0]
    l_enc = m['le_location'].transform([location])[0]
    c_enc = m['le_category'].transform([category])[0]
    a_enc = m['le_app'].transform([app_usage])[0]

    inp = pd.DataFrame([[age,membership,visit_freq,avg_session,pages_viewed,
                         purchase_freq,avg_order,total_spent,last_purchase,disc,
                         satisfaction,return_count,complaint_hist,cart_abandon,
                         g_enc,l_enc,c_enc,a_enc]], columns=m['feat_cols'])

    churn_prob = m['churn'].predict_proba(m['scaler'].transform(inp))[0][1]
    churn_pred = m['churn'].predict(m['scaler'].transform(inp))[0]

    clv_inp  = pd.DataFrame([[age,membership,purchase_freq,avg_order,
                               total_spent,satisfaction,return_count,visit_freq,disc]],
                              columns=m['clv_feat'])
    pred_clv = m['clv'].predict(m['clv_scaler'].transform(clv_inp))[0]

    rfm_scaled = m['rfm_scaler'].transform(
        [[last_purchase,purchase_freq,total_spent,satisfaction,pred_clv]])
    segment  = m['segment_map'][m['kmeans'].predict(rfm_scaled)[0]]
    offer    = get_offer(segment, churn_prob, pred_clv, category)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)

    risk_color = "#ef4444" if churn_pred==1 else "#22c55e"
    risk_label = "Will Churn" if churn_pred==1 else "Will Return"

    r1,r2,r3,r4 = st.columns(4)
    for col, lbl, val, clr in [
        (r1,"Prediction",        risk_label,              risk_color),
        (r2,"Churn Probability", f"{churn_prob*100:.1f}%","#3b82f6"),
        (r3,"Predicted CLV",     f"₹{pred_clv:,.0f}",    "#a78bfa"),
        (r4,"Segment",           segment,                 SEG_COLORS.get(segment,"#94a3b8")),
    ]:
        col.markdown(f"""
        <div style='background:#1a1d24; border:1px solid #2d3748;
                    border-left:3px solid {clr};
                    border-radius:8px; padding:16px 18px;'>
            <div style='font-size:0.72em; color:#64748b; text-transform:uppercase;
                        letter-spacing:0.06em; font-weight:600;'>{lbl}</div>
            <div style='font-size:1.45em; font-weight:700;
                        color:{clr}; margin-top:4px;'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    g_col, o_col = st.columns(2)

    with g_col:
        st.markdown("<div class='section-title'>Churn Risk Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(gauge(churn_prob*100), use_container_width=True)
        if churn_prob < 0.35:
            st.success("Low Risk — Customer is likely to return.")
        elif churn_prob < 0.65:
            st.warning("Medium Risk — Monitor closely and send a retention offer.")
        else:
            st.error("High Risk — Immediate action required to prevent churn.")

    with o_col:
        st.markdown("<div class='section-title'>Personalised Retention Offer</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#1a1d24; border:1px solid #2d3748;
                    border-left:3px solid {offer["color"]};
                    border-radius:8px; padding:20px; height:240px;'>
            <div style='font-size:1em; font-weight:700;
                        color:{offer["color"]}; margin-bottom:14px;'>
                {offer["title"]}
            </div>
        </div>""", unsafe_allow_html=True)
        for item in offer["offers"]:
            st.markdown(f"""
            <div style='background:#13151a; border-left:2px solid {offer["color"]}44;
                        border-radius:4px; padding:8px 12px; margin-bottom:6px;
                        color:#cbd5e1; font-size:0.88em;'>
                {item}
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-top:10px; padding:10px 12px; background:#1a1d24;
                    border-radius:6px; font-size:0.8em; color:#64748b;
                    border:1px solid #2d3748;'>
            💡 {offer["message"]}
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Customer Profile Summary</div>", unsafe_allow_html=True)
    p1,p2,p3 = st.columns(3)
    for col, title, color, lines in [
        (p1,"Demographics","#3b82f6",
         [f"Gender: {gender}", f"Age: {age}", f"Location: {location}", f"Category: {category}"]),
        (p2,"Purchase Behaviour","#a78bfa",
         [f"Total Spent: ₹{total_spent:,}", f"Purchase Freq: {purchase_freq}/month",
          f"Avg Order: ₹{avg_order:,}", f"Last Purchase: {last_purchase} days ago"]),
        (p3,"Engagement","#22c55e",
         [f"Satisfaction: {'★'*satisfaction}{'☆'*(5-satisfaction)}",
          f"Complaints: {complaint_hist}", f"Cart Abandon: {cart_abandon*100:.0f}%",
          f"App Usage: {app_usage}"]),
    ]:
        col.markdown(f"""
        <div style='background:#1a1d24; border:1px solid #2d3748;
                    border-radius:8px; padding:16px;'>
            <div style='font-size:0.72em; color:{color}; text-transform:uppercase;
                        letter-spacing:0.06em; font-weight:600; margin-bottom:10px;'>{title}</div>
            {''.join([f"<div style='font-size:0.88em; color:#cbd5e1; padding:3px 0; border-bottom:1px solid #2d374844;'>{l}</div>" for l in lines])}
        </div>""", unsafe_allow_html=True)

    # ── Download report ───────────────────────────────────────────
    from datetime import date as dt
    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Download Report</div>", unsafe_allow_html=True)

    import io
    report = pd.DataFrame([{
        'Gender': gender, 'Age': age, 'Location': location,
        'PreferredCategory': category, 'AppUsage': app_usage,
        'MembershipPeriod': membership, 'TotalSpent': total_spent,
        'PurchaseFrequency': purchase_freq, 'AvgOrderValue': avg_order,
        'LastPurchaseDaysAgo': last_purchase, 'CustomerSatisfaction': satisfaction,
        'ComplaintHistory': complaint_hist, 'CartAbandonRate': cart_abandon,
        'ChurnProbability': f"{churn_prob*100:.2f}%",
        'Prediction': 'Will Churn' if churn_pred==1 else 'Will Return',
        'PredictedCLV': f"Rs.{pred_clv:,.0f}",
        'Segment': segment,
        'RetentionOffer': offer['title'],
        'ReportDate': dt.today().strftime("%d %B %Y"),
    }])

    dl1, dl2 = st.columns([1,4])
    dl1.download_button(
        label="⬇️ Download Report",
        data=report.to_csv(index=False).encode('utf-8'),
        file_name="customer_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )
    dl2.markdown(
        f"<p style='color:#64748b; font-size:0.8em; margin-top:8px;'>"
        f"Report generated on {dt.today().strftime('%d %B %Y')}</p>",
        unsafe_allow_html=True
    )
