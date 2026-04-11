"""
retrain.py — Regenerates all models from scratch.
Run this once locally with your cloud Python version,
then re-upload the models/ folder to GitHub.
OR place this in root and it auto-runs on Streamlit Cloud startup.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, "data", "churnshield_dataset.csv")
MODELS_PATH = os.path.join(BASE, "models")
os.makedirs(MODELS_PATH, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ── Encode categoricals ───────────────────────────────────────────
le_gender   = LabelEncoder()
le_location = LabelEncoder()
le_category = LabelEncoder()
le_app      = LabelEncoder()

df['Gender_enc']   = le_gender.fit_transform(df['Gender'])
df['Location_enc'] = le_location.fit_transform(df['Location'])
df['Category_enc'] = le_category.fit_transform(df['PreferredCategory'])
df['AppUsage_enc'] = le_app.fit_transform(df['AppUsage'])

joblib.dump(le_gender,   os.path.join(MODELS_PATH, 'le_gender.pkl'))
joblib.dump(le_location, os.path.join(MODELS_PATH, 'le_location.pkl'))
joblib.dump(le_category, os.path.join(MODELS_PATH, 'le_category.pkl'))
joblib.dump(le_app,      os.path.join(MODELS_PATH, 'le_app.pkl'))
print("✅ Encoders saved")

# ── Churn model ───────────────────────────────────────────────────
FEAT_COLS = [
    'Age','MembershipPeriod','VisitFrequency','AvgSessionTime',
    'PagesViewed','PurchaseFrequency','AvgOrderValue','TotalSpent',
    'LastPurchaseDaysAgo','DiscountUsed','CustomerSatisfaction',
    'ReturnCount','ComplaintHistory','CartAbandonRate',
    'Gender_enc','Location_enc','Category_enc','AppUsage_enc'
]

X = df[FEAT_COLS]
y = df['ChurnRisk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler,     os.path.join(MODELS_PATH, 'scaler.pkl'))
joblib.dump(FEAT_COLS,  os.path.join(MODELS_PATH, 'feature_columns.pkl'))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                     eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                              min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

best_model = xgb if xgb_auc >= rf_auc else rf
print(f"✅ Best model: {'XGBoost' if xgb_auc >= rf_auc else 'Random Forest'} (AUC: {max(xgb_auc, rf_auc):.4f})")
joblib.dump(best_model, os.path.join(MODELS_PATH, 'churn_model.pkl'))

# ── CLV model ─────────────────────────────────────────────────────
CLV_FEAT = ['Age','MembershipPeriod','PurchaseFrequency','AvgOrderValue',
            'TotalSpent','CustomerSatisfaction','ReturnCount',
            'VisitFrequency','DiscountUsed']

X_clv = df[CLV_FEAT]
y_clv = df['CLV']

clv_scaler   = StandardScaler()
X_clv_scaled = clv_scaler.fit_transform(X_clv)
joblib.dump(clv_scaler, os.path.join(MODELS_PATH, 'clv_scaler.pkl'))
joblib.dump(CLV_FEAT,   os.path.join(MODELS_PATH, 'clv_feature_columns.pkl'))

X_clv_train, X_clv_test, y_clv_train, y_clv_test = train_test_split(
    X_clv_scaled, y_clv, test_size=0.2, random_state=42)

print("Training CLV model...")
clv_model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                       learning_rate=0.05, random_state=42)
clv_model.fit(X_clv_train, y_clv_train)
joblib.dump(clv_model, os.path.join(MODELS_PATH, 'clv_model.pkl'))
print("✅ CLV model saved")

# ── Segmentation ──────────────────────────────────────────────────
rfm_cols  = ['LastPurchaseDaysAgo','PurchaseFrequency','TotalSpent',
             'CustomerSatisfaction','CLV']
rfm_scaler = StandardScaler()
rfm_scaled = rfm_scaler.fit_transform(df[rfm_cols])
joblib.dump(rfm_scaler, os.path.join(MODELS_PATH, 'rfm_scaler.pkl'))

print("Training K-Means...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)
joblib.dump(kmeans, os.path.join(MODELS_PATH, 'kmeans_model.pkl'))

cluster_summary = df.groupby('Cluster').agg(
    Avg_CLV=('CLV','mean'), Churn_Rate=('ChurnRisk','mean')
).round(2)

clv_order   = cluster_summary['Avg_CLV'].rank(ascending=False)
churn_order = cluster_summary['Churn_Rate'].rank(ascending=True)
combined    = clv_order + churn_order
sorted_c    = combined.sort_values().index.tolist()
seg_labels  = ['Champions','Loyal','At-Risk','Lost']
segment_map = {c: l for c, l in zip(sorted_c, seg_labels)}
joblib.dump(segment_map, os.path.join(MODELS_PATH, 'segment_map.pkl'))
print(f"✅ Segmentation saved. Map: {segment_map}")

print("\n✅ All models retrained and saved successfully!")
print(f"   Models saved to: {MODELS_PATH}")
