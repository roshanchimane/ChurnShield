import sys
import os

# ensure root path is available for imports on Streamlit Cloud
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

GLOBAL_CSS = """
<style>
.block-container {
    padding-top: 1.2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] {
    background: #13151a;
    color: #e2e8f0;
}
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
[data-testid="stToolbar"]       { display: none !important; }
#MainMenu                        { display: none !important; }
[data-testid="stSidebar"] {
    background: #1a1d24 !important;
    border-right: 1px solid #2d3748 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}
[data-testid="stSidebarNav"] {
    padding-top: 0.5rem !important;
}
[data-testid="stSidebarNav"] ul {
    padding-top: 0 !important;
}
[data-testid="stSidebarNav"] a {
    color: #94a3b8 !important;
    font-size: 0.88em !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    border-radius: 6px !important;
    transition: none !important;
    text-decoration: none !important;
    animation: none !important;
    transform: none !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: #2d3748 !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #2563eb22 !important;
    color: #60a5fa !important;
    font-weight: 600 !important;
    border-left: 3px solid #2563eb !important;
}
[data-testid="stSidebarNav"]::before {
    content: "🛡️  ChurnShield";
    display: block;
    font-size: 1em;
    font-weight: 700;
    color: #f1f5f9;
    padding: 0 12px 12px 12px;
    border-bottom: 1px solid #2d3748;
    margin-bottom: 8px;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] p  { color: #94a3b8 !important; font-size: 0.82em !important; }
[data-testid="metric-container"] {
    background: #1a1d24;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 14px 16px !important;
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.75em !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.55em !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8em !important;
}
div[data-baseweb="select"] > div {
    background: #1a1d24 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-size: 0.88em !important;
}
input[type="number"], .stNumberInput input {
    background: #1a1d24 !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: #2563eb;
    color: white; border: none;
    border-radius: 6px; padding: 8px 22px;
    font-weight: 600; font-size: 0.9em;
}
.stButton > button:hover { background: #1d4ed8; }
.stDownloadButton > button {
    background: #1a1d24; color: #94a3b8;
    border: 1px solid #2d3748; border-radius: 6px;
    font-size: 0.88em; font-weight: 500;
}
.stDownloadButton > button:hover {
    background: #2d3748; color: #e2e8f0;
}
h1,h2,h3,h4 { color: #f1f5f9 !important; font-weight: 700 !important; }
hr { border-color: #2d3748 !important; margin: 10px 0 !important; }
[data-testid="stDataFrame"] {
    border: 1px solid #2d3748;
    border-radius: 8px;
}
[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    font-size: 0.88em !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f1f5f9 !important;
    border-bottom-color: #2563eb !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] {
    background: #1a1d24 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
}
[data-testid="stSlider"] { padding: 0 !important; }
[data-testid="stFileUploader"] {
    background: #1a1d24 !important;
    border: 1px dashed #2d3748 !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.88em !important;
}
* {
    animation-duration: 0s !important;
    animation-delay: 0s !important;
}
.stApp, [data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] li,
[data-testid="stSidebarNav"] span {
    animation: none !important;
    transition: none !important;
    transform: none !important;
}
.filter-label {
    font-size: 0.72em; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 4px; font-weight: 600;
}
.section-title {
    font-size: 0.72em; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 6px; font-weight: 600;
}
.page-title   { font-size: 1.4em; font-weight: 700; color: #f1f5f9; }
.page-subtitle { font-size: 0.85em; color: #64748b; margin-top: 2px; margin-bottom: 0; }
</style>
"""

CHART_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#1a1d24',
    font=dict(color='#94a3b8', size=11),
    margin=dict(t=24, b=24, l=8, r=8),
)

GRID      = dict(showgrid=True, gridcolor='#2d3748', zeroline=False)
NO_GRID   = dict(showgrid=False, zeroline=False)
SEG_COLORS= {'Champions':'#22c55e','Loyal':'#3b82f6','At-Risk':'#f59e0b','Lost':'#ef4444'}
BLUE_SEQ  = ['#1e3a5f','#2563eb','#60a5fa']
GREEN_RED = ['#22c55e','#f59e0b','#ef4444']
