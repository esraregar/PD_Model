import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="PD Model — Loan Risk Assessor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #f4f7fb; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Input text black — only targets actual input elements, not labels */
input[type="number"], input[type="text"],
div[data-baseweb="input"] input,
div[data-baseweb="select"] input {
    color: #111827 !important;
    background: white !important;
}
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] { color: #111827 !important; }
div[data-baseweb="select"] div[role="option"] { color: #111827 !important; }
div[data-baseweb="popover"] * { color: #111827 !important; background: white !important; }
div[data-baseweb="select"] div[aria-selected] { color: #111827 !important; }

/* Sidebar — keep labels white, fix inputs */
[data-testid="stSidebar"] { background: #0d1b2a; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-baseweb]) { color: white !important; }
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] { color: #111827 !important; background: white !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] div[class*="ValueContainer"] { color: #111827 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] div[class*="singleValue"] { color: #111827 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] svg { fill: #111827 !important; }
[data-testid="stSidebar"] div[data-baseweb="input"] input { color: #111827 !important; }
[data-testid="stSidebar"] .stSlider span { color: white !important; }
[data-testid="stSidebar"] .stSlider div[data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider div[data-testid="stTickBarMax"] { color: #94a3b8 !important; }

.metric-card {
    background: white; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); border-top: 4px solid; text-align: center;
}
.metric-card .val  { font-size: 1.9rem; font-weight: 800; margin-bottom: 2px; }
.metric-card .lbl  { font-size: 0.72rem; color: #6b7280; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em; }
.risk-banner {
    border-radius: 14px; padding: 20px 26px; margin-bottom: 18px;
    display: flex; align-items: center; gap: 18px;
}
.sec-hdr {
    font-size: 0.68rem; font-weight: 700; color: #9ca3af;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; margin-top: 16px;
}
.fchip { border-radius: 8px; padding: 8px 12px; margin-bottom: 7px; font-size: 0.82rem; }
.chat-u {
    background: #1a3c6e; color: white;
    border-radius: 14px 14px 4px 14px; padding: 10px 15px;
    margin: 5px 0 5px 20%; font-size: 0.87rem; line-height:1.5;
}
.chat-a {
    background: white; border: 1px solid #e5e7eb; color: #1a2535;
    border-radius: 14px 14px 14px 4px; padding: 10px 15px;
    margin: 5px 20% 5px 0; font-size: 0.87rem; line-height: 1.6;
}
.stButton>button { border-radius: 10px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = Path(__file__).parent / "pd_model.pkl"
    d = pickle.load(open(path, "rb"))
    return d["model"], d["feature_cols"]

model, feature_cols = load_model()

# Exact income-band quintile boundaries from training data
INCOME_BINS   = [0, 41000, 55000, 70000, 95000, float('inf')]
INCOME_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']


# ── Feature engineering — exact mirror of training pipeline ──────────
def engineer_single(inp: dict) -> pd.DataFrame:
    df = pd.DataFrame([inp])
    df['acc_now_delinq'] = df['acc_now_delinq'].fillna(0)

    grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    df['grade_num']           = df['grade'].map(grade_map).fillna(4)
    df['term_months']         = df['term'].str.extract(r'(\d+)').astype(int)
    df['long_term']           = (df['term_months'] == 60).astype(int)
    df['log_annual_inc']      = np.log1p(df['annual_inc'])
    df['high_rate']           = (df['int_rate'] > 13.65).astype(int)
    df['high_inq']            = (df['inq_last_6mths'] >= 3).astype(int)
    df['long_employment']     = (df['emp_length_int'] >= 5).astype(int)
    df['income_not_verified'] = (df['verification_status'] == 'Not Verified').astype(int)
    df['high_risk_purpose']   = df['purpose'].isin(
        ['small_business','educational','moving']).astype(int)
    df['income_band']         = pd.cut(
        df['annual_inc'], bins=INCOME_BINS, labels=INCOME_LABELS).astype(str)

    for col in ['home_ownership','purpose','verification_status','income_band']:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)

    df['int_dti_risk']  = df['int_rate'] * df['dti']
    df['cr_age_diff']   = df['mths_since_earliest_cr_line'] - df['mths_since_issue_d']
    df['inc_dti_ratio'] = df['annual_inc'] / (df['dti'] + 1)

    drop_cols = ['grade','home_ownership','purpose','verification_status','term','income_band']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for c in feature_cols:
        if c not in df.columns: df[c] = 0
    return df[feature_cols]


# ── Helpers ───────────────────────────────────────────────────────────
def get_tier(p):
    if p < 0.08:  return "LOW RISK",  "#166534", "#dcfce7", "🟢"
    if p < 0.18:  return "MODERATE",  "#92400e", "#fef3c7", "🟡"
    if p < 0.30:  return "ELEVATED",  "#9a3412", "#ffedd5", "🟠"
    return               "HIGH RISK", "#991b1b", "#fee2e2", "🔴"

def run_model(inp):
    df    = engineer_single(inp)
    proba = model.predict_proba(df)[0]
    # proba[0] = P(default=0, i.e. will default)
    # proba[1] = P(good_bad=1, i.e. non-default)
    p_default = float(proba[0])   # class 0 = default
    return {"p_default": p_default, "p_nond": float(proba[1])}

def make_gauge(p):
    color = ("#16a34a" if p<0.08 else "#d97706" if p<0.18
             else "#ea580c" if p<0.30 else "#dc2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p*100, 1),
        number={"suffix":"%","font":{"size":30,"color":color,"family":"DM Sans"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#9ca3af","tickfont":{"size":10}},
            "bar":{"color":color,"thickness":0.28}, "bgcolor":"white",
            "steps":[{"range":[0,8],"color":"#dcfce7"},
                     {"range":[8,18],"color":"#fef3c7"},
                     {"range":[18,30],"color":"#ffedd5"},
                     {"range":[30,100],"color":"#fee2e2"}],
            "threshold":{"line":{"color":color,"width":3},"value":p*100},
        },
        title={"text":"Default Probability","font":{"size":12,"color":"#6b7280"}},
    ))
    fig.update_layout(height=200, margin=dict(t=28,b=0,l=15,r=15),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def make_fi_bar():
    fi = (pd.DataFrame({"feature":feature_cols,"importance":model.feature_importances_})
            .sort_values("importance",ascending=False).head(12))
    fig = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                 orientation="h", color="importance",
                 color_continuous_scale=["#dbeafe","#1a3c6e"])
    fig.update_layout(
        height=290, margin=dict(t=5,b=0,l=0,r=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        showlegend=False, coloraxis_showscale=False,
        xaxis_title="Importance", yaxis_title="",
        font=dict(family="DM Sans",size=10,color="#374151"),
    )
    fig.update_traces(marker_line_width=0)
    return fig

def ask_gemini(messages, inp, result):
    tier, *_ = get_tier(result["p_default"])
    system = f"""You are an expert credit risk analyst for a Probability of Default (PD) model.

Current assessment:
- Default Probability: {result['p_default']*100:.1f}%  |  Risk Tier: {tier}
- Interest Rate: {inp['int_rate']}%  |  DTI: {inp['dti']}%
- Annual Income: ${inp['annual_inc']:,}  |  Grade: {inp['grade']}
- Employment: {inp['emp_length_int']} yrs  |  Term: {inp['term']}
- Inquiries 6mo: {inp['inq_last_6mths']}  |  Delinquent Accounts: {inp['acc_now_delinq']}
- Purpose: {inp['purpose']}  |  Ownership: {inp['home_ownership']}
- Int×DTI Score: {inp['int_rate']*inp['dti']:.1f}

Model: XGBoost | AUC 0.6647 | Accuracy 85.5% | 45 features | SMOTE
Top predictors: int_rate, annual_inc, int_dti_risk, grade_num

Be concise and practical. Max 3 short paragraphs."""

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return ("⚠️ GEMINI_API_KEY not set. "
                "Add it in Streamlit Settings → Secrets → GEMINI_API_KEY.")
    genai.configure(api_key=api_key)
    ai_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", system_instruction=system)
    history = [
        {"role":"user" if m["role"]=="user" else "model",
         "parts":[m["content"]]}
        for m in messages[:-1]
    ]
    chat = ai_model.start_chat(history=history)
    return chat.send_message(messages[-1]["content"]).text


# ── Session state ─────────────────────────────────────────────────────
for k, v in [("result",None),("assessed_inputs",{}),
              ("chat_history",[]),("api_messages",[]),("pending_msg","")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 PD Model")
    st.markdown("*Loan Risk Assessor*")
    st.markdown("---")

    st.markdown("### Loan Details")
    # All sliders have matching number_input for typing float values
    col_s, col_n = st.columns([2, 1])
    int_rate = col_s.slider("Interest Rate (%)", 5.0, 30.0, 13.5, 0.1)
    int_rate = col_n.number_input("rate", 5.0, 30.0, float(int_rate), 0.01,
                                   label_visibility="hidden")

    term    = st.selectbox("Loan Term", ["36 months","60 months"])
    purpose = st.selectbox("Loan Purpose", [
        "debt_consolidation","credit_card","home_improvement","major_purchase",
        "small_business","car","medical","moving","vacation","wedding",
        "educational","house","renewable_energy","other"])
    grade   = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"], index=2)

    st.markdown("### Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", 10000, 9999999, 65000, 1000)

    col_s2, col_n2 = st.columns([2, 1])
    dti = col_s2.slider("Debt-to-Income (%)", 0.0, 40.0, 18.0, 0.1)
    dti = col_n2.number_input("dti", 0.0, 40.0, float(dti), 0.01,
                               label_visibility="hidden")

    col_s3, col_n3 = st.columns([2, 1])
    emp_length = col_s3.slider("Employment (years)", 0, 10, 5)
    emp_length = col_n3.number_input("emp", 0, 10, int(emp_length), 1,
                                      label_visibility="hidden")

    home_own = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"])
    verif    = st.selectbox("Verification Status",
                             ["Source Verified","Verified","Not Verified"])

    st.markdown("### Credit History")
    col_s4, col_n4 = st.columns([2, 1])
    inq_6mths = col_s4.slider("Inquiries (6mo)", 0, 10, 1)
    inq_6mths = col_n4.number_input("inq", 0, 10, int(inq_6mths), 1,
                                     label_visibility="hidden")

    col_s5, col_n5 = st.columns([2, 1])
    mths_issue = col_s5.slider("Months Since Issue", 0, 120, 48)
    mths_issue = col_n5.number_input("mis", 0, 120, int(mths_issue), 1,
                                      label_visibility="hidden")

    col_s6, col_n6 = st.columns([2, 1])
    mths_cr = col_s6.slider("Months Since 1st Credit", 12, 400, 180)
    mths_cr = col_n6.number_input("mcr", 12, 400, int(mths_cr), 1,
                                   label_visibility="hidden")

    col_s7, col_n7 = st.columns([2, 1])
    acc_delinq = col_s7.slider("Delinquent Accounts", 0, 5, 0)
    acc_delinq = col_n7.number_input("del", 0, 5, int(acc_delinq), 1,
                                      label_visibility="hidden")

    st.markdown("---")
    run_btn = st.button("⚡ Run Assessment", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("##### Quick Presets")
    c1, c2, c3 = st.columns(3)
    prime_btn    = c1.button("🟢 Prime", use_container_width=True)
    avg_btn      = c2.button("🟡 Avg",   use_container_width=True)
    highrisk_btn = c3.button("🔴 High",  use_container_width=True)

PRESETS = {
    "prime": dict(int_rate=7.5,annual_inc=120000,dti=8.0,emp_length_int=10,
                  inq_last_6mths=0,mths_since_issue_d=60,mths_since_earliest_cr_line=240,
                  acc_now_delinq=0,grade="A",home_ownership="OWN",purpose="credit_card",
                  verification_status="Verified",term="36 months"),
    "avg":   dict(int_rate=13.5,annual_inc=65000,dti=18.0,emp_length_int=5,
                  inq_last_6mths=1,mths_since_issue_d=48,mths_since_earliest_cr_line=180,
                  acc_now_delinq=0,grade="C",home_ownership="RENT",purpose="debt_consolidation",
                  verification_status="Source Verified",term="36 months"),
    "high":  dict(int_rate=24.0,annual_inc=28000,dti=35.0,emp_length_int=1,
                  inq_last_6mths=4,mths_since_issue_d=12,mths_since_earliest_cr_line=60,
                  acc_now_delinq=1,grade="F",home_ownership="RENT",purpose="small_business",
                  verification_status="Not Verified",term="60 months"),
}

if prime_btn:    st.session_state["preset"] = "prime"; st.rerun()
if avg_btn:      st.session_state["preset"] = "avg";   st.rerun()
if highrisk_btn: st.session_state["preset"] = "high";  st.rerun()

preset = st.session_state.pop("preset", None)
if preset and preset in PRESETS:
    inputs = PRESETS[preset]
else:
    inputs = dict(
        int_rate=float(int_rate), annual_inc=float(annual_inc), dti=float(dti),
        emp_length_int=int(emp_length), inq_last_6mths=int(inq_6mths),
        mths_since_issue_d=int(mths_issue), mths_since_earliest_cr_line=int(mths_cr),
        acc_now_delinq=int(acc_delinq), grade=grade, home_ownership=home_own,
        purpose=purpose, verification_status=verif, term=term,
    )

if run_btn or preset:
    st.session_state["result"]          = run_model(inputs)
    st.session_state["assessed_inputs"] = inputs


# ════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════
st.markdown("# 🏦 Probability of Default — Loan Risk Assessor")
st.markdown("*XGBoost · AUC 0.6647 · 45 Features · SMOTE · Gemini AI*")

result = st.session_state["result"]
inp    = st.session_state["assessed_inputs"] if st.session_state["assessed_inputs"] else inputs

if result is None:
    st.info("👈 Fill in the sidebar and click **⚡ Run Assessment** to score the loan.")
    st.stop()

p_def = result["p_default"]
tier, tc, tb, icon = get_tier(p_def)

# ── Verdict banner ────────────────────────────────────────────
st.markdown(f"""
<div class="risk-banner" style="background:{tb};border:2px solid {tc}33">
    <div style="font-size:2.8rem">{icon}</div>
    <div style="flex:1">
        <div style="font-size:1.9rem;font-weight:800;color:{tc}">{tier}</div>
        <div style="color:#64748b;font-size:0.88rem;margin-top:4px">
            <strong style="color:{tc}">{p_def*100:.2f}%</strong> default probability
            &nbsp;·&nbsp;
            <strong>{result['p_nond']*100:.2f}%</strong> repayment probability
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────
cols = st.columns(5)
for col, val, lbl, color in [
    (cols[0], f"{p_def*100:.1f}%",             "Default Prob.", tc),
    (cols[1], inp["grade"],                     "Loan Grade",   "#1a3c6e"),
    (cols[2], f"{inp['int_rate']}%",            "Interest Rate","#f39c12"),
    (cols[3], f"{inp['dti']}%",                 "DTI Ratio",    "#2e86ab"),
    (cols[4], f"${inp['annual_inc']/1000:.0f}k","Annual Income","#27ae60"),
]:
    col.markdown(
        f'<div class="metric-card" style="border-color:{color}">'
        f'<div class="val" style="color:{color}">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Gauge + Factors ───────────────────────────────────────────
col_g, col_f = st.columns([1, 1.6])

with col_g:
    st.plotly_chart(make_gauge(p_def), use_container_width=True)
    st.markdown('<div class="sec-hdr">Engineered Features</div>', unsafe_allow_html=True)
    ea, eb = st.columns(2)
    ea.metric("Int×DTI",      f"{inp['int_rate']*inp['dti']:.1f}")
    eb.metric("Log Income",   f"{np.log1p(inp['annual_inc']):.3f}")
    ec, ed = st.columns(2)
    ec.metric("Credit Age",   f"{inp['mths_since_earliest_cr_line']-inp['mths_since_issue_d']}mo")
    ed.metric("Inc/DTI",      f"{inp['annual_inc']/(inp['dti']+1):.0f}")

with col_f:
    st.markdown('<div class="sec-hdr">Risk Factor Analysis</div>', unsafe_allow_html=True)
    CHIP = {
        "high":    ("#fee2e2","#991b1b","#dc2626"),
        "low":     ("#dcfce7","#166534","#16a34a"),
        "neutral": ("#f3f4f6","#374151","#9ca3af"),
    }
    factors = [
        ("Interest Rate",  f"{inp['int_rate']}%",
         "high" if inp['int_rate']>18 else "low" if inp['int_rate']<10 else "neutral",
         "Above-average — lender pricing elevated risk" if inp['int_rate']>18
         else "Low rate reflects strong credit" if inp['int_rate']<10 else "Market-rate loan"),
        ("Annual Income",  f"${inp['annual_inc']:,}",
         "low" if inp['annual_inc']>80000 else "high" if inp['annual_inc']<35000 else "neutral",
         "Strong income buffer" if inp['annual_inc']>80000
         else "Limited income headroom" if inp['annual_inc']<35000 else "Average income"),
        ("Debt-to-Income", f"{inp['dti']}%",
         "high" if inp['dti']>25 else "low" if inp['dti']<12 else "neutral",
         "High leverage increases default risk" if inp['dti']>25
         else "Low debt burden supports repayment" if inp['dti']<12 else "Moderate leverage"),
        ("Loan Grade",     inp['grade'],
         "low" if inp['grade'] in ['A','B'] else "high" if inp['grade'] in ['F','G'] else "neutral",
         "Prime — strong credit quality" if inp['grade'] in ['A','B']
         else "Sub-prime — elevated risk" if inp['grade'] in ['F','G'] else "Mid-grade"),
        ("Inquiries (6mo)",str(inp['inq_last_6mths']),
         "high" if inp['inq_last_6mths']>=3 else "low" if inp['inq_last_6mths']==0 else "neutral",
         "Multiple inquiries — possible credit stress" if inp['inq_last_6mths']>=3
         else "No recent credit-seeking"),
        ("Delinquency",    f"{inp['acc_now_delinq']} active",
         "high" if inp['acc_now_delinq']>0 else "low",
         "Active delinquency — strong default signal" if inp['acc_now_delinq']>0
         else "Clean payment record"),
    ]
    for fname, fval, ftype, fdesc in factors:
        bg, tc2, dot = CHIP[ftype]
        st.markdown(f"""
        <div class="fchip" style="background:{bg}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600;color:{tc2}">{fname}</span>
            <span style="font-weight:700;color:{tc2}">
              <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                background:{dot};margin-right:5px;vertical-align:middle"></span>{fval}
            </span>
          </div>
          <div style="font-size:0.77rem;color:{tc2};opacity:0.8;margin-top:2px">{fdesc}</div>
        </div>""", unsafe_allow_html=True)

# ── Feature importance ────────────────────────────────────────
st.markdown('<div class="sec-hdr">Top 12 Feature Importances (XGBoost)</div>',
            unsafe_allow_html=True)
st.plotly_chart(make_fi_bar(), use_container_width=True)

# ── Summary chips ─────────────────────────────────────────────
chips = [f"Grade: {inp['grade']}", f"Rate: {inp['int_rate']}%",
         f"DTI: {inp['dti']}%",   f"Term: {inp['term']}",
         f"Purpose: {inp['purpose'].replace('_',' ')}",
         f"Ownership: {inp['home_ownership']}"]
st.markdown(
    " &nbsp; ".join(
        f'<span style="background:#f1f5f9;padding:3px 9px;border-radius:6px;'
        f'font-size:0.79rem;color:#475569">{c}</span>'
        for c in chips),
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════════
# AI ANALYST — same page below results
# ════════════════════════════════════════════════════════════════
st.markdown("## 🤖 AI Credit Risk Analyst")
st.markdown("*Powered by Gemini 1.5 Flash — ask anything about the result above.*")

# Quick prompts
qcols = st.columns(4)
for i, qp in enumerate([
    "Explain the key risk drivers",
    "How to reduce this borrower's risk?",
    "What does AUC 0.6647 mean?",
    "Compare Grade A vs current grade",
]):
    if qcols[i].button(qp, key=f"qp{i}", use_container_width=True):
        st.session_state["pending_msg"] = qp

st.markdown("<br>", unsafe_allow_html=True)

# Input box
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask the AI analyst",
        value=st.session_state.get("pending_msg",""),
        placeholder="e.g. What's the biggest risk factor? How would a lower DTI help?",
    )
    send = st.form_submit_button("Send ➜", use_container_width=False)

if "pending_msg" in st.session_state:
    st.session_state.pop("pending_msg", None)

if send and user_input.strip():
    st.session_state["chat_history"].append({"role":"user","content":user_input})
    st.session_state["api_messages"].append({"role":"user","content":user_input})

    with st.spinner("Thinking..."):
        try:
            reply = ask_gemini(st.session_state["api_messages"], inp, result)
        except Exception as e:
            reply = f"⚠️ Gemini API error: {e}"

    st.session_state["chat_history"].append({"role":"assistant","content":reply})
    st.session_state["api_messages"].append({"role":"assistant","content":reply})
    st.rerun()

# Display conversation — most recent first so answers are visible immediately
if st.session_state["chat_history"]:
    st.markdown("---")
    st.markdown("**Conversation** *(most recent first)*")
    for msg in reversed(st.session_state["chat_history"]):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-u">👤 {msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-a">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear chat"):
        st.session_state["chat_history"] = []
        st.session_state["api_messages"] = []
        st.rerun()
