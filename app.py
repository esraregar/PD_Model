import streamlit as st
import pandas as pd
import numpy as np
import pickle
from google import genai
from google.genai import types
import plotly.graph_objects as go
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

[data-testid="stSidebar"] { background: #0d1b2a !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h5 { color: white !important; }

/* Inputs always black text */
input[type="number"], input[type="text"] { color: #111827 !important; background: white !important; }
[data-baseweb="select"] [class*="singleValue"] { color: #111827 !important; }
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"] { color: #111827 !important; background: white !important; }
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #94a3b8 !important; }

.mcard {
    background: white; border-radius: 12px; padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07); border-top: 4px solid; text-align: center;
}
.mcard .v { font-size: 1.8rem; font-weight: 800; }
.mcard .l { font-size: 0.7rem; color: #6b7280; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em; }
.rbanner {
    border-radius: 14px; padding: 18px 24px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 16px; border: 2px solid;
}
.shdr { font-size: 0.68rem; font-weight: 700; color: #9ca3af;
    letter-spacing: 0.12em; text-transform: uppercase; margin: 14px 0 7px; }
.fchip { border-radius: 8px; padding: 8px 11px; margin-bottom: 6px; font-size: 0.81rem; }
.chat-u {
    background: #1a3c6e; color: white;
    border-radius: 14px 14px 4px 14px;
    padding: 10px 14px; margin: 5px 0 5px 18%; font-size: 0.87rem; line-height: 1.5;
}
.chat-a {
    background: white; border: 1px solid #e5e7eb; color: #1a2535;
    border-radius: 14px 14px 14px 4px;
    padding: 10px 14px; margin: 5px 18% 5px 0; font-size: 0.87rem; line-height: 1.6;
}
.stButton>button { border-radius: 9px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = Path(__file__).parent / "pd_model.pkl"
    d = pickle.load(open(path, "rb"))
    return d["model"], d["feature_cols"]

model, feature_cols = load_model()

# Exact income-band boundaries from training qcut
INCOME_BINS   = [0, 41000, 55000, 70000, 95000, float("inf")]
INCOME_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]


# ── Feature engineering ───────────────────────────────────────────────
def engineer_single(inp: dict) -> pd.DataFrame:
    df = pd.DataFrame([inp])
    df["acc_now_delinq"] = df["acc_now_delinq"].fillna(0)
    grade_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
    df["grade_num"]           = df["grade"].map(grade_map).fillna(4)
    df["term_months"]         = df["term"].str.extract(r"(\d+)").astype(int)
    df["long_term"]           = (df["term_months"] == 60).astype(int)
    df["log_annual_inc"]      = np.log1p(df["annual_inc"])
    df["high_rate"]           = (df["int_rate"] > 13.65).astype(int)
    df["high_inq"]            = (df["inq_last_6mths"] >= 3).astype(int)
    df["long_employment"]     = (df["emp_length_int"] >= 5).astype(int)
    df["income_not_verified"] = (df["verification_status"] == "Not Verified").astype(int)
    df["high_risk_purpose"]   = df["purpose"].isin(
        ["small_business","educational","moving"]).astype(int)
    df["income_band"]         = pd.cut(
        df["annual_inc"], bins=INCOME_BINS, labels=INCOME_LABELS).astype(str)
    for col in ["home_ownership","purpose","verification_status","income_band"]:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)
    df["int_dti_risk"]  = df["int_rate"] * df["dti"]
    df["cr_age_diff"]   = df["mths_since_earliest_cr_line"] - df["mths_since_issue_d"]
    df["inc_dti_ratio"] = df["annual_inc"] / (df["dti"] + 1)
    drop_cols = ["grade","home_ownership","purpose","verification_status","term","income_band"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for c in feature_cols:
        if c not in df.columns: df[c] = 0
    return df[feature_cols]


# ── Helpers ───────────────────────────────────────────────────────────
def get_tier(p):
    if p < 0.08:  return "LOW RISK",  "#166534","#dcfce7","🟢"
    if p < 0.18:  return "MODERATE",  "#92400e","#fef3c7","🟡"
    if p < 0.30:  return "ELEVATED",  "#9a3412","#ffedd5","🟠"
    return               "HIGH RISK", "#991b1b","#fee2e2","🔴"

def run_model(inp):
    df    = engineer_single(inp)
    proba = model.predict_proba(df)[0]
    return {"p_default": float(proba[0]), "p_nond": float(proba[1])}

def make_gauge(p):
    color = ("#16a34a" if p<0.08 else "#d97706" if p<0.18
             else "#ea580c" if p<0.30 else "#dc2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p*100,1),
        number={"suffix":"%","font":{"size":28,"color":color,"family":"DM Sans"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#9ca3af","tickfont":{"size":9}},
            "bar":{"color":color,"thickness":0.26}, "bgcolor":"white",
            "steps":[{"range":[0,8],"color":"#dcfce7"},{"range":[8,18],"color":"#fef3c7"},
                     {"range":[18,30],"color":"#ffedd5"},{"range":[30,100],"color":"#fee2e2"}],
            "threshold":{"line":{"color":color,"width":3},"value":p*100},
        },
        title={"text":"Default Probability","font":{"size":11,"color":"#6b7280"}},
    ))
    fig.update_layout(height=195, margin=dict(t=25,b=0,l=12,r=12),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def ask_gemini(messages, inp, result):
    tier, *_ = get_tier(result["p_default"])
    api_key  = st.secrets.get("GEMINI_API_KEY","")
    if not api_key:
        return "⚠️ GEMINI_API_KEY not set. Go to Streamlit Settings → Secrets and add GEMINI_API_KEY."

    p_default_pct = f"{result['p_default']*100:.1f}"
    int_x_dti     = f"{inp['int_rate']*inp['dti']:.1f}"

    system = (
        "You are an expert credit risk analyst for a Probability of Default model.\n\n"
        "Current assessment:\n"
        f"- Default Probability: {p_default_pct}%  |  Risk Tier: {tier}\n"
        f"- Interest Rate: {inp['int_rate']}%  |  DTI: {inp['dti']}%\n"
        f"- Annual Income: ${inp['annual_inc']:,}  |  Grade: {inp['grade']}\n"
        f"- Employment: {inp['emp_length_int']} yrs  |  Term: {inp['term']}\n"
        f"- Inquiries 6mo: {inp['inq_last_6mths']}  |  Delinquent Accounts: {inp['acc_now_delinq']}\n"
        f"- Purpose: {inp['purpose']}  |  Ownership: {inp['home_ownership']}\n"
        f"- Int x DTI Score: {int_x_dti}\n\n"
        "Model: XGBoost trained on full data | AUC ~0.66 | 45 features | SMOTE\n"
        "Top predictors: int_rate, annual_inc, int_dti_risk, grade_num\n\n"
        "Be concise and practical. Plain language. Max 3 short paragraphs. "
        "Always write complete sentences and never cut off mid-thought."
    )

    try:
        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])]
            )
            for m in messages
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1500,
                temperature=0.4,
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI error: {e}\n\nTip: Make sure GEMINI_API_KEY is valid at aistudio.google.com"


# ── Session state ─────────────────────────────────────────────────────
for k, v in [("result",None),("assessed_inputs",{}),
              ("chat_history",[]),("api_messages",[])]:
    if k not in st.session_state: st.session_state[k] = v


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 PD Model")
    st.markdown("*Loan Risk Assessor*")
    st.markdown("---")

    st.markdown("##### Quick Presets")
    pc1, pc2, pc3 = st.columns(3)
    prime_btn    = pc1.button("🟢 Prime", use_container_width=True)
    avg_btn      = pc2.button("🟡 Avg",   use_container_width=True)
    highrisk_btn = pc3.button("🔴 High",  use_container_width=True)

    PRESETS = {
        "prime": dict(int_rate=7.5, annual_inc=120000, dti=8.0, emp_length_int=10,
                      inq_last_6mths=0, mths_since_issue_d=60, mths_since_earliest_cr_line=240,
                      acc_now_delinq=0, grade="A",
                      home_ownership="Own",
                      purpose="Credit Card",
                      verification_status="Verified", term="36 months"),
        "avg":   dict(int_rate=13.5, annual_inc=65000, dti=18.0, emp_length_int=5,
                      inq_last_6mths=1, mths_since_issue_d=48, mths_since_earliest_cr_line=180,
                      acc_now_delinq=0, grade="C",
                      home_ownership="Rent",
                      purpose="Debt Consolidation",
                      verification_status="Source Verified", term="36 months"),
        "high":  dict(int_rate=24.0, annual_inc=28000, dti=35.0, emp_length_int=1,
                      inq_last_6mths=4, mths_since_issue_d=12, mths_since_earliest_cr_line=60,
                      acc_now_delinq=1, grade="F",
                      home_ownership="Rent",
                      purpose="Small Business",
                      verification_status="Not Verified", term="60 months"),
    }

    if prime_btn:    st.session_state["preset"] = "prime"; st.rerun()
    if avg_btn:      st.session_state["preset"] = "avg";   st.rerun()
    if highrisk_btn: st.session_state["preset"] = "high";  st.rerun()

    preset_key = st.session_state.get("preset")
    P = PRESETS.get(preset_key, {})

    st.markdown("---")

    def sync_number_to_slider(key):
        st.session_state[f"sl_{key}"] = st.session_state[f"ni_{key}"]

    def sync_slider_to_number(key):
        st.session_state[f"ni_{key}"] = st.session_state[f"sl_{key}"]

    def num_slider(label, key, min_v, max_v, default, step):
        init_val = float(P.get(key, default))
        if f"ni_{key}" not in st.session_state:
            st.session_state[f"ni_{key}"] = init_val
        if f"sl_{key}" not in st.session_state:
            st.session_state[f"sl_{key}"] = init_val
        col_n, col_s = st.columns([1, 2])
        col_n.number_input(
            label, min_value=float(min_v), max_value=float(max_v),
            step=float(step), key=f"ni_{key}",
            on_change=sync_number_to_slider, args=(key,),
            label_visibility="collapsed",
        )
        col_s.slider(
            label, min_value=float(min_v), max_value=float(max_v),
            step=float(step), key=f"sl_{key}",
            on_change=sync_slider_to_number, args=(key,),
        )
        return st.session_state[f"ni_{key}"]

    def int_slider(label, key, min_v, max_v, default):
        init_val = int(P.get(key, default))
        if f"ni_{key}" not in st.session_state:
            st.session_state[f"ni_{key}"] = init_val
        if f"sl_{key}" not in st.session_state:
            st.session_state[f"sl_{key}"] = init_val
        col_n, col_s = st.columns([1, 2])
        col_n.number_input(
            label, min_value=min_v, max_value=max_v, step=1,
            key=f"ni_{key}", on_change=sync_number_to_slider, args=(key,),
            label_visibility="collapsed",
        )
        col_s.slider(
            label, min_value=min_v, max_value=max_v, step=1,
            key=f"sl_{key}", on_change=sync_slider_to_number, args=(key,),
        )
        return int(st.session_state[f"ni_{key}"])

    st.markdown("### Loan Details")
    int_rate = num_slider("Interest Rate (%)", "int_rate", 5.0, 30.0, 13.5, 0.1)
    term     = st.selectbox("Loan Term", ["36 months","60 months"],
                             index=0 if P.get("term","36 months")=="36 months" else 1,
                             key="sel_term")
    purpose  = st.selectbox("Loan Purpose", [
        "Debt Consolidation", "Credit Card", "Home Improvement", "Major Purchase",
        "Small Business", "Car", "Medical", "Moving", "Vacation", "Wedding",
        "Educational", "House", "Renewable Energy", "Other"],
        index=["Debt Consolidation", "Credit Card", "Home Improvement", "Major Purchase",
               "Small Business", "Car", "Medical", "Moving", "Vacation", "Wedding",
               "Educational", "House", "Renewable Energy", "Other"
               ].index(P.get("purpose", "Debt Consolidation")),
        key="sel_purpose")
    grade    = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"],
                             index=["A","B","C","D","E","F","G"].index(P.get("grade","C")),
                             key="sel_grade")

    st.markdown("### Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", 10000, 9999999,
                                  int(P.get("annual_inc",65000)), 1000, key="ni_annual_inc")
    dti        = num_slider("Debt-to-Income (%)", "dti", 0.0, 40.0, 18.0, 0.1)
    emp_length = int_slider("Employment (years)", "emp_length_int", 0, 10, 5)
    home_own   = st.selectbox("Home Ownership", ["Rent", "Mortgage", "Own"],
                               index=["Rent", "Mortgage", "Own"].index(P.get("home_ownership", "Rent")),
                               key="sel_home")
    verif      = st.selectbox("Verification Status",
                               ["Source Verified","Verified","Not Verified"],
                               index=["Source Verified","Verified","Not Verified"
                                      ].index(P.get("verification_status","Source Verified")),
                               key="sel_verif")

    st.markdown("### Credit History")
    inq_6mths  = int_slider("Inquiries (6mo)",        "inq_last_6mths",              0,  10,  1)
    mths_issue = int_slider("Months Since Issue",      "mths_since_issue_d",          0, 120, 48)
    mths_cr    = int_slider("Months Since 1st Credit", "mths_since_earliest_cr_line", 0, 1000, 60)
    acc_delinq = int_slider("Delinquent Accounts",     "acc_now_delinq",              0,   5,  0)

    st.markdown("---")
    run_btn = st.button("⚡ Run Assessment", use_container_width=True, type="primary")

# Clear preset after render
if preset_key:
    st.session_state.pop("preset", None)

PURPOSE_MAP = {
    "Debt Consolidation": "debt_consolidation",
    "Credit Card":        "credit_card",
    "Home Improvement":   "home_improvement",
    "Major Purchase":     "major_purchase",
    "Small Business":     "small_business",
    "Car":                "car",
    "Medical":            "medical",
    "Moving":             "moving",
    "Vacation":           "vacation",
    "Wedding":            "wedding",
    "Educational":        "educational",
    "House":              "house",
    "Renewable Energy":   "renewable_energy",
    "Other":              "other",
}
HOME_MAP = {
    "Rent":     "RENT",
    "Mortgage": "MORTGAGE",
    "Own":      "OWN",
}

inputs = dict(
    int_rate=float(int_rate), annual_inc=float(annual_inc), dti=float(dti),
    emp_length_int=int(emp_length), inq_last_6mths=int(inq_6mths),
    mths_since_issue_d=int(mths_issue), mths_since_earliest_cr_line=int(mths_cr),
    acc_now_delinq=int(acc_delinq), grade=grade,
    home_ownership=HOME_MAP[home_own],
    purpose=PURPOSE_MAP[purpose],
    verification_status=verif, term=term,
)

if run_btn:
    st.session_state["result"]          = run_model(inputs)
    st.session_state["assessed_inputs"] = inputs.copy()


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
st.markdown("# 🏦 Probability of Default — Loan Risk Assessor")
st.markdown("*XGBoost · 45 Features · SMOTE · Gemini AI*")

result = st.session_state["result"]
inp    = st.session_state["assessed_inputs"] if st.session_state["assessed_inputs"] else inputs

if result is None:
    st.info("👈 Fill in the sidebar and click **⚡ Run Assessment** to score the loan.")
    st.stop()

p_def = result["p_default"]
tier, tc, tb, icon = get_tier(p_def)

# Verdict banner
p_def_fmt  = f"{p_def*100:.2f}"
p_nond_fmt = f"{result['p_nond']*100:.2f}"
st.markdown(f"""
<div class="rbanner" style="background:{tb};border-color:{tc}33">
    <div style="font-size:2.6rem">{icon}</div>
    <div>
        <div style="font-size:1.8rem;font-weight:800;color:{tc}">{tier}</div>
        <div style="color:#64748b;font-size:0.86rem;margin-top:3px">
            Default probability: <strong style="color:{tc}">{p_def_fmt}%</strong>
            &nbsp;·&nbsp;
            Repayment probability: <strong>{p_nond_fmt}%</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI row
p_def_kpi    = f"{p_def*100:.1f}%"
inc_kpi      = f"${inp['annual_inc']/1000:.0f}k"
kc = st.columns(5)
for col, val, lbl, color in [
    (kc[0], p_def_kpi,               "Default Prob.", tc),
    (kc[1], inp["grade"],             "Loan Grade",   "#1a3c6e"),
    (kc[2], f"{inp['int_rate']}%",    "Interest Rate","#f39c12"),
    (kc[3], f"{inp['dti']}%",         "DTI Ratio",    "#2e86ab"),
    (kc[4], inc_kpi,                  "Annual Income","#27ae60"),
]:
    col.markdown(
        f'<div class="mcard" style="border-color:{color}">'
        f'<div class="v" style="color:{color}">{val}</div>'
        f'<div class="l">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Gauge + Factors
col_g, col_f = st.columns([1, 1.6])
with col_g:
    st.plotly_chart(make_gauge(p_def), use_container_width=True)
    st.markdown('<div class="shdr">Engineered Features</div>', unsafe_allow_html=True)
    ea, eb = st.columns(2)
    ea.metric("Int×DTI",    f"{inp['int_rate']*inp['dti']:.1f}")
    eb.metric("Log Income", f"{np.log1p(inp['annual_inc']):.3f}")
    ec, ed = st.columns(2)
    ec.metric("Credit Age", f"{inp['mths_since_earliest_cr_line']-inp['mths_since_issue_d']}mo")
    ed.metric("Inc/DTI",    f"{inp['annual_inc']/(inp['dti']+1):.0f}")

with col_f:
    st.markdown('<div class="shdr">Risk Factor Analysis</div>', unsafe_allow_html=True)
    CHIP = {"high":("#fee2e2","#991b1b","#dc2626"),
            "low": ("#dcfce7","#166534","#16a34a"),
            "neutral":("#f3f4f6","#374151","#9ca3af")}
    for fname, fval, ftype, fdesc in [
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
    ]:
        bg, tc2, dot = CHIP[ftype]
        st.markdown(f"""
        <div class="fchip" style="background:{bg}">
          <div style="display:flex;justify-content:space-between">
            <span style="font-weight:600;color:{tc2}">{fname}</span>
            <span style="font-weight:700;color:{tc2}">
              <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                background:{dot};margin-right:4px;vertical-align:middle"></span>{fval}
            </span>
          </div>
          <div style="font-size:0.76rem;color:{tc2};opacity:0.8;margin-top:2px">{fdesc}</div>
        </div>""", unsafe_allow_html=True)

# Summary chips
st.markdown('<div class="shdr">Loan Summary</div>', unsafe_allow_html=True)
st.markdown(" &nbsp; ".join(
    f'<span style="background:#f1f5f9;padding:3px 9px;border-radius:6px;'
    f'font-size:0.78rem;color:#475569">{c}</span>'
    for c in [f"Grade: {inp['grade']}", f"Rate: {inp['int_rate']}%",
              f"DTI: {inp['dti']}%", f"Term: {inp['term']}",
              f"Purpose: {inp['purpose'].replace('_',' ')}",
              f"Ownership: {inp['home_ownership']}"]),
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════════
# AI ANALYST
# ════════════════════════════════════════════════════════════════
st.markdown("## 🤖 AI Credit Risk Analyst")
st.markdown("*Powered by Gemini — ask anything about the result above.*")

# Quick question buttons — send directly without form
qc = st.columns(4)
for i, qp in enumerate(["Explain the key risk drivers",
                          "How to reduce this borrower's risk?",
                          "What does AUC mean here?",
                          "Compare Grade A vs current grade"]):
    if qc[i].button(qp, key=f"qp{i}", use_container_width=True):
        st.session_state["chat_history"].append({"role": "user", "content": qp})
        st.session_state["api_messages"].append({"role": "user", "content": qp})
        with st.spinner("Thinking..."):
            reply = ask_gemini(st.session_state["api_messages"], inp, result)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.session_state["api_messages"].append({"role": "assistant", "content": reply})
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Manual input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question",
        placeholder="e.g. What is the biggest risk factor? How would a lower DTI help?",
    )
    send = st.form_submit_button("Send ➜")

if send and user_input.strip():
    st.session_state["chat_history"].append({"role":"user","content":user_input})
    st.session_state["api_messages"].append({"role":"user","content":user_input})
    with st.spinner("Thinking..."):
        reply = ask_gemini(st.session_state["api_messages"], inp, result)
    st.session_state["chat_history"].append({"role":"assistant","content":reply})
    st.session_state["api_messages"].append({"role":"assistant","content":reply})
    st.rerun()

# Show conversation newest first
if st.session_state["chat_history"]:
    st.markdown("---")
    st.markdown("**Conversation** *(newest first)*")
    for msg in reversed(st.session_state["chat_history"]):
        css = "chat-u" if msg["role"]=="user" else "chat-a"
        pfx = "👤" if msg["role"]=="user" else "🤖"
        st.markdown(f'<div class="{css}">{pfx} {msg["content"]}</div>',
                    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear chat"):
        st.session_state["chat_history"] = []
        st.session_state["api_messages"] = []
        st.rerun()
