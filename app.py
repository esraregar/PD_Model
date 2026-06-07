import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PD Model — Loan Risk Assessor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #f4f7fb; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid;
    text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 800; margin-bottom: 2px; }
.metric-card .label { font-size: 0.75rem; color: #6b7280; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em; }

.risk-banner {
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.section-header {
    font-size: 0.7rem; font-weight: 700; color: #9ca3af;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 8px; margin-top: 18px;
}
.factor-chip {
    border-radius: 8px; padding: 8px 12px;
    margin-bottom: 8px; font-size: 0.82rem;
}
.chat-msg-user {
    background: #1a3c6e; color: white;
    border-radius: 14px 14px 4px 14px;
    padding: 10px 15px; margin: 6px 0;
    margin-left: 20%; font-size: 0.88rem;
}
.chat-msg-ai {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 14px 14px 14px 4px;
    padding: 10px 15px; margin: 6px 0;
    margin-right: 20%; font-size: 0.88rem; line-height: 1.6;
}
.stButton>button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stSidebarContent"] { background: #0d1b2a; }
div[data-testid="stSidebarContent"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = Path(__file__).parent / "pd_model.pkl"
    data = pickle.load(open(path, "rb"))
    return data["model"], data["feature_cols"]

model, feature_cols = load_model()


# ── Feature engineering (mirrors training pipeline exactly) ──────────
def engineer_single(inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([inputs])
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
        ['small_business', 'educational', 'moving']).astype(int)

    bins   = [0, 34000, 48000, 65000, 90000, float('inf')]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df['income_band'] = pd.cut(df['annual_inc'], bins=bins, labels=labels).astype(str)

    for col in ['home_ownership', 'purpose', 'verification_status', 'income_band']:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)

    df['int_dti_risk']  = df['int_rate'] * df['dti']
    df['cr_age_diff']   = df['mths_since_earliest_cr_line'] - df['mths_since_issue_d']
    df['inc_dti_ratio'] = df['annual_inc'] / (df['dti'] + 1)

    drop_cols = ['grade', 'home_ownership', 'purpose', 'verification_status',
                 'term', 'income_band']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_cols]
    return df


# ── Risk tier ─────────────────────────────────────────────────────────
def get_tier(p):
    if p < 0.08:  return "LOW RISK",  "#166534", "#dcfce7", "🟢"
    if p < 0.18:  return "MODERATE",  "#92400e", "#fef3c7", "🟡"
    if p < 0.30:  return "ELEVATED",  "#9a3412", "#ffedd5", "🟠"
    return               "HIGH RISK", "#991b1b", "#fee2e2", "🔴"


# ── Gauge chart ───────────────────────────────────────────────────────
def make_gauge(p):
    color = ("#16a34a" if p < 0.08 else
             "#d97706" if p < 0.18 else
             "#ea580c" if p < 0.30 else "#dc2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(p * 100, 1),
        number={"suffix": "%", "font": {"size": 32, "color": color, "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9ca3af", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  8],  "color": "#dcfce7"},
                {"range": [8,  18], "color": "#fef3c7"},
                {"range": [18, 30], "color": "#ffedd5"},
                {"range": [30, 100],"color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": p * 100},
        },
        title={"text": "Default Probability", "font": {"size": 13, "color": "#6b7280"}},
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Feature importance chart ──────────────────────────────────────────
def make_fi_bar():
    fi = pd.DataFrame({"feature": feature_cols,
                        "importance": model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False).head(12)
    fig = px.bar(fi.sort_values("importance"), x="importance", y="feature",
                 orientation="h", color="importance",
                 color_continuous_scale=["#dbeafe", "#1a3c6e"])
    fig.update_layout(
        height=320, margin=dict(t=10, b=0, l=0, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        showlegend=False, coloraxis_showscale=False,
        xaxis_title="Importance", yaxis_title="",
        font=dict(family="DM Sans", size=11, color="#374151"),
    )
    fig.update_traces(marker_line_width=0)
    return fig


# ── Gemini AI analyst ─────────────────────────────────────────────────
def ask_gemini(messages, inputs, result):
    tier, _, _, _ = get_tier(result["p_default"])

    system = f"""You are an expert credit risk analyst for a Probability of Default (PD) model app.

Current loan assessment:
- Default Probability: {result['p_default']*100:.1f}%
- Risk Tier: {tier}
- Interest Rate: {inputs['int_rate']}%  |  DTI: {inputs['dti']}%
- Annual Income: ${inputs['annual_inc']:,}  |  Grade: {inputs['grade']}
- Employment: {inputs['emp_length_int']} years  |  Term: {inputs['term']}
- Inquiries 6mo: {inputs['inq_last_6mths']}  |  Delinquent Accounts: {inputs['acc_now_delinq']}
- Purpose: {inputs['purpose']}  |  Home Ownership: {inputs['home_ownership']}
- Int×DTI Risk Score: {inputs['int_rate'] * inputs['dti']:.1f}

Model: XGBoost | AUC 0.6647 | Accuracy 85.5% | 45 engineered features | SMOTE for class imbalance
Top predictors: int_rate, annual_inc, int_dti_risk, grade_num, mths_since_issue_d

Be concise, specific, and practical. Use plain language. Max 3 short paragraphs."""

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ GEMINI_API_KEY not found in secrets. Please add it in Streamlit settings."

    genai.configure(api_key=api_key)
    ai_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system,
    )

    # Convert history to Gemini format (exclude last message — it's the current one)
    history = []
    for msg in messages[:-1]:
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })

    chat = ai_model.start_chat(history=history)
    response = chat.send_message(messages[-1]["content"])
    return response.text


# ════════════════════════════════════════════════════════════════
# SIDEBAR — Input Form
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 PD Model")
    st.markdown("*Probability of Default Assessor*")
    st.markdown("---")

    st.markdown("### Loan Details")
    int_rate  = st.slider("Interest Rate (%)", 5.0, 30.0, 13.5, 0.1)
    term      = st.selectbox("Loan Term", ["36 months", "60 months"])
    purpose   = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
        "small_business", "car", "medical", "moving", "vacation", "wedding",
        "educational", "house", "renewable_energy", "other"])
    grade     = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"], index=2)

    st.markdown("### Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 65000, 1000)
    dti        = st.slider("Debt-to-Income (%)", 0.0, 40.0, 18.0, 0.1)
    emp_length = st.slider("Employment Length (years)", 0, 10, 5)
    home_own   = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
    verif      = st.selectbox("Verification Status",
                               ["Source Verified", "Verified", "Not Verified"])

    st.markdown("### Credit History")
    inq_6mths  = st.slider("Inquiries Last 6 Months", 0, 10, 1)
    mths_issue = st.slider("Months Since Loan Issue", 0, 120, 48)
    mths_cr    = st.slider("Months Since First Credit Line", 12, 400, 180)
    acc_delinq = st.slider("Current Delinquent Accounts", 0, 5, 0)

    st.markdown("---")
    run_btn = st.button("⚡ Run Assessment", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("##### Quick Presets")
    c1, c2, c3 = st.columns(3)
    prime_btn    = c1.button("🟢 Prime", use_container_width=True)
    avg_btn      = c2.button("🟡 Avg",   use_container_width=True)
    highrisk_btn = c3.button("🔴 High",  use_container_width=True)


# ── Preset definitions ────────────────────────────────────────────────
PRESETS = {
    "prime": dict(int_rate=7.5, annual_inc=120000, dti=8.0, emp_length_int=10,
                  inq_last_6mths=0, mths_since_issue_d=60,
                  mths_since_earliest_cr_line=240, acc_now_delinq=0,
                  grade="A", home_ownership="OWN", purpose="credit_card",
                  verification_status="Verified", term="36 months"),
    "avg":   dict(int_rate=13.5, annual_inc=65000, dti=18.0, emp_length_int=5,
                  inq_last_6mths=1, mths_since_issue_d=48,
                  mths_since_earliest_cr_line=180, acc_now_delinq=0,
                  grade="C", home_ownership="RENT", purpose="debt_consolidation",
                  verification_status="Source Verified", term="36 months"),
    "high":  dict(int_rate=24.0, annual_inc=28000, dti=35.0, emp_length_int=1,
                  inq_last_6mths=4, mths_since_issue_d=12,
                  mths_since_earliest_cr_line=60, acc_now_delinq=1,
                  grade="F", home_ownership="RENT", purpose="small_business",
                  verification_status="Not Verified", term="60 months"),
}

if prime_btn:
    st.session_state["preset"] = "prime"; st.rerun()
if avg_btn:
    st.session_state["preset"] = "avg";   st.rerun()
if highrisk_btn:
    st.session_state["preset"] = "high";  st.rerun()


# ── Build inputs dict ─────────────────────────────────────────────────
preset = st.session_state.get("preset")
if preset and preset in PRESETS:
    inputs = PRESETS[preset]
else:
    inputs = dict(
        int_rate=int_rate, annual_inc=annual_inc, dti=dti,
        emp_length_int=emp_length, inq_last_6mths=inq_6mths,
        mths_since_issue_d=mths_issue, mths_since_earliest_cr_line=mths_cr,
        acc_now_delinq=acc_delinq, grade=grade, home_ownership=home_own,
        purpose=purpose, verification_status=verif, term=term,
    )


# ── Run model ─────────────────────────────────────────────────────────
def run_model(inp):
    df = engineer_single(inp)
    p_default = float(model.predict_proba(df)[0, 1])
    return {"p_default": p_default, "p_nond": 1 - p_default}

# Initialise session state
for key, val in [("result", None), ("chat_history", []),
                  ("api_messages", []), ("assessed_inputs", inputs)]:
    if key not in st.session_state:
        st.session_state[key] = val

if run_btn or preset:
    st.session_state["result"]           = run_model(inputs)
    st.session_state["assessed_inputs"]  = inputs
    if preset:
        st.session_state["preset"] = None


# ════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════
st.markdown("# 🏦 Probability of Default Model")
st.markdown("*XGBoost · AUC 0.6647 · 45 Features · SMOTE · Gemini AI Powered*")
st.markdown("---")

tab_assess, tab_result, tab_ai = st.tabs(
    ["📋 Assessment", "📊 Risk Report", "🤖 AI Analyst"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — ASSESSMENT INFO
# ════════════════════════════════════════════════════════════════
with tab_assess:
    st.markdown("### How to use this app")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**1. Fill in the sidebar** with borrower and loan details.

**2. Click ⚡ Run Assessment** to score the loan using the trained XGBoost model.

**3. View the Risk Report tab** for the full breakdown — default probability, risk tier,
factor analysis, and feature importances.

**4. Ask the AI Analyst tab** to explain the result, compare scenarios,
or suggest how to reduce risk.
        """)
    with c2:
        st.info("""
**Model Details**
- Algorithm: XGBoost (300 trees)
- Training records: 20,400
- Features: 45 engineered
- AUC: 0.6647
- Accuracy: 85.5%
- Class imbalance: SMOTE (85:15 → 50:50)
- AI: Gemini 1.5 Flash (free)
        """)

    st.markdown("### Feature Engineering Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Features",    "13", help="Original dataset columns")
    col2.metric("Transformations", "9",  help="grade_num, log_annual_inc, binary flags…")
    col3.metric("Combinations",    "3",  help="int_dti_risk, cr_age_diff, inc_dti_ratio")
    col4.metric("OHE Dummies",     "20", help="home_ownership, purpose, verification, income_band")

    st.markdown("### Risk Tiers")
    t1, t2, t3, t4 = st.columns(4)
    t1.success("🟢 Low Risk — < 8%")
    t2.warning("🟡 Moderate — 8–18%")
    t3.error("🟠 Elevated — 18–30%")
    t4.error("🔴 High Risk — > 30%")


# ════════════════════════════════════════════════════════════════
# TAB 2 — RISK REPORT
# ════════════════════════════════════════════════════════════════
with tab_result:
    result = st.session_state["result"]
    inp    = st.session_state["assessed_inputs"]

    if result is None:
        st.info("👈 Fill in the sidebar and click **⚡ Run Assessment** to see results.")
    else:
        p_def = result["p_default"]
        tier, tc, tb, icon = get_tier(p_def)

        # Verdict banner
        st.markdown(f"""
        <div class="risk-banner" style="background:{tb};border:2px solid {tc}33">
            <div style="font-size:3rem">{icon}</div>
            <div style="flex:1">
                <div style="font-size:2rem;font-weight:800;color:{tc}">{tier}</div>
                <div style="color:#64748b;font-size:0.9rem;margin-top:4px">
                    <strong style="color:{tc}">{p_def*100:.2f}%</strong>
                    default probability &nbsp;·&nbsp;
                    <strong>{result['p_nond']*100:.2f}%</strong> repayment probability
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(f'<div class="metric-card" style="border-color:{tc}"><div class="value" style="color:{tc}">{p_def*100:.1f}%</div><div class="label">Default Prob.</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="metric-card" style="border-color:#1a3c6e"><div class="value" style="color:#1a3c6e">{inp["grade"]}</div><div class="label">Loan Grade</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="metric-card" style="border-color:#f39c12"><div class="value" style="color:#f39c12">{inp["int_rate"]}%</div><div class="label">Interest Rate</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="metric-card" style="border-color:#2e86ab"><div class="value" style="color:#2e86ab">{inp["dti"]}%</div><div class="label">DTI Ratio</div></div>', unsafe_allow_html=True)
        k5.markdown(f'<div class="metric-card" style="border-color:#27ae60"><div class="value" style="color:#27ae60">${inp["annual_inc"]/1000:.0f}k</div><div class="label">Annual Income</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_g, col_f = st.columns([1, 1.6])

        with col_g:
            st.plotly_chart(make_gauge(p_def), use_container_width=True)

            st.markdown('<div class="section-header">Engineered Feature Values</div>',
                        unsafe_allow_html=True)
            ef1, ef2 = st.columns(2)
            ef1.metric("Int×DTI Risk",  f"{inp['int_rate']*inp['dti']:.1f}")
            ef2.metric("Log Income",    f"{np.log1p(inp['annual_inc']):.3f}")
            ef3, ef4 = st.columns(2)
            ef3.metric("Credit Age Gap",
                       f"{inp['mths_since_earliest_cr_line']-inp['mths_since_issue_d']} mo")
            ef4.metric("Inc/DTI Ratio",
                       f"{inp['annual_inc']/(inp['dti']+1):.0f}")

        with col_f:
            st.markdown('<div class="section-header">Risk Factor Analysis</div>',
                        unsafe_allow_html=True)

            factors = [
                ("Interest Rate",   f"{inp['int_rate']}%",
                 "high"    if inp['int_rate'] > 18 else
                 "low"     if inp['int_rate'] < 10 else "neutral",
                 "Above-average rate — lender already pricing elevated risk"
                 if inp['int_rate'] > 18 else
                 "Low rate reflects strong creditworthiness"
                 if inp['int_rate'] < 10 else "Market-rate loan"),

                ("Annual Income",   f"${inp['annual_inc']:,}",
                 "low"  if inp['annual_inc'] > 80000 else
                 "high" if inp['annual_inc'] < 35000 else "neutral",
                 "Strong income buffer"        if inp['annual_inc'] > 80000 else
                 "Limited income headroom"     if inp['annual_inc'] < 35000 else
                 "Average income profile"),

                ("Debt-to-Income",  f"{inp['dti']}%",
                 "high" if inp['dti'] > 25 else
                 "low"  if inp['dti'] < 12 else "neutral",
                 "High leverage increases default probability" if inp['dti'] > 25 else
                 "Low debt burden supports repayment"          if inp['dti'] < 12 else
                 "Moderate leverage"),

                ("Loan Grade",      inp['grade'],
                 "low"  if inp['grade'] in ['A','B'] else
                 "high" if inp['grade'] in ['F','G'] else "neutral",
                 "Prime grade — strong credit quality" if inp['grade'] in ['A','B'] else
                 "Sub-prime — elevated risk"           if inp['grade'] in ['F','G'] else
                 "Mid-grade borrower"),

                ("Inquiries (6mo)", str(inp['inq_last_6mths']),
                 "high" if inp['inq_last_6mths'] >= 3 else
                 "low"  if inp['inq_last_6mths'] == 0 else "neutral",
                 "Multiple inquiries suggest credit stress"
                 if inp['inq_last_6mths'] >= 3 else
                 "No recent credit-seeking activity"),

                ("Delinquency",     f"{inp['acc_now_delinq']} active",
                 "high" if inp['acc_now_delinq'] > 0 else "low",
                 "Active delinquency is a strong default signal"
                 if inp['acc_now_delinq'] > 0 else
                 "Clean payment record"),
            ]

            CHIP_COLORS = {
                "high":    ("#fee2e2", "#991b1b", "#dc2626"),
                "low":     ("#dcfce7", "#166534", "#16a34a"),
                "neutral": ("#f3f4f6", "#374151", "#9ca3af"),
            }

            for fname, fval, ftype, fdesc in factors:
                bg, tc2, dot = CHIP_COLORS[ftype]
                st.markdown(f"""
                <div class="factor-chip" style="background:{bg}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-weight:600;color:{tc2}">{fname}</span>
                        <span style="font-weight:700;color:{tc2}">
                            <span style="display:inline-block;width:8px;height:8px;
                                border-radius:50%;background:{dot};
                                margin-right:5px;vertical-align:middle"></span>
                            {fval}
                        </span>
                    </div>
                    <div style="font-size:0.78rem;color:{tc2};opacity:0.8;margin-top:3px">
                        {fdesc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Feature importance
        st.markdown('<div class="section-header">Top 12 Feature Importances (XGBoost)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_fi_bar(), use_container_width=True)

        # Loan summary chips
        st.markdown('<div class="section-header">Loan Summary</div>',
                    unsafe_allow_html=True)
        chips = [
            f"Grade: {inp['grade']}",
            f"Rate: {inp['int_rate']}%",
            f"DTI: {inp['dti']}%",
            f"Term: {inp['term']}",
            f"Purpose: {inp['purpose'].replace('_',' ')}",
            f"Ownership: {inp['home_ownership']}",
            f"Verification: {inp['verification_status']}",
        ]
        st.markdown(
            " &nbsp; ".join([
                f'<span style="background:#f1f5f9;padding:4px 10px;'
                f'border-radius:6px;font-size:0.8rem;color:#475569">{c}</span>'
                for c in chips
            ]),
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════
# TAB 3 — AI ANALYST
# ════════════════════════════════════════════════════════════════
with tab_ai:
    result = st.session_state["result"]
    inp    = st.session_state["assessed_inputs"]

    st.markdown("### 🤖 AI Credit Risk Analyst")
    st.markdown("*Powered by Gemini 1.5 Flash (free) — ask anything about "
                "the assessment, model, or risk factors.*")

    if result is None:
        st.warning("Run an assessment first, then come back to chat with the AI analyst.")
    else:
        # Quick prompt buttons
        st.markdown("**Quick prompts:**")
        qp_cols = st.columns(4)
        qp_list = [
            "Explain the key risk drivers",
            "How to reduce this borrower's risk?",
            "What does AUC 0.6647 mean?",
            "Compare Grade A vs current grade",
        ]
        for i, qp in enumerate(qp_list):
            if qp_cols[i].button(qp, use_container_width=True):
                st.session_state["pending_msg"] = qp

        # Chat history display
        for msg in st.session_state["chat_history"]:
            role_class = ("chat-msg-user" if msg["role"] == "user"
                          else "chat-msg-ai")
            prefix = "👤 " if msg["role"] == "user" else "🤖 "
            st.markdown(
                f'<div class="{role_class}">{prefix}{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask the AI analyst...",
                value=st.session_state.pop("pending_msg", ""),
                placeholder="e.g. What's driving the risk? How would lower DTI help?",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Send →")

        if send and user_input.strip():
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input})
            st.session_state["api_messages"].append(
                {"role": "user", "content": user_input})

            with st.spinner("Analysing..."):
                try:
                    reply = ask_gemini(
                        st.session_state["api_messages"], inp, result)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": reply})
                    st.session_state["api_messages"].append(
                        {"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("🗑️ Clear chat"):
                st.session_state["chat_history"] = []
                st.session_state["api_messages"] = []
                st.rerun()

        # Context panel
        with st.expander("ℹ️ Assessment context sent to AI"):
            tier2, _, _, _ = get_tier(result["p_default"])
            st.json({
                "default_probability": f"{result['p_default']*100:.2f}%",
                "risk_tier": tier2,
                "int_rate": inp["int_rate"],
                "dti": inp["dti"],
                "annual_inc": inp["annual_inc"],
                "grade": inp["grade"],
                "int_dti_risk": round(inp["int_rate"] * inp["dti"], 2),
                "model": "XGBoost | AUC 0.6647 | 45 features",
            })
