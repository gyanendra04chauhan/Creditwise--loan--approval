import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise — Loan Approval",
    page_icon="💳",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .stButton > button {
      background: #185FA5; color: white; font-weight: 600;
      border: none; border-radius: 8px; padding: 0.6rem 1.5rem;
      width: 100%; font-size: 16px;
  }
  .stButton > button:hover { background: #0C447C; }
  .approved-box {
      background: #EAF3DE; border: 1.5px solid #3B6D11;
      border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
  }
  .rejected-box {
      background: #FCEBEB; border: 1.5px solid #A32D2D;
      border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
  }
  .metric-box {
      background: #f1f0eb; border-radius: 10px;
      padding: 0.8rem; text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ── Data generation (same distribution as original dataset) ───────────────────
@st.cache_data
def generate_and_train():
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        'Applicant_Income':    np.random.exponential(8000, n) + 2000,
        'Coapplicant_Income':  np.random.exponential(3000, n),
        'Age':                 np.random.randint(22, 65, n).astype(float),
        'Dependents':          np.random.choice([0,1,2,3], n, p=[0.4,0.3,0.2,0.1]).astype(float),
        'Credit_Score':        np.clip(np.random.normal(650, 80, n), 300, 850),
        'Existing_Loans':      np.random.choice([0,1,2,3,4], n, p=[0.2,0.3,0.25,0.15,0.1]).astype(float),
        'DTI_Ratio':           np.clip(np.random.normal(0.35, 0.15, n), 0.01, 0.99),
        'Savings':             np.random.exponential(15000, n),
        'Collateral_Value':    np.random.exponential(30000, n),
        'Loan_Amount':         np.random.exponential(20000, n) + 5000,
        'Loan_Term':           np.random.choice([12,24,36,48,60,72,84], n).astype(float),
        'Education_Level':     np.random.choice(['Graduate','Not Graduate'], n, p=[0.65,0.35]),
        'Employment_Status':   np.random.choice(['Salaried','Self-employed','Unemployed'], n, p=[0.6,0.3,0.1]),
        'Marital_Status':      np.random.choice(['Married','Single'], n, p=[0.6,0.4]),
        'Gender':              np.random.choice(['Male','Female'], n, p=[0.65,0.35]),
        'Employer_Category':   np.random.choice(['Government','Private','MNC','Unemployed'], n, p=[0.2,0.5,0.2,0.1]),
        'Loan_Purpose':        np.random.choice(['Home','Car','Business','Personal','Education'], n),
        'Property_Area':       np.random.choice(['Urban','Semiurban','Rural'], n, p=[0.4,0.35,0.25]),
    })

    # Target: loan approved based on sensible rules
    score = (
        (df['Credit_Score'] - 300) / 550 * 35 +
        np.clip((0.6 - df['DTI_Ratio']) / 0.6, 0, 1) * 20 +
        np.clip(df['Applicant_Income'] / 50000, 0, 1) * 15 +
        np.clip(df['Savings'] / 100000, 0, 1) * 10 +
        (df['Education_Level'] == 'Graduate').astype(int) * 5 +
        (df['Employment_Status'] == 'Salaried').astype(int) * 5 +
        np.random.normal(0, 5, n)
    )
    df['Loan_Approved'] = (score > 45).astype(int)

    # ── Preprocessing (exact same as notebook) ─────────────────────────────────
    df['Education_Level']  = (df['Education_Level'] == 'Graduate').astype(int)
    df['Marital_Status']   = (df['Marital_Status'] == 'Married').astype(int)

    df = pd.get_dummies(df, columns=[
        'Employment_Status','Loan_Purpose','Property_Area',
        'Gender','Employer_Category'
    ], drop_first=False)

    df['Credit_Score_sq'] = df['Credit_Score'] ** 2
    df['DTI_Ratio_sq']    = df['DTI_Ratio'] ** 2

    feature_cols = [c for c in df.columns if c != 'Loan_Approved']
    X = df[feature_cols]
    y = df['Loan_Approved']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    log_model  = LogisticRegression(max_iter=1000)
    nb_model   = GaussianNB()
    knn_model  = KNeighborsClassifier(n_neighbors=5)

    log_model.fit(X_train_s, y_train)
    nb_model.fit(X_train_s, y_train)
    knn_model.fit(X_train_s, y_train)

    return log_model, nb_model, knn_model, scaler, feature_cols


def preprocess_input(data: dict, feature_cols: list) -> np.ndarray:
    """Apply exact same preprocessing as notebook to a single input row."""
    row = {
        'Applicant_Income':   data['Applicant_Income'],
        'Coapplicant_Income': data['Coapplicant_Income'],
        'Age':                data['Age'],
        'Dependents':         data['Dependents'],
        'Credit_Score':       data['Credit_Score'],
        'Existing_Loans':     data['Existing_Loans'],
        'DTI_Ratio':          data['DTI_Ratio'],
        'Savings':            data['Savings'],
        'Collateral_Value':   data['Collateral_Value'],
        'Loan_Amount':        data['Loan_Amount'],
        'Loan_Term':          data['Loan_Term'],
        'Education_Level':    1 if data['Education_Level'] == 'Graduate' else 0,
        'Marital_Status':     1 if data['Marital_Status'] == 'Married' else 0,
        'Credit_Score_sq':    data['Credit_Score'] ** 2,
        'DTI_Ratio_sq':       data['DTI_Ratio'] ** 2,
    }

    # One-hot columns
    for col in feature_cols:
        if col not in row:
            row[col] = 0

    for emp in ['Salaried','Self-employed','Unemployed']:
        row[f'Employment_Status_{emp}'] = 1 if data['Employment_Status'] == emp else 0
    for purp in ['Home','Car','Business','Personal','Education']:
        row[f'Loan_Purpose_{purp}'] = 1 if data['Loan_Purpose'] == purp else 0
    for area in ['Urban','Semiurban','Rural']:
        row[f'Property_Area_{area}'] = 1 if data['Property_Area'] == area else 0
    for gen in ['Male','Female']:
        row[f'Gender_{gen}'] = 1 if data['Gender'] == gen else 0
    for emp_cat in ['Government','Private','MNC','Unemployed']:
        row[f'Employer_Category_{emp_cat}'] = 1 if data['Employer_Category'] == emp_cat else 0

    df_row = pd.DataFrame([row])
    # Align to training columns
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_cols]
    return df_row.values


# ── Load models ────────────────────────────────────────────────────────────────
with st.spinner("Models train ho rahe hain..."):
    log_model, nb_model, knn_model, scaler, feature_cols = generate_and_train()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 💳 CreditWise — Loan Approval Prediction")
st.markdown("*B.Tech CS (AI & ML) Project — Logistic Regression | Naive Bayes | KNN*")
st.divider()

# ── Model selector ─────────────────────────────────────────────────────────────
model_choice = st.radio(
    "**ML Model chunein:**",
    ["Logistic Regression (87.5%)", "Naive Bayes (86.5%)", "KNN — k=5 (75.5%)"],
    horizontal=True,
)

st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("### 📋 Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Information**")
    applicant_income   = st.number_input("Applicant Income (₹/month)", 0, 500000, 50000, step=1000)
    coapplicant_income = st.number_input("Co-applicant Income (₹/month)", 0, 500000, 0, step=1000)
    age                = st.number_input("Age", 18, 80, 30)
    dependents         = st.number_input("Dependents", 0, 10, 0)
    gender             = st.selectbox("Gender", ["Male", "Female"])
    marital_status     = st.selectbox("Marital Status", ["Married", "Single"])

with col2:
    st.markdown("**Financial Profile**")
    credit_score     = st.slider("Credit Score", 300, 850, 700)
    dti_ratio        = st.slider("DTI Ratio", 0.01, 0.99, 0.30, step=0.01)
    savings          = st.number_input("Savings (₹)", 0, 1000000, 100000, step=5000)
    collateral_value = st.number_input("Collateral Value (₹)", 0, 2000000, 200000, step=10000)
    existing_loans   = st.number_input("Existing Loans", 0, 10, 1)
    education_level  = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

with col3:
    st.markdown("**Loan Details**")
    loan_amount      = st.number_input("Loan Amount (₹)", 1000, 2000000, 300000, step=5000)
    loan_term        = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=4)
    loan_purpose     = st.selectbox("Loan Purpose", ["Home", "Car", "Business", "Personal", "Education"])
    property_area    = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    employer_category = st.selectbox("Employer Category", ["Government", "Private", "MNC", "Unemployed"])

st.divider()

# ── Predict button ─────────────────────────────────────────────────────────────
predict_clicked = st.button("🔮 Predict Loan Approval")

if predict_clicked:
    input_data = {
        'Applicant_Income':   applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Age':                age,
        'Dependents':         dependents,
        'Credit_Score':       credit_score,
        'Existing_Loans':     existing_loans,
        'DTI_Ratio':          dti_ratio,
        'Savings':            savings,
        'Collateral_Value':   collateral_value,
        'Loan_Amount':        loan_amount,
        'Loan_Term':          float(loan_term),
        'Education_Level':    education_level,
        'Employment_Status':  employment_status,
        'Marital_Status':     marital_status,
        'Gender':             gender,
        'Employer_Category':  employer_category,
        'Loan_Purpose':       loan_purpose,
        'Property_Area':      property_area,
    }

    X_input = preprocess_input(input_data, feature_cols)
    X_scaled = scaler.transform(X_input)

    if "Logistic" in model_choice:
        model = log_model
        model_name = "Logistic Regression"
    elif "Naive" in model_choice:
        model = nb_model
        model_name = "Naive Bayes"
    else:
        model = knn_model
        model_name = "KNN (k=5)"

    prediction = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0]
    confidence = round(max(proba) * 100, 1)
    approved   = prediction == 1

    # ── Result display ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Prediction Result")
    r1, r2, r3 = st.columns([2, 1, 1])

    with r1:
        if approved:
            st.markdown(f"""
            <div class="approved-box">
              <h2 style="color:#3B6D11;margin:0">✅ Loan Approved</h2>
              <p style="margin:6px 0 0;color:#27500A">Model: <b>{model_name}</b> &nbsp;|&nbsp; Confidence: <b>{confidence}%</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rejected-box">
              <h2 style="color:#A32D2D;margin:0">❌ Loan Rejected</h2>
              <p style="margin:6px 0 0;color:#791F1F">Model: <b>{model_name}</b> &nbsp;|&nbsp; Confidence: <b>{confidence}%</b></p>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.metric("Approval Probability", f"{round(proba[1]*100, 1)}%")
        st.metric("Rejection Probability", f"{round(proba[0]*100, 1)}%")

    with r3:
        st.metric("Credit Score", credit_score)
        st.metric("DTI Ratio", dti_ratio)

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Analysis")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Model Accuracy Comparison**")
        fig, ax = plt.subplots(figsize=(5, 3))
        models  = ['Logistic\nRegression', 'Naive\nBayes', 'KNN']
        accs    = [87.5, 86.5, 75.5]
        colors  = ['#185FA5' if model_name.split()[0] in m else '#B5D4F4' for m in models]
        bars = ax.bar(models, accs, color=colors, edgecolor='none', width=0.5)
        ax.set_ylim(60, 95)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Test Accuracy', fontsize=11)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**Prediction Confidence**")
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ['Rejected', 'Approved']
        sizes  = [proba[0]*100, proba[1]*100]
        colors_pie = ['#F09595', '#97C459']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight('bold')
        ax.set_title(f'Probability Distribution\n({model_name})', fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Key factors ────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Key Factors")
    factors = []

    if credit_score >= 700:
        factors.append(("✅", f"Strong credit score ({credit_score})", "positive"))
    elif credit_score < 600:
        factors.append(("❌", f"Low credit score ({credit_score}) — major concern", "negative"))
    else:
        factors.append(("⚠️", f"Moderate credit score ({credit_score})", "neutral"))

    if dti_ratio <= 0.35:
        factors.append(("✅", f"Healthy DTI ratio ({dti_ratio:.2f})", "positive"))
    elif dti_ratio >= 0.5:
        factors.append(("❌", f"High DTI ratio ({dti_ratio:.2f}) — risk signal", "negative"))
    else:
        factors.append(("⚠️", f"Borderline DTI ratio ({dti_ratio:.2f})", "neutral"))

    total_income = applicant_income + coapplicant_income
    if total_income >= 60000:
        factors.append(("✅", f"Good combined income (₹{total_income:,.0f}/mo)", "positive"))
    elif total_income < 20000:
        factors.append(("❌", f"Low combined income (₹{total_income:,.0f}/mo)", "negative"))
    else:
        factors.append(("⚠️", f"Moderate combined income (₹{total_income:,.0f}/mo)", "neutral"))

    if education_level == "Graduate":
        factors.append(("✅", "Graduate — positive eligibility factor", "positive"))
    if employment_status == "Salaried":
        factors.append(("✅", "Salaried employment — stable income", "positive"))
    elif employment_status == "Unemployed":
        factors.append(("❌", "Unemployed — high-risk flag", "negative"))
    if employer_category == "Government":
        factors.append(("✅", "Government employer — high job stability", "positive"))

    f1, f2 = st.columns(2)
    for i, (icon, text, _) in enumerate(factors[:6]):
        col = f1 if i % 2 == 0 else f2
        col.markdown(f"{icon} {text}")

    st.divider()
    st.caption("⚠️ Note: This is a simulated demo. For production use, train with your actual loan_approval_data.csv and save the model using `joblib`.")
