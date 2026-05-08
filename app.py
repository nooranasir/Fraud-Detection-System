"""
Fraud Detection Web App — Streamlit Deployment
Matches the Fraud-Detection.ipynb pipeline exactly:
  - Models: XGBoost (Optuna-tuned), LightGBM, Random Forest, Decision Tree, Logistic Regression
  - Preprocessing: fillna(-999), label-encode categoricals, StandardScaler
  - Hybrid: KMeans cluster feature
  - Evaluation: F1 Score, ROC-AUC, Confusion Matrix, model comparison bar chart
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fraud Detection System")
st.markdown("**Dataset:** Vesta Real-World E-Commerce Transactions (Kaggle)")
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Introduction", "🎯 Live Prediction Demo"]
)

# ─────────────────────────────────────────────
# Helper: load saved models if they exist
# ─────────────────────────────────────────────
@st.cache_resource
def load_saved_models():
    models = {}
    files = {
        "XGBoost (Optuna)": "xgb_model.pkl",
        "LightGBM": "lgbm_model.pkl"
    }
    scaler, kmeans = None, None
    try:
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open("kmeans.pkl", "rb") as kmeans_file:
            kmeans = pickle.load(kmeans_file)
        for name, f in files.items():
            with open(f, "rb") as model_file:
                models[name] = pickle.load(model_file)
        return models, scaler, kmeans, True
    except (FileNotFoundError, pickle.UnpicklingError, OSError) as exc:
        st.error(f"Error loading saved model files: {exc}")
        return models, scaler, kmeans, False
saved_models, saved_scaler, saved_kmeans, pkl_found = load_saved_models()

DEFAULT_FEATURE_COLUMNS = [
    "TransactionAmt", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "C1", "C2", "D1", "P_emaildomain",
    "ProductCD", "V258", "V308", "C6", "C11", "C13", "C14"
]


def get_feature_columns(scaler):
    if hasattr(scaler, "feature_names_in_"):
        return scaler.feature_names_in_.tolist()
    return DEFAULT_FEATURE_COLUMNS

# ─────────────────────────────────────────────
# PAGE: Introduction
# ─────────────────────────────────────────────
if page == "🏠 Introduction":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Problem Statement")
        st.write("""
        Financial fraud causes billions of dollars in losses annually.
        This system uses a **Hybrid Machine Learning approach** to detect
        fraudulent transactions in real-world e-commerce data.
        """)
        st.subheader("🎯 Objectives")
        st.markdown("""
        - Train and compare **multiple ML models**
        - Handle **severe class imbalance** using SMOTE
        - Use **KMeans clustering** as a hybrid unsupervised feature
        - Optimize with **Optuna hyperparameter tuning**
        - Deploy an interactive **prediction demo**
        """)

    with col2:
        st.subheader("📂 Dataset Overview")
        st.markdown("""
        | Detail | Info |
        |--------|------|
        | Source | Kaggle – IEEE-CIS Fraud Detection |
        | Rows | 590,540 transactions |
        | Columns | 434 features |
        | Target | `isFraud` (binary) |
        | Fraud Rate | ~3.5% (highly imbalanced) |
        """)

        st.subheader("🤖 Models Used")
        st.markdown("""
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. XGBoost *(Optuna-tuned)*
        5. LightGBM
        """)

    st.subheader("🔄 Pipeline Overview")
    steps = ["Data Collection", "Data Cleaning", "Preprocessing",
             "Feature Engineering (KMeans)", "SMOTE", "Model Training",
             "Evaluation", "Deployment"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        col.metric(f"Step {i+1}", step)

# ─────────────────────────────────────────────
# PAGE: Live Prediction Demo
# ─────────────────────────────────────────────
elif page == "🎯 Live Prediction Demo":
    st.subheader("🎯 Live Transaction Fraud Prediction")

    # ── Load real saved models ──────────────────────────────────────
    if not pkl_found:
        st.error("❌ Saved model files (xgb_model.pkl, lgbm_model.pkl, scaler.pkl, kmeans.pkl) not found. "
                 "Please place them in the same directory as app.py.")
        st.stop()

    # Model selector
    selected_model_name = st.selectbox(
        "🤖 Select Model",
        options=list(saved_models.keys()),
        index=0
    )
    demo_model  = saved_models[selected_model_name]
    demo_scaler = saved_scaler
    demo_kmeans = saved_kmeans

    st.success(f"✅ Using **{selected_model_name}** — trained on real IEEE-CIS Vesta dataset")

    # Feature columns the scaler was trained on
    feature_cols = get_feature_columns(demo_scaler)

    st.markdown("Fill in the transaction details below and click **Predict** to get an instant fraud verdict.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 💳 Transaction Info")
        trans_amt  = st.number_input("Transaction Amount (USD)", min_value=0.0,
                                      max_value=50000.0, value=150.0, step=10.0)
        product_cd = st.selectbox("Product Code",
                                   options=[0, 1, 2, 3, 4],
                                   format_func=lambda x: ["W — Wallet", "H — Home",
                                                           "C — Card", "S — Service",
                                                           "R — Retail"][x])
        dist1      = st.number_input("Distance from billing address (dist1)",
                                      min_value=0.0, max_value=10000.0, value=50.0, step=1.0)
        d1         = st.number_input("Days since card account opened (D1)",
                                      min_value=0.0, max_value=640.0, value=100.0, step=1.0)

    with col2:
        st.markdown("##### 🏦 Card Details")
        card1 = st.number_input("Card Issuer ID (card1)", min_value=1000,
                                 max_value=18000, value=9000, step=100)
        card2 = st.number_input("Card Number Group (card2)", min_value=100.0,
                                 max_value=600.0, value=320.0, step=10.0)
        card3 = st.selectbox("Card Country Code (card3)",
                              options=[111.0, 150.0, 185.0],
                              index=1)
        card4 = st.selectbox("Card Network",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: ["Discover", "Mastercard",
                                                      "Visa", "American Express"][x])
        card5 = st.number_input("Card Product Code (card5)", min_value=100.0,
                                 max_value=230.0, value=142.0, step=1.0)
        card6 = st.selectbox("Card Type",
                              options=[0, 1],
                              format_func=lambda x: ["Credit", "Debit"][x])

    with col3:
        st.markdown("##### 📍 Address & Behaviour")
        addr1       = st.number_input("Billing ZIP (addr1)", min_value=0.0,
                                       max_value=500.0, value=299.0, step=1.0)
        addr2       = st.number_input("Billing Country (addr2)", min_value=0.0,
                                       max_value=100.0, value=87.0, step=1.0)
        p_email     = st.selectbox("Purchaser Email Domain",
                                    options=[0, 1, 2, 3, 4],
                                    format_func=lambda x: ["gmail.com", "yahoo.com",
                                                            "hotmail.com", "outlook.com",
                                                            "other"][x])
        c1          = st.slider("How many cards linked to billing address? (C1)", 0, 3000, 1)
        c2          = st.slider("How many addresses linked to card? (C2)",        0, 3000, 1)

    st.markdown("---")
    predict_btn = st.button("🔍  Predict This Transaction", type="primary", use_container_width=True)

    if predict_btn:
        # Build input row: all features default to -999 (same as notebook fillna(-999))
        input_dict = {feature: -999.0 for feature in feature_cols}

        # Overwrite with user-supplied values
        user_values = {
            "TransactionAmt": float(trans_amt),
            "ProductCD":      int(product_cd),
            "card1":          float(card1),
            "card2":          float(card2),
            "card3":          float(card3),
            "card4":          int(card4),
            "card5":          float(card5),
            "card6":          int(card6),
            "addr1":          float(addr1),
            "addr2":          float(addr2),
            "dist1":          float(dist1),
            "C1":             float(c1),
            "C2":             float(c2),
            "D1":             float(d1),
            "P_emaildomain":  int(p_email),
        }
        for key, value in user_values.items():
            if key in input_dict:
                input_dict[key] = value

        input_df = pd.DataFrame([input_dict], columns=feature_cols)
        input_sc = demo_scaler.transform(input_df)
        cluster = demo_kmeans.predict(input_sc)
        input_hybrid = np.hstack([input_sc, cluster.reshape(-1, 1)])

        prob = float(demo_model.predict_proba(input_hybrid)[0][1])
        pred = int(prob >= 0.5)

        st.markdown("---")
        st.subheader("🧾 Prediction Result")

        res_col, gauge_col = st.columns([1, 2])

        with res_col:
            if pred == 1:
                st.error("### 🚨 FRAUDULENT TRANSACTION")
                st.markdown("This transaction has been flagged as **high risk**.")
            else:
                st.success("### ✅ LEGITIMATE TRANSACTION")
                st.markdown("This transaction appears to be **safe**.")

            st.metric("Fraud Probability", f"{prob*100:.2f}%")
            st.metric("Confidence (Safe)", f"{(1-prob)*100:.2f}%")

            if prob < 0.3:
                risk_label, risk_color = "🟢 LOW RISK", "green"
            elif prob < 0.6:
                risk_label, risk_color = "🟡 MEDIUM RISK", "orange"
            else:
                risk_label, risk_color = "🔴 HIGH RISK", "red"
            st.markdown(
                f"**Risk Level:** <span style='color:{risk_color}; font-size:18px'>{risk_label}</span>",
                unsafe_allow_html=True,
            )

        with gauge_col:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.barh(["Risk Meter"], [1 - prob], color="#2ecc71", label="Safe")
            ax.barh(["Risk Meter"], [prob], left=[1 - prob], color="#e74c3c", label="Fraud Risk")
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
            ax.set_title(f"Fraud Probability: {prob*100:.1f}%", fontsize=13, fontweight="bold")
            ax.set_xlabel("Probability")
            ax.legend(loc="lower right", fontsize=9)
            ax.text(
                prob / 2 + (1 - prob),
                0,
                f"{prob*100:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=11,
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("**📋 Input Summary**")
            summary = pd.DataFrame({
                "Field": ["Amount", "Product", "Card Network", "Card Type",
                           "Email Domain", "C1", "C2", "D1", "dist1"],
                "Value": [f"${trans_amt:.2f}",
                           ["W", "H", "C", "S", "R"][product_cd],
                           ["Discover", "Mastercard", "Visa", "Amex"][card4],
                           ["Credit", "Debit"][card6],
                           ["gmail", "yahoo", "hotmail", "outlook", "other"][p_email],
                           c1, c2, d1, dist1],
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px'>"
    "Fraud Detection System · Hybrid ML Approach · IEEE-CIS Vesta Dataset"
    "</div>",
    unsafe_allow_html=True
)