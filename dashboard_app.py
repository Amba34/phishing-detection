# dashboard_app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from utils import engineer_features


MODELS_DIR = "models"

# Load artifacts
tfidf = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.joblib")
xgb_model = joblib.load(f"{MODELS_DIR}/xgb_model.joblib")
bert = joblib.load(f"{MODELS_DIR}/bert_embedder.joblib")
numeric_cols = joblib.load(f"{MODELS_DIR}/numeric_feature_names.joblib")
explainer = joblib.load(f"{MODELS_DIR}/xgb_shap_explainer.joblib")


st.set_page_config(page_title="Phishing / Spam Detection Dashboard", layout="wide")
st.title("Real-time Phishing / Spam Detection 🚨")
st.markdown(
    """
This dashboard takes raw email text and classifies it as **Spam** or **Not Spam**.
We use:
- TF-IDF + BERT features  
- XGBoost + Random Forest models  
- SHAP for feature explainability
"""
)

with st.sidebar:
    st.header("Input message")
    sample_text = st.text_area(
        "Paste email / HTTP request body / log line:",
        height=250,
        value=(
            "Dear user, your account has been suspended. "
            "Click http://192.168.0.1/verify now to restore access."
        ),
    )
    run_button = st.button("Analyze")

col_pred, col_shap = st.columns([1, 2])

def make_features_from_text(text: str):
    """Return BERT+numeric feature vector for a single email."""
    df = pd.DataFrame({"text": [text]})
    df = engineer_features(df)
    X_num = df[numeric_cols].values

    # BERT embeddings
    X_bert = bert.encode(list(df["text"].values))
    X_full = np.hstack([X_bert, X_num])
    return X_full, df


if run_button and sample_text.strip():
    with st.spinner("Analyzing message..."):
        X_full, df_feat = make_features_from_text(sample_text)
        proba = xgb_model.predict_proba(X_full)[0]
        pred = int(proba[1] > 0.5)
        score = float(proba[1])

    with col_pred:
        st.subheader("Prediction")

        if pred == 1:
            st.error(f"🚫 Classified as **SPAM** (score={score:.3f})")
        else:
            st.success(f"✅ Classified as **NOT SPAM** (score={score:.3f})")

        st.markdown("**Hand-crafted feature values:**")
        st.dataframe(df_feat[numeric_cols])

    with col_shap:
        st.subheader("SHAP Explanation (BERT + numeric features)")

        # Compute SHAP values
        shap_values = explainer(X_full)

        # SHAP bar plot for the prediction
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        st.caption(
            "Positive SHAP values push the prediction towards *spam*, "
            "negative values towards *not spam*."
        )

else:
    col_pred.info("Paste an email / log text and click **Analyze** to see prediction.")
