# train_models.py
import os
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import shap

from utils import load_dataset, engineer_features, build_tfidf_vectorizer, BertEmbedder


DATA_PATH = "data/emails.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    return acc, f1


def main():
    # 1. Load + feature engineering
    df = load_dataset(DATA_PATH)
    df = engineer_features(df)

    X_text = df["text"].values
    y = df["label_binary"].values  # Use binary labels (1=spam, 0=not spam)
    numeric_cols = ["num_urls", "num_digits", "text_len", "has_ip_url"]
    X_num = df[numeric_cols].values

    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
    )

    #############################
    # 2. Baseline - TF-IDF + LR #
    #############################

    print("Training TF-IDF + Logistic Regression baseline...")
    tfidf = build_tfidf_vectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    # concatenate small numeric features
    from scipy.sparse import hstack
    X_train_base = hstack([X_train_tfidf, X_train_num])
    X_test_base = hstack([X_test_tfidf, X_test_num])

    lr = LogisticRegression(max_iter=200, n_jobs=-1)
    lr.fit(X_train_base, y_train)
    y_pred_lr = lr.predict(X_test_base)
    base_acc, base_f1 = print_metrics("Baseline TF-IDF + Logistic Regression", y_test, y_pred_lr)

    ###############################################
    # 3. Random Forest on TF-IDF + numeric feats  #
    ###############################################

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train_base, y_train)
    y_pred_rf = rf.predict(X_test_base)
    rf_acc, rf_f1 = print_metrics("Random Forest (TF-IDF + hand-crafted)", y_test, y_pred_rf)

    #######################################################
    # 4. BERT embeddings + XGBoost (tree + SHAP friendly) #
    #######################################################

    print("\nComputing BERT embeddings (this can take a bit)...")
    bert = BertEmbedder()

    X_train_bert = bert.encode(list(X_train_text))
    X_test_bert = bert.encode(list(X_test_text))

    # concatenate numeric features to BERT
    X_train_bert_full = np.hstack([X_train_bert, X_train_num])
    X_test_bert_full = np.hstack([X_test_bert, X_test_num])

    print("\nTraining XGBoost on BERT + numeric features...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist"
    )

    xgb_model.fit(X_train_bert_full, y_train)
    y_pred_xgb = xgb_model.predict(X_test_bert_full)
    xgb_acc, xgb_f1 = print_metrics("XGBoost (BERT + hand-crafted)", y_test, y_pred_xgb)

    print("\nDetailed classification report for best model (XGBoost):")
    print(classification_report(y_test, y_pred_xgb, digits=4))

    # 5. Show improvement over baseline (for your resume bullet)
    acc_improvement = (xgb_acc - base_acc) * 100
    print(f"\nDetection accuracy improvement over baseline LR: {acc_improvement:.2f}%")

    #####################################
    # 6. Save models & preprocess stuff #
    #####################################

    print("\nSaving models and preprocessors to 'models/' ...")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(lr, MODELS_DIR / "baseline_lr.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    joblib.dump(bert, MODELS_DIR / "bert_embedder.joblib")  # wrapper object
    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(numeric_cols, MODELS_DIR / "numeric_feature_names.joblib")

    ################################
    # 7. SHAP explainer for XGBoost
    ################################

    print("\nBuilding SHAP explainer for XGBoost...")
    # Use a small background subset to keep it light
    background = shap.sample(X_train_bert_full, 200)
    explainer = shap.TreeExplainer(xgb_model, data=background)
    joblib.dump(explainer, MODELS_DIR / "xgb_shap_explainer.joblib")

    print("\nAll done! You can now run `streamlit run dashboard_app.py`.")


if __name__ == "__main__":
    main()
