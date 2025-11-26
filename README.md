# Phishing/Spam Email Detector

A machine learning project that detects spam and phishing emails using TF-IDF, BERT embeddings, and XGBoost, with an interactive Streamlit dashboard for real-time predictions.

## Features

- **Multiple ML Models**: TF-IDF + Logistic Regression, Random Forest, XGBoost
- **Advanced Features**: Hand-crafted features (URL count, digit count, text length, IP detection) + BERT embeddings
- **Interactive Dashboard**: Streamlit-based web interface for real-time email classification
- **Explainability**: SHAP values to understand model predictions
- **120+ Training Examples**: Diverse dataset with spam and legitimate emails

## Project Structure

```
phishing-detector/
│
├── data/
│   └── emails.csv          # Dataset with columns: text, label
├── models/                 # Trained models (auto-generated)
│   ├── tfidf_vectorizer.joblib
│   ├── xgb_model.joblib
│   ├── bert_embedder.joblib
│   └── ...
├── train_models.py         # Training script
├── dashboard_app.py        # Streamlit dashboard
├── utils.py                # Helper functions
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd phishing-detector
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install pandas scikit-learn xgboost transformers torch joblib shap streamlit matplotlib
```

## Usage

### Train the Models

```bash
python train_models.py
```

This will:
- Load the dataset from `data/emails.csv`
- Engineer features and create embeddings
- Train multiple models (TF-IDF + LR, Random Forest, XGBoost)
- Save trained models to the `models/` directory
- Display accuracy and F1-score metrics

### Run the Dashboard

```bash
streamlit run dashboard_app.py
```

Then open your browser to `http://localhost:8501` to:
- Paste email text for classification
- See predictions (Spam vs Not Spam)
- View SHAP explanations showing which features influenced the prediction
- Analyze hand-crafted feature values

## Dataset

The `data/emails.csv` file contains 120 labeled examples:
- **70 spam emails**: Phishing attempts, scams, fake prizes, fraudulent messages
- **50 legitimate emails**: Work communications, order confirmations, notifications

**Format**: CSV with columns `text` and `label` (values: "spam" or "not spam")

## Models

1. **Baseline**: TF-IDF + Logistic Regression
2. **Random Forest**: With combined TF-IDF and numeric features
3. **XGBoost**: With BERT embeddings + hand-crafted features (primary model for dashboard)

## Features Engineered

- Number of URLs in email
- Number of digits
- Text length
- Presence of IP addresses in URLs
- BERT sentence embeddings (768 dimensions)

## Results

Training metrics are displayed after running `train_models.py`. Typical performance:
- Accuracy: ~95%+
- F1-Score: ~95%+

*(Results may vary based on train/test split)*

## Author

Created as a demonstration project for email spam/phishing detection.
