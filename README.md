# PhishCheck – Phishing URL Detection System

-A lightweight, local phishing URL classifier using supervised machine learning. The system extracts lexical, structural, and domain-based features from URLs and applies a **Random Forest** classifier to predict phishing probability.

-Random Forest-based phishing URL classifier with lexical &amp; structural feature extraction. Served via FastAPI, zero-dependency HTML/JS frontend.
## Architecture Overview

- **Frontend**: Single-file HTML + CSS + vanilla JavaScript (no build step, no npm)
  - Real-time form submission via `fetch`
  - Confidence score, binary classification, risk factor list
  - Dark/light mode toggle with localStorage persistence
  - Pure CSS loading spinner

- **Backend**: FastAPI (ASGI) REST API
  - Endpoint: `POST /api/v1/detect`
  - Model: scikit-learn `RandomForestClassifier` (persisted via joblib)
  - Feature engineering: length, entropy, subdomain count, suspicious TLDs, login keywords, query parameters, percent encoding, etc.
  - CORS enabled for local HTML client

- **Model Training Pipeline**:
  - Dataset: balanced phishing & legitimate URLs (custom CSV or public datasets)
  - Features: 13 hand-engineered numerical & binary features
  - Preprocessing: StandardScaler
  - Evaluation: stratified train/test split, classification report, macro F1-score

## Model Performance (example on small test set)
precision    recall  f1-score   support
0       1.00      1.00      1.00         1
1       1.00      1.00      1.00         1
accuracy                           1.00         2
macro avg       1.00      1.00      1.00         2


(Note: performance improves significantly with larger, real-world datasets)

## Project Structure

phish-project/
├── backend/
│   ├── config/               # YAML config (dataset paths, hyperparameters)
│   ├── data/
│   │   └── raw/              # phishing.csv, legitimate.csv
│   ├── models/               # trained artifacts (.joblib)
│   ├── src/
│   │   ├── api/
│   │   │   ├── main.py       # FastAPI app + endpoints
│   │   │   └── predictor.py  # model loading & inference
│   │   ├── data/
│   │   │   └── loader.py
│   │   ├── features/
│   │   │   └── extractor.py  # URL feature engineering
│   │   └── models/
│   │       └── train.py      # training pipeline
│   └── requirements.txt
└── frontend/
└── phish-check.html      # single-file client (HTML + CSS + JS)


## Quick Start

1. **Backend setup**


cd backend
pip install -r requirements.txt

# Prepare small test datasets in data/raw/
py src/models/train.py

## Start server (with auto-reload)
py -3.12 -m uvicorn src.api.main:app --reload

2.Frontend usage


Double-click frontend/phish-check.html
Enter URL → submit
View prediction, confidence, and risk factors

##Feature Engineering Details

#Feature,Type,Description
-url_length,numeric,Total characters in URL
-num_dots,numeric,Count of '.' characters
-num_slashes,numeric,Count of '/' characters
-num_query_params,numeric,Number of query string parameters
-has_ip_address,binary,Domain is an IP address
-url_entropy,numeric,Shannon entropy of URL string
-num_subdomains,numeric,Count of subdomains
-is_https,binary,Uses HTTPS scheme
-domain_length,numeric,Length of registered domain
-has_suspicious_tld,binary,TLD in known phishing list
-path_length,numeric,Length of path component
-has_login_keyword,binary,Contains login-related keywords
-percent_encoded_chars,numeric,Count of %XX encodings






##Future Improvements (planned / welcome contributions)

-Ensemble methods (XGBoost, LightGBM)
-Deep learning approach (LSTM / Transformer on URL tokens)
-WHOIS-based features (domain age, registrar)
-Model drift detection & periodic retraining
-Browser extension version
-Larger public dataset integration (PhishTank, URLhaus, etc.)

##License

-Feel free to fork, modify, and use in academic or personal projects.


##Acknowledgments

-scikit-learn for the classifier implementation
-FastAPI for rapid API development
-Unsplash / public domain images for UI background






