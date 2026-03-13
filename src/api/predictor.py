import joblib
import pandas as pd
from typing import Dict, List

from src.features.extractor import URLFeatureExtractor


class PhishingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = None
        self.extractor = URLFeatureExtractor()

        self.model_path = "models/phishing_model.joblib"
        self.scaler_path = "models/scaler.joblib"
        self.names_path = "models/feature_names.joblib"

    def _load_if_needed(self):
        if self.model is None:
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.feature_names = joblib.load(self.names_path)
                if not isinstance(self.feature_names, list):
                    self.feature_names = list(self.feature_names)
            except FileNotFoundError:
                raise RuntimeError(
                    "Model files not found.\n"
                    "Please run training first:\n"
                    "   python src/models/train.py"
                )
            except Exception as e:
                raise RuntimeError(f"Cannot load model: {str(e)}")

    def predict(self, url: str) -> Dict:
        self._load_if_needed()

        features_dict = self.extractor.extract(url)

        df = pd.DataFrame([features_dict])
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            for col in missing:
                df[col] = 0.0

        X = df[self.feature_names]

        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]

        is_phishing = bool(pred)
        confidence = float(proba[pred])

        risk_factors = []
        if is_phishing:
            importances = self.model.feature_importances_
            for feat, imp in zip(self.feature_names, importances):
                if imp > 0.04 and features_dict.get(feat, 0) != 0:
                    risk_factors.append(feat)

        return {
            "url": url,
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "probability_phishing": round(float(proba[1]), 4),
            "risk_factors": risk_factors
        }