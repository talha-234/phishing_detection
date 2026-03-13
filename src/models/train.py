import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from src.data.loader import load_config, load_and_split
from src.features.extractor import URLFeatureExtractor


def main():
    try:
        config = load_config()
        extractor = URLFeatureExtractor()

        print("Loading & splitting data...")
        train_df, test_df, y_train, y_test = load_and_split(config)

        print(f"→ Train samples: {len(train_df):,d} | Test samples: {len(test_df):,d}")

        print("Extracting features...")
        X_train = extractor.transform(train_df['url'].tolist())
        X_test  = extractor.transform(test_df['url'].tolist())

        feature_names = extractor.get_feature_names()
        X_train = X_train[feature_names]
        X_test  = X_test[feature_names]

        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            random_state=config['model']['random_state'],
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)

        print("Evaluating...")
        preds = model.predict(X_test_scaled)
        print("\n" + classification_report(y_test, preds, digits=4))
        print(f"Macro F1-score: {f1_score(y_test, preds, average='macro'):.4f}")

        print("\nSaving model artifacts...")
        joblib.dump(model,        "models/phishing_model.joblib")
        joblib.dump(scaler,       "models/scaler.joblib")
        joblib.dump(feature_names, "models/feature_names.joblib")

        print("\nTraining finished successfully ✓")
    except Exception as e:
        print(f"\nERROR during training:\n{str(e)}")
        raise


if __name__ == "__main__":
    main()