import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")


def predict_customer(data: dict) -> dict:
    df = pd.DataFrame([data])

    proba = model.predict_proba(df)[0][1]

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": int(proba > 0.5)
    }