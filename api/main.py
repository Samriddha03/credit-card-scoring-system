from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from api.transformer import CreditFeatureAdder

app = FastAPI(title="Taiwan Credit Default Prediction API")

# Load saved bundle dictionary
bundle = joblib.load("api/model/credit_risk_model.pkl")
feature_adder = bundle['feature_adder']
model_pipeline = bundle['pipeline']

class CreditInput(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

@app.post("/predict")
def predict_risk(data: CreditInput):
    input_df = pd.DataFrame([data.dict()])
    
    # 1. Transform engineered features
    engineered_df = feature_adder.transform(input_df)

    # 2. Predict using model pipeline
    prediction = model_pipeline.predict(engineered_df)[0]
    probability = model_pipeline.predict_proba(engineered_df)[0][1]

    return {
        "risk_prediction": int(prediction),
        "risk_probability": round(float(probability), 3),
        "risk_label": "Default Risk" if prediction == 1 else "No Default Risk"
    }