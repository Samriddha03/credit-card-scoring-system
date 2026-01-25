from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
#from pathlib import Path

app = FastAPI(title="Credit Risk Prediction API")

#BASE_DIR = Path(__file__).resolve().parent
model = joblib.load("api/model/credit_risk_model.pkl")




# Input schema (friendly names for API)
class CreditInput(BaseModel):
    Age: int
    Job: int
    Credit_amount: int
    Duration: int
    Sex: str
    Housing: str
    Saving_accounts: str
    Checking_account: str
    Purpose: str


@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict_risk(data: CreditInput):

    # 🔥 MAP API FIELDS → MODEL FIELDS (THIS FIXES EVERYTHING)
    input_df = pd.DataFrame([{
        "Age": data.Age,
        "Job": data.Job,
        "Credit amount": data.Credit_amount,
        "Duration": data.Duration,
        "Sex": data.Sex,
        "Housing": data.Housing,
        "Saving accounts": data.Saving_accounts,
        "Checking account": data.Checking_account,
        "Purpose": data.Purpose
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "risk_prediction": int(prediction),
        "risk_probability": round(float(probability), 3),
        "risk_label": "Bad" if prediction == 1 else "Good"
    }
