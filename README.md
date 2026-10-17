# Credit Card Default Scoring & Prediction System

An end-to-end Machine Learning pipeline and web application designed for predicting credit card customer default risk. This system features a robust **FastAPI** backend for real-time inference using pre-trained scikit-learn models, accompanied by an interactive **Streamlit** user interface.

## 🚀 Key Features
* **Predictive ML Backend:** Powered by a serialized classification model trained on standard credit risk datasets (such as the UCI Credit Card default dataset).
* **FastAPI REST API:** High-performance, asynchronous endpoints for programmatic risk scoring and server health checks.
* **Interactive Streamlit Frontend:** A clean, user-friendly dashboard allowing users to input financial parameters and visualize risk scores instantly.
* **Production-Ready Structure:** Cleanly separated modular components for API routes, frontend layouts, and machine learning assets.

---

## 🛠️ Project Structure
```text
credit-card-scoring-system/
│
├── api/
│   ├── model/
│   │   ├── credit_risk_model.pkl
│   │   ├── german_credit_model.pkl
│   │   └── predictions_results.csv
│   └── credit_risk_api.py       # FastAPI backend
│
├── frontend/
│   └── streamlit_app.py         # Streamlit user interface dashboard
│
├── data/                        # Dataset files
├── notebooks/                   # Jupyter notebooks for model training & exploration
├── requirements.txt             # Project Python dependencies
├── .gitignore                   # Git ignore configuration
└── README.md                    # Project documentation
---
```
## ⚙️ Local Installation & Running Guide

1. Clone the Repository 
[https://github.com/Samriddha03/credit-card-scoring-system.git](https://github.com/Samriddha03/credit-card-scoring-system.git)
cd credit-card-scoring-system

2. Set Up a Virtual Environment
Bashpython -m venv venv
# On Windows:
.\venv\Scripts\Activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt
---
```
##🚀 Running the Application Locally 

To run the system locally, you will need to open two separate terminal windows (both with your virtual environment activated):
Step A: Start the FastAPI Backend: uvicorn api.credit_risk_api:app --reload

API Root / Health Check: http://127.0.0.1:8000Interactive 

Swagger Docs: http://127.0.0.1:8000/docsInteractive 

ReDoc Docs: http://127.0.0.1:8000/redocStep 

B: Start the Streamlit Frontend 
streamlit run frontend/streamlit_app.py

Streamlit UI Link: http://localhost:8501🔌 
---

## API Endpoints & Data Schema
```
1. Health CheckGET / or /healthResponse: Confirms the backend server status.
2. Risk PredictionPOST /predictExample Request Payload (UCI Credit Card Schema):   

JSON{
  "LIMIT_BAL": 20000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 24,
  "PAY_0": 2,
  "PAY_2": 2,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -2,
  "PAY_6": -2,
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0
}
---
```
Example Response:JSON{
  "risk_prediction": 1,
  "risk_probability": 0.713,
  "risk_label": "Default Risk"
}
---
```
💻 Technologies UsedPython – Core programming languageFastAPI – Backend API frameworkStreamlit – Frontend user interface frameworkScikit-learn – Machine learning model training and inferenceGit & GitHub – Version control and repository management