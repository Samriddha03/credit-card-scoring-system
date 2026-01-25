# Credit Card Scoring System API (ML + FastAPI)

##  Project Overview

The Credit Card Scoring System API predicts credit risk scores for customers using pre-trained machine learning models.

# This system integrates:

## Machine Learning Models: 

- credit_risk_model.pkl
- german_credit_model.pkl

* FastAPI Backend: Exposes REST API endpoints for predictions

* Deployment Ready: Compatible with Docker, Railway, Render, Heroku, or Koyeb

## Users can send customer data through API requests and receive real-time credit risk predictions.

# System Architecture:
Frontend / Client (optional)
        ↓
   FastAPI Backend
        ↓
 Machine Learning Models (.pkl)

## Project Structure
credit-card-scoring-system/
├── api/
│   ├── model/
│   │   ├── credit_risk_model.pkl
│   │   ├── german_credit_model.pkl
│   │   └── predictions_results.csv
│   ├── credit_risk_api.py       # FastAPI backend
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Docker configuration
│   └── runtime.txt              # Deployment runtime configuration
├── data/                        # Dataset (if any)
├── notebooks/                   # Jupyter notebooks for model training / exploration
├── .gitignore                   # Git ignore file
└── README.md                    # 

## Project documentation

Dataset & Model Training

- Pre-trained models are included in api/model/
- Models are trained on standard credit risk datasets
- Predictions are based on customer financial features
- Sample prediction results are stored in predictions_results.csv

⚠️ This implementation is for learning, demonstration, and portfolio purposes.

* Backend API (FastAPI)
Available Endpoints
* POST /predict – Predict credit risk score for a customer
* GET /health – Check server health

## Example Request
```
{
  "Age": 35,
  "Credit_amount": 12000,
  "Duration": 36,
  "Sex": "female",
  "Housing": "rent",
  "Saving_accounts": "moderate",
  "Checking_account": "little",
  "Purpose": "education"
}
```

## Example Response
```
{
  "risk_prediction": 1,
  "risk_probability": 0.605,
  "risk_label": "Bad"
}
```

## Running the Project Locally

1️⃣ Start the Backend Server
uvicorn api.credit_risk_api:app --reload
Backend will run at: http://127.0.0.1:8000

Swagger docs: http://127.0.0.1:8000/docs

ReDoc docs: http://127.0.0.1:8000/redoc

Deployment Docker Deployment
## Build Docker image
docker build -t credit-card-api .

##  Run Docker container
docker run -p 8000:8000 credit-card-api

## Supported Platforms

* Railway
* Render
* Heroku
* Koyeb

## The project is Docker-ready, making deployment simple and consistent.

# Technologies Used

1. Python
2. FastAPI
3. Scikit-learn
4. Docker
5. Git & GitHub

# Deployment Platforms: 
Railway, Render, Heroku, Koyeb

# Project Highlights

-REST API for credit risk prediction
Pre-trained ML models for instant predictions
- Fully Dockerized and deployment-ready
- Clean and professional API design
Portfolio-ready, suitable for interviews

# Disclaimer
This project is created for educational and demonstration purposes only.
It should not be used in real-world financial systems without proper validation, security, and testing.