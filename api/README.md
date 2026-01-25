# Credit Card Scoring System API (ML + FastAPI)

## Project Overview

This project provides a **Credit Card Scoring System API** that predicts the credit risk score of individual customers using pre-trained machine learning models.  

The system integrates:

- **Machine Learning models** (`credit_risk_model.pkl`, `german_credit_model.pkl`) for credit risk prediction  
- **FastAPI** backend to expose a REST API  
- Ready for deployment with Docker or platforms like **Railway**, **Render**, **Heroku**, or **Koyeb**

Users can send customer data via API requests and receive real-time credit risk predictions.

---

## System Architecture

Frontend / Client (optional)  
        ↓  
FastAPI Backend (REST API)  
        ↓  
Machine Learning Models (.pkl)

---

## Project Structure

credit-card-scoring-system/
├── api/
│   ├── model/
│   │   ├── credit_risk_model.pkl
│   │   ├── german_credit_model.pkl
│   │   └── predictions_results.csv
│   ├── credit_risk_api.py        # FastAPI backend
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile                # Docker configuration
│   └── runtime.txt               # Deployment runtime configuration
├── data/                         # Dataset (if any)
├── notebooks/                     # Jupyter notebooks for model training / exploration
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation

---

## Dataset & Model Training

Pre-trained machine learning models are included in the api/model/ folder
Models were trained on credit risk datasets
Predictions are based on customer financial features
Results are saved in predictions_results.csv (example output)
This implementation is primarily for learning, demonstration, and portfolio purposes.

---

## Backend API (FastAPI)
Available Endpoints

POST /predict – Predicts the credit risk score of a customer
GET /health – Health check endpoint

Example Request
{
  "Age": 42,
  "Job": 2,
  "Credit_amount": 12000,
  "Duration": 36,
  "Sex": "female",
  "Housing": "rent",
  "Saving_accounts": "moderate",
  "Checking_account": "little",
  "Purpose": "education"
}

Example Response
{
  "risk_prediction": 1,
  "risk_probability": 0.605,
  "risk_label": "Bad"
}

---

## Running the Project Locally
Start the Backend Server
uvicorn api.credit_risk_api:app --reload
Backend will run at: http://127.0.0.1:8000
Swagger documentation: http://127.0.0.1:8000/docs
ReDoc documentation: http://127.0.0.1:8000/redoc

---

## Deployment

The API is Docker-ready:
docker build -t credit-card-api .
docker run -p 8000:8000 credit-card-api
Can be deployed on platforms like: Railway, Render, Heroku, or Koyeb

---

## Technologies Used

Python
FastAPI
Scikit-learn
Docker
Git & GitHub
Deployment platforms: Railway, Render, Heroku, Koyeb

---

## Project Highlights

REST API for credit risk prediction
Pre-trained ML models for instant predictions
Dockerized and deployment-ready
Clean, professional API design
Portfolio-ready project suitable for interviews

---

## Disclaimer

This project is created for educational and demonstration purposes only.
It should not be used in real-world financial systems without further validation, security, and testing.





