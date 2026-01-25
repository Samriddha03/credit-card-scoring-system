# Credit Card Scoring System API

This project provides a **Credit Card Scoring System API** using **FastAPI**. The API allows you to predict credit risk scores using pre-trained machine learning models.

---

## Features

- Predict credit risk for individual customers.
- Supports multiple models (`credit_risk_model.pkl` and `german_credit_model.pkl`).
- FastAPI-based REST API with Swagger documentation.
- Ready for deployment using **Docker**, **Railway**, or similar platforms.

---

## Project Structure

api/
├─ model/
│ ├─ credit_risk_model.pkl
│ ├─ german_credit_model.pkl
│ └─ predictions_results.csv
├─ credit_risk_api.py
├─ requirements.txt
├─ Dockerfile
└─ runtime.txt
data/
notebooks/
.gitignore
README.md

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Samriddha03/credit-card-scoring-system.git
cd credit-card-scoring-system

Install dependencies:

pip install -r api/requirements.txt

Running the API Locally
uvicorn api.credit_risk_api:app --reload


API will run at: http://127.0.0.1:8000

Swagger documentation: http://127.0.0.1:8000/docs

ReDoc documentation: http://127.0.0.1:8000/redoc

Deployment

The project is ready for deployment on platforms like:

Railway

Render

Heroku

Koyeb

Example Docker deployment:

docker build -t credit-card-api .
docker run -p 8000:8000 credit-card-api

Usage

Once deployed, open the Swagger UI /docs to test your API endpoints. You can send requests and see predicted credit risk scores directly from the live API.