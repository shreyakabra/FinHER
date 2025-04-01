from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load Trained Credit Model
model = joblib.load("credit_model.pkl")

def calculate_credit_score(user_data):
    input_data = np.array([[
        user_data['loan_amount'],
        user_data['business_age'],
        user_data['monthly_revenue'],
        np.random.rand()  # Random feature for now
    ]])
    return int(model.predict(input_data)[0])  # Predict credit score

# User registration model
class UserRegistration(BaseModel):
    name: str
    email: str
    phone: str
    business_name: str
    revenue: float

# Loan application model
class LoanApplication(BaseModel):
    user_id: int
    loan_amount: float
    business_age: int
    monthly_revenue: float

@app.get("/")
def home():
    return {"message": "Welcome to FinHER API"}

@app.post("/register")
def register_user(user: UserRegistration):
    # Store user in DB (mock for now)
    return {"message": "User registered successfully", "user": user.dict()}

@app.post("/apply-loan")
def apply_loan(application: LoanApplication):
    credit_score = calculate_credit_score(application.dict())  # AI-based scoring
    return {"loan_status": "Processing", "credit_score": credit_score}
