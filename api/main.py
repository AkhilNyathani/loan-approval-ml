"""
api/main.py
-----------
FastAPI backend for the Loan Approval Prediction System.

Endpoints:
  GET  /          → health check
  POST /predict   → run ML prediction

Run locally:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

from src.predict import predict


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predict whether a loan application will be approved based on applicant details.",
    version="1.0.0",
)

# Allow all origins during development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────────
class LoanApplication(BaseModel):
    Gender:            Literal["Male", "Female"]                    = Field(..., example="Male")
    Married:           Literal["Yes", "No"]                         = Field(..., example="Yes")
    Dependents:        Literal["0", "1", "2", "3+"]                = Field(..., example="0")
    Education:         Literal["Graduate", "Not Graduate"]          = Field(..., example="Graduate")
    Self_Employed:     Literal["Yes", "No"]                         = Field(..., example="No")
    ApplicantIncome:   float = Field(..., gt=0, example=5000,       description="Monthly applicant income")
    CoapplicantIncome: float = Field(..., ge=0, example=1500,       description="Monthly co-applicant income")
    LoanAmount:        float = Field(..., gt=0, example=120,        description="Loan amount in thousands")
    Loan_Amount_Term:  float = Field(..., gt=0, example=360,        description="Term in months")
    Credit_History:    Literal[0, 1]                                = Field(..., example=1)
    Property_Area:     Literal["Urban", "Semiurban", "Rural"]       = Field(..., example="Urban")

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1500,
                "LoanAmount": 120,
                "Loan_Amount_Term": 360,
                "Credit_History": 1,
                "Property_Area": "Urban",
            }
        }


class PredictionResponse(BaseModel):
    prediction:  int
    label:       str
    probability: float
    message:     str


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    """API health check."""
    return {"status": "ok", "service": "Loan Approval Prediction API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_loan(application: LoanApplication):
    """
    Predict loan approval from applicant details.

    - **prediction**: 1 = Approved, 0 = Rejected
    - **label**: Human-readable result
    - **probability**: Model confidence (0-1)
    """
    try:
        raw = application.model_dump()
        result = predict(raw)

        if result["label"] == "Approved":
            msg = "Congratulations! Based on your profile, your loan is likely to be approved."
        else:
            msg = "We're sorry. Based on the provided information, your loan application may not be approved at this time."

        return PredictionResponse(
            prediction  = result["prediction"],
            label       = result["label"],
            probability = result["probability"],
            message     = msg,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Please train the model first. ({e})"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve frontend ───────────────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/app", include_in_schema=False)
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
