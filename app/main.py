import os
import sys
import logging
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List

# 1. Environment Variables Read Karein
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

# 2. Standard Output Logging Setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("fastapi_service")

# Global model object
model = None

# 3. Fail-Fast Model Loader at Startup (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading model artifact from: '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"CRITICAL ERROR: Model file not found at '{MODEL_PATH}'. Failing fast!")
        sys.exit(1)
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model successfully loaded into memory once at startup.")
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to load model artifact. Details: {e}")
        sys.exit(1)
        
    yield
    logger.info("Shutting down API service...")

# FastAPI App Instance
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="MLOps Assignment 2 - FastAPI Model Serving",
    version="1.0",
    lifespan=lifespan
)

# 4. Pydantic Request Body Validation Schema
class CancerInput(BaseModel):
    features: List[float] = Field(
        ..., 
        description="List of exactly 30 numerical features matching the breast cancer dataset."
    )

    @field_validator('features')
    @classmethod
    def check_30_features(cls, v):
        if len(v) != 30:
            raise ValueError(f"Expected exactly 30 feature values, but received {len(v)}.")
        return v

# 5. API Endpoints

@app.get("/")
def read_root():
    return {"message": "Breast Cancer API Service is running."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.get("/stats")
def get_stats():
    # Milestone 4 tak placeholder endpoint
    return {
        "status": "placeholder",
        "message": "Stats endpoint placeholder for Milestone 4"
    }

@app.post("/predict")
def predict(payload: CancerInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Original dataset feature names in exact order
        feature_names = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
        
        df = pd.DataFrame([payload.features], columns=feature_names)
        
        # Class probabilities: index 0 = Malignant class
        probabilities = model.predict_proba(df)[0]
        malignant_prob = float(probabilities[0])
        
        is_malignant = malignant_prob >= DECISION_THRESHOLD

        logger.info(f"Prediction requested | Malignant Prob: {malignant_prob:.4f} | Decision Threshold: {DECISION_THRESHOLD}")

        return {
            "malignant_probability": malignant_prob,
            "decision_threshold": DECISION_THRESHOLD,
            "is_malignant": is_malignant,
            "prediction": "Malignant" if is_malignant else "Benign"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))