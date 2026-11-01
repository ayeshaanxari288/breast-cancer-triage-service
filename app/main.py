import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("fastapi_service")

app = FastAPI(title="Breast Cancer Prediction API")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
REDIS_HOST = os.getenv("REDIS_HOST", None)
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

model = None
redis_client = None

@app.on_event("startup")
def load_model_and_redis():
    global model, redis_client
    # Load Model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model successfully loaded from {MODEL_PATH}")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")

    # Connect to Redis if REDIS_HOST is set
    if REDIS_HOST:
        try:
            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_timeout=2)
            redis_client.ping()
            logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. API will run without counter.")
            redis_client = None

class PatientData(BaseModel):
    features: list[float]

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "threshold": DECISION_THRESHOLD}

@app.get("/stats")
def get_stats():
    # Return count from Redis, or 0 if unreachable
    if redis_client:
        try:
            count = redis_client.get("predict_counter")
            return {"prediction_count": int(count) if count else 0}
        except Exception as e:
            logger.warning(f"Failed to fetch stats from Redis: {e}")
    return {"prediction_count": 0, "note": "Redis counter unavailable"}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(data.features)}")

    try:
        df = pd.DataFrame([data.features])
        probs = model.predict_proba(df)[0]
        malignant_prob = float(probs[1])
        is_malignant = malignant_prob >= DECISION_THRESHOLD

        # Increment Redis Counter (Fail-safe: prediction won't break if Redis fails)
        if redis_client:
            try:
                redis_client.incr("predict_counter")
            except Exception as e:
                logger.warning(f"Failed to increment Redis counter: {e}")

        return {
            "malignant_probability": malignant_prob,
            "decision_threshold": DECISION_THRESHOLD,
            "is_malignant": is_malignant,
            "prediction": "Malignant" if is_malignant else "Benign"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))