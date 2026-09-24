"""MindCare Backend API - FastAPI Inference Server.

Exposes REST endpoints for health checks and text classification inference.
Connects with the local Deep Learning Bi-LSTM model and rule-based response manager.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is on Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.config import (
    APP_NAME,
    APP_SUBTITLE,
    CLASS_LABELS,
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    RESPONSE_DIR,
    TOKENIZER_PATH,
)
from model.model_config import ModelConfig
from model.model_loader import load_model, load_tokenizer
from model.predictor import ModelNotReadyError, Predictor
from response.response_loader import ResponseLoader
from response.response_manager import ResponseManager
from utils.helpers import check_system_status, validate_user_input
from utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=f"{APP_NAME} Inference API",
    description=APP_SUBTITLE,
    version="1.0.0",
)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://*.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits local Next.js dev and Vercel preview
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline singletons
_model_config = ModelConfig()
_model_obj, _model_err = load_model(MODEL_PATH)
_tok_obj, _tok_err = load_tokenizer(TOKENIZER_PATH)
_predictor = Predictor(model=_model_obj, tokenizer=_tok_obj, config=_model_config)
_response_loader = ResponseLoader(response_dir=RESPONSE_DIR)
_response_manager = ResponseManager(loader=_response_loader)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    last_response: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    is_high_risk: bool = False
    is_ood: bool = False
    latency_ms: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class StatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_ready: bool
    model_error: Optional[str] = None
    architecture: str
    task: str
    num_classes: int
    classes: List[str]
    confidence_threshold: float


@app.get("/health", response_model=StatusResponse)
@app.get("/api/health", response_model=StatusResponse)
def health_check():
    """Return health and model status."""
    is_ready = _predictor.is_ready
    return StatusResponse(
        status="ready" if is_ready else "model_unavailable",
        model_ready=is_ready,
        model_error=_model_err or _tok_err,
        architecture=_model_config.architecture,
        task=_model_config.task,
        num_classes=_model_config.num_classes,
        classes=_model_config.class_labels,
        confidence_threshold=_model_config.confidence_threshold,
    )


@app.post("/api/chat", response_model=ChatResponse)
@app.post("/predict", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """Classify user text and return safe, controlled response."""
    is_valid, validation_result = validate_user_input(payload.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=validation_result)

    clean_text = validation_result

    if not _predictor.is_ready:
        return ChatResponse(
            response=(
                "The application interface is operational, but the Deep Learning model weights "
                "are currently offline. Please ensure the model artifact ('model_final.h5') is loaded."
            ),
            predicted_class="Unavailable",
            confidence=0.0,
            is_high_risk=False,
            is_ood=False,
        )

    try:
        pred_result = _predictor.predict(clean_text)
        predicted_class = pred_result["predicted_class"]
        confidence = pred_result["confidence"]

        mapped = _response_manager.get_response(
            predicted_class=predicted_class,
            confidence=confidence,
            last_response_text=payload.last_response,
        )

        return ChatResponse(
            response=mapped["text"],
            predicted_class=predicted_class,
            confidence=confidence,
            is_high_risk=mapped["is_high_risk"],
            is_ood=pred_result["is_ood"],
            latency_ms=pred_result["latency_ms"],
            probabilities=pred_result["probabilities"],
        )
    except Exception as exc:
        logger.error("Inference exception: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your message.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
