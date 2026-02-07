"""
FastAPI Application for Credit Risk Prediction
Production-ready structure
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import schemas
from api.schemas import (
    CreditRiskInput,
    PredictionResponse,
    HealthResponse,
    BatchPredictionInput,
    BatchPredictionResponse
)

from api.middleware import LoggingMiddleware, RequestValidationMiddleware
from src.models.predict import CreditRiskPredictor
from src.utils.helpers import setup_logging, load_config


# Setup logging
logger = setup_logging(
    log_file="logs/api.log",
    log_level="INFO"
)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production API for predicting credit default risk with ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production me restrict karna
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestValidationMiddleware)

# Global predictor instance
predictor: CreditRiskPredictor | None = None


@app.on_event("startup")
async def load_models():
    """Load models at API startup"""
    global predictor

    try:
        logger.info("Loading model artifacts...")
        predictor = CreditRiskPredictor.from_directory("models/saved_models")
        logger.info("All model artifacts loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        predictor = None


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("Shutting down API...")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "Credit Risk Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(input_data: CreditRiskInput):
    """Predict credit default risk for single applicant"""
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

        logger.info("Processing prediction request")

        # Pydantic v2 safe conversion
        input_dict = input_data.model_dump()

        # Correct argument name: return_proba
        result = predictor.predict_single(
            input_dict,
            return_proba=True,
            return_risk_level=True
        )

        # Reason codes
        reason_codes = _generate_reason_codes(input_dict, result["probability_default"])

        response = PredictionResponse(
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            probability=result["probability_default"],
            probability_no_default=result["probability_no_default"],
            probability_default=result["probability_default"],
            risk_level=result["risk_level"],
            reason_codes=reason_codes,
            timestamp=datetime.now().isoformat()
        )

        logger.info(
            f"Prediction={result['prediction']} "
            f"Prob={result['probability_default']:.4f} "
            f"Risk={result['risk_level']}"
        )

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(batch_input: BatchPredictionInput):
    """Batch predictions for multiple applicants"""
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

        logger.info(f"Processing batch prediction request with {len(batch_input.instances)} instances")

        # Convert instances safely (Pydantic v2)
        instances = [inst.model_dump() for inst in batch_input.instances]

        # FIX: predict_batch does NOT accept return_probability
        results = predictor.predict_batch(instances)

        # Add reason codes
        for result, instance in zip(results, instances):
            result["reason_codes"] = _generate_reason_codes(
                instance,
                result.get("probability_default", 0.0)
            )

        # Summary stats
        predictions = [r["prediction"] for r in results]
        total = len(predictions)

        summary = {
            "total_instances": total,
            "predicted_defaults": int(sum(predictions)),
            "predicted_no_defaults": int(total - sum(predictions)),
            "default_rate": float(sum(predictions) / total) if total > 0 else 0.0
        }

        response = BatchPredictionResponse(
            predictions=results,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )

        logger.info(
            f"Batch prediction complete: "
            f"{summary['predicted_defaults']} defaults, "
            f"{summary['predicted_no_defaults']} no defaults"
        )

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


def _generate_reason_codes(features: dict, probability: float, top_n: int = 4) -> list:
    """Generate reason codes for prediction explanation"""
    reason_codes = []

    # Severe delinquency
    if features.get("late_90", 0) > 0:
        count = int(features["late_90"])
        reason_codes.append(f"History of severe delinquency (90+ days late: {count} times)")

    # Credit utilization
    rev_util = features.get("rev_util", 0)
    if rev_util > 0.8:
        reason_codes.append(f"Very high credit utilization ({rev_util:.1%})")
    elif rev_util > 0.5 and probability > 0.3:
        reason_codes.append(f"High credit utilization ({rev_util:.1%})")

    # Debt ratio
    debt_ratio = features.get("debt_ratio", 0)
    if debt_ratio > 0.5:
        reason_codes.append(f"High debt-to-income ratio ({debt_ratio:.2f})")
    elif debt_ratio > 0.35 and probability > 0.3:
        reason_codes.append(f"Elevated debt-to-income ratio ({debt_ratio:.2f})")

    # Recent late payments
    late_30_59 = features.get("late_30_59", 0)
    late_60_89 = features.get("late_60_89", 0)
    total_recent_late = late_30_59 + late_60_89

    if total_recent_late > 2:
        reason_codes.append(f"Multiple recent late payments ({int(total_recent_late)} instances)")

    # Income
    monthly_inc = features.get("monthly_inc", 0)
    if monthly_inc < 2000 and probability > 0.3:
        reason_codes.append(f"Low monthly income (${monthly_inc:,.0f})")

    # Age
    age = features.get("age", 0)
    if age < 25 and probability > 0.4:
        reason_codes.append(f"Limited credit history (age: {int(age)})")

    # Positive factors
    if probability < 0.3:
        if features.get("late_90", 0) == 0 and total_recent_late == 0:
            reason_codes.append("Clean payment history with no delinquencies")

        if rev_util < 0.3:
            reason_codes.append(f"Responsible credit utilization ({rev_util:.1%})")

        if debt_ratio < 0.3:
            reason_codes.append(f"Low debt burden (debt ratio: {debt_ratio:.2f})")

    # fallback
    if not reason_codes:
        reason_codes.append("Overall credit profile indicates elevated risk" if probability > 0.5 else "Overall credit profile indicates low risk")

    return reason_codes[:top_n]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


def start_server():
    """Start server using config file"""
    import uvicorn

    try:
        config = load_config()
        api_config = config.get("api", {})
    except Exception:
        api_config = {}

    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    workers = api_config.get("workers", 1)
    log_level = api_config.get("log_level", "info")

    logger.info(f"Starting API server on {host}:{port}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level.lower(),
        reload=False
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
