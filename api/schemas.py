"""
Pydantic schemas for API request/response validation
Defines data models for all API endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime


class CreditRiskInput(BaseModel):
    """
    Input schema for credit risk prediction
    
    All features required for making a prediction
    """
    rev_util: float = Field(
        ..., 
        ge=0, 
        description="Revolving credit utilization ratio (balance/limit)",
        example=0.45
    )
    age: int = Field(
        ..., 
        ge=18, 
        le=120, 
        description="Age of borrower in years",
        example=35
    )
    late_30_59: int = Field(
        ..., 
        ge=0, 
        description="Number of times 30-59 days past due in last 2 years",
        example=0
    )
    debt_ratio: float = Field(
        ..., 
        ge=0, 
        description="Monthly debt payments / monthly income",
        example=0.35
    )
    monthly_inc: float = Field(
        ..., 
        ge=0, 
        description="Monthly income in dollars",
        example=5000.0
    )
    open_credit: int = Field(
        ..., 
        ge=0, 
        description="Number of open credit lines and loans",
        example=8
    )
    late_90: int = Field(
        ..., 
        ge=0, 
        description="Number of times 90+ days late in last 2 years",
        example=0
    )
    real_estate: int = Field(
        ..., 
        ge=0, 
        description="Number of mortgage and real estate loans",
        example=1
    )
    late_60_89: int = Field(
        ..., 
        ge=0, 
        description="Number of times 60-89 days past due in last 2 years",
        example=0
    )
    dependents: int = Field(
        ..., 
        ge=0, 
        description="Number of dependents excluding self",
        example=2
    )
    
    @validator('rev_util')
    def validate_utilization(cls, v):
        """Warn about very high utilization"""
        if v > 2.0:
            raise ValueError("Utilization ratio seems unusually high (>200%)")
        return v
    
    @validator('debt_ratio')
    def validate_debt_ratio(cls, v):
        """Warn about extreme debt ratios"""
        if v > 10.0:
            raise ValueError("Debt ratio seems unusually high (>1000%)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "rev_util": 0.45,
                "age": 35,
                "late_30_59": 0,
                "debt_ratio": 0.35,
                "monthly_inc": 5000.0,
                "open_credit": 8,
                "late_90": 0,
                "real_estate": 1,
                "late_60_89": 0,
                "dependents": 2
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for single prediction
    
    Contains prediction, probabilities, risk level, and explanations
    """
    prediction: int = Field(
        ..., 
        description="Predicted class: 0 = No Default, 1 = Default",
        example=0
    )
    prediction_label: str = Field(
        ...,
        description="Human-readable prediction label",
        example="No Default"
    )
    probability: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Probability of default (class 1)",
        example=0.1234
    )
    probability_no_default: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of no default (class 0)",
        example=0.8766
    )
    probability_default: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of default (class 1)",
        example=0.1234
    )
    risk_level: str = Field(
        ..., 
        description="Risk category: Low, Medium, High, Very High",
        example="Low"
    )
    reason_codes: List[str] = Field(
        ..., 
        description="List of reasons for the prediction",
        example=[
            "No history of severe delinquency",
            "Low credit utilization (45.0%)"
        ]
    )
    timestamp: str = Field(
        ..., 
        description="Prediction timestamp in ISO format",
        example="2024-02-01T10:30:00.123456"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "prediction_label": "No Default",
                "probability": 0.1234,
                "probability_no_default": 0.8766,
                "probability_default": 0.1234,
                "risk_level": "Low",
                "reason_codes": [
                    "No history of severe delinquency",
                    "Low credit utilization (45.0%)",
                    "Low debt burden (debt ratio: 0.35)"
                ],
                "timestamp": "2024-02-01T10:30:00.123456"
            }
        }


class BatchPredictionInput(BaseModel):
    """
    Input schema for batch predictions
    
    Contains multiple credit applications
    """
    instances: List[CreditRiskInput] = Field(
        ...,
        description="List of credit applications to predict",
        min_items=1,
        max_items=1000  # Limit batch size
    )
    
    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "rev_util": 0.45,
                        "age": 35,
                        "late_30_59": 0,
                        "debt_ratio": 0.35,
                        "monthly_inc": 5000.0,
                        "open_credit": 8,
                        "late_90": 0,
                        "real_estate": 1,
                        "late_60_89": 0,
                        "dependents": 2
                    },
                    {
                        "rev_util": 0.85,
                        "age": 28,
                        "late_30_59": 2,
                        "debt_ratio": 0.75,
                        "monthly_inc": 2500.0,
                        "open_credit": 12,
                        "late_90": 1,
                        "real_estate": 0,
                        "late_60_89": 1,
                        "dependents": 1
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[Dict[str, Any]] = Field(
        ...,
        description="List of predictions for each instance"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics for the batch"
    )
    timestamp: str = Field(
        ...,
        description="Batch prediction timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "id": 0,
                        "prediction": 0,
                        "prediction_label": "No Default",
                        "probability_default": 0.1234,
                        "risk_level": "Low",
                        "reason_codes": ["Clean payment history"]
                    },
                    {
                        "id": 1,
                        "prediction": 1,
                        "prediction_label": "Default",
                        "probability_default": 0.7856,
                        "risk_level": "High",
                        "reason_codes": ["History of severe delinquency"]
                    }
                ],
                "summary": {
                    "total_instances": 2,
                    "predicted_defaults": 1,
                    "predicted_no_defaults": 1,
                    "default_rate": 0.5
                },
                "timestamp": "2024-02-01T10:30:00.123456"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(
        ...,
        description="Service status: healthy, degraded, or unhealthy",
        example="healthy"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether ML model is loaded and ready",
        example=True
    )
    timestamp: str = Field(
        ...,
        description="Health check timestamp",
        example="2024-02-01T10:30:00.123456"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-02-01T10:30:00.123456"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(
        ...,
        description="Error type or category",
        example="ValidationError"
    )
    detail: str = Field(
        ...,
        description="Detailed error message",
        example="Invalid input: age must be between 18 and 120"
    )
    timestamp: str = Field(
        ...,
        description="Error timestamp",
        example="2024-02-01T10:30:00.123456"
    )
    path: Optional[str] = Field(
        None,
        description="Request path that caused the error",
        example="/predict"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid input: age must be between 18 and 120",
                "timestamp": "2024-02-01T10:30:00.123456",
                "path": "/predict"
            }
        }