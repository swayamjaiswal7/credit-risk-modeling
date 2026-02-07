"""
Unit tests for FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestPredictEndpoint:
    """Test prediction endpoints"""
    
    @pytest.fixture
    def valid_input(self):
        """Valid prediction input"""
        return {
            "rev_util": 0.45,
            "age": 35,
            "late_30_59": 0,
            "debt_ratio": 0.35,
            "monthly_inc": 5000,
            "open_credit": 8,
            "late_90": 0,
            "real_estate": 1,
            "late_60_89": 0,
            "dependents": 2
        }
    
    @pytest.fixture
    def high_risk_input(self):
        """High risk profile"""
        return {
            "rev_util": 0.95,
            "age": 25,
            "late_30_59": 3,
            "debt_ratio": 0.85,
            "monthly_inc": 1500,
            "open_credit": 15,
            "late_90": 2,
            "real_estate": 0,
            "late_60_89": 2,
            "dependents": 3
        }
    
    def test_predict_valid_input(self, valid_input):
        """Test prediction with valid input"""
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert "reason_codes" in data
        assert "timestamp" in data
        
        # Check data types
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]
        assert isinstance(data["probability"], float)
        assert 0 <= data["probability"] <= 1
        assert isinstance(data["risk_level"], str)
        assert isinstance(data["reason_codes"], list)
    
    def test_predict_high_risk(self, high_risk_input):
        """Test prediction returns high risk for risky profile"""
        response = client.post("/predict", json=high_risk_input)
        assert response.status_code == 200
        
        data = response.json()
        # High risk profile should have higher probability
        assert data["probability"] > 0.3  # Adjust based on your model
    
    def test_predict_missing_field(self):
        """Test prediction with missing required field"""
        incomplete_input = {
            "rev_util": 0.45,
            "age": 35,
            # Missing other required fields
        }
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_type(self, valid_input):
        """Test prediction with invalid data type"""
        invalid_input = valid_input.copy()
        invalid_input["age"] = "thirty-five"  # String instead of int
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_negative_value(self, valid_input):
        """Test prediction with negative value"""
        invalid_input = valid_input.copy()
        invalid_input["monthly_inc"] = -1000
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_age_out_of_range(self, valid_input):
        """Test prediction with age out of valid range"""
        invalid_input = valid_input.copy()
        invalid_input["age"] = 150  # Too old
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint"""
    
    @pytest.fixture
    def batch_input(self):
        """Multiple valid inputs"""
        return {
            "instances": [
                {
                    "rev_util": 0.45,
                    "age": 35,
                    "late_30_59": 0,
                    "debt_ratio": 0.35,
                    "monthly_inc": 5000,
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
                    "monthly_inc": 2500,
                    "open_credit": 12,
                    "late_90": 1,
                    "real_estate": 0,
                    "late_60_89": 1,
                    "dependents": 1
                }
            ]
        }
    
    def test_batch_predict(self, batch_input):
        """Test batch prediction with multiple instances"""
        response = client.post("/batch_predict", json=batch_input)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_instances" in data
        assert "predictions" in data
        assert "timestamp" in data
        
        assert data["total_instances"] == 2
        assert len(data["predictions"]) == 2
        
        # Check each prediction
        for pred in data["predictions"]:
            assert "instance_id" in pred
            assert "prediction" in pred
            assert "probability" in pred
            assert "risk_level" in pred
    
    def test_batch_predict_empty(self):
        """Test batch prediction with empty list"""
        empty_batch = {"instances": []}
        response = client.post("/batch_predict", json=empty_batch)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_instances"] == 0


class TestAPIPerformance:
    """Test API performance"""
    
    def test_prediction_latency(self, benchmark):
        """Benchmark prediction latency"""
        input_data = {
            "rev_util": 0.45,
            "age": 35,
            "late_30_59": 0,
            "debt_ratio": 0.35,
            "monthly_inc": 5000,
            "open_credit": 8,
            "late_90": 0,
            "real_estate": 1,
            "late_60_89": 0,
            "dependents": 2
        }
        
        def make_prediction():
            response = client.post("/predict", json=input_data)
            return response
        
        # Benchmark the prediction
        result = benchmark(make_prediction)
        assert result.status_code == 200


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])