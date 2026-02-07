import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CreditRiskPredictor:
    """Make predictions using saved models"""
    
    def __init__(self, model_path, preprocessor_path, feature_engineer_path):
        """Load all saved components"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        logger.info(" All components loaded")
    
    def predict_single(self, features_dict, return_probability=True, return_risk_level=True):
        """Predict for single instance"""
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Apply transformations
        df = self.preprocessor.transform(df)
        df = self.feature_engineer.transform(df)
        
        # Predict
        prediction = self.model.predict(df)[0]
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default'
        }
        
        if return_probability:
            proba = self.model.predict_proba(df)[0]
            result['probability_no_default'] = float(proba[0])
            result['probability_default'] = float(proba[1])
            
            if return_risk_level:
                result['risk_level'] = self._get_risk_level(proba[1])
        
        return result
    
    def predict_batch(self, features_list):
        """Predict for multiple instances"""
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Apply transformations
        df = self.preprocessor.transform(df)
        df = self.feature_engineer.transform(df)
        
        # Predict
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                'id': i,
                'prediction': int(pred),
                'prediction_label': 'Default' if pred == 1 else 'No Default',
                'probability_default': float(proba[1]),
                'risk_level': self._get_risk_level(proba[1])
            })
        
        return results
    
    def _get_risk_level(self, probability):
        """Categorize risk level"""
        if probability < 0.25:
            return "Low"
        elif probability < 0.50:
            return "Medium"
        elif probability < 0.75:
            return "High"
        else:
            return "Very High"
    
    @classmethod
    def from_directory(cls, model_dir='models'):
        """Load from directory"""
        return cls(
            model_path=f'{model_dir}/best_model.pkl',
            preprocessor_path=f'{model_dir}/preprocessor.pkl',
            feature_engineer_path=f'{model_dir}/feature_engineer.pkl'
        )
