# ========================================
# FILE 1: src/data/preprocessor.py
# PURPOSE: Handle missing values, outliers, and scaling
# USES: RobustScaler for handling outliers better than StandardScaler
# ========================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import logging
import joblib

logger = logging.getLogger(__name__)

class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing pipeline for credit risk data
    
    Steps:
    1. Handle missing values (median imputation)
    2. Handle outliers (cap at 1st-99th percentile)
    3. Scale features (RobustScaler - good for outliers)
    """
    
    def __init__(self):
        self.fill_values_ = {}
        self.outlier_bounds_ = {}
        self.scaler = RobustScaler()
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Learn preprocessing parameters from training data"""
        X = X.copy()
        
        # 1. Learn fill values for missing data
        for col in X.columns:
            if X[col].isnull().any():
                self.fill_values_[col] = X[col].median()
                logger.info(f"Will fill {col} NaN with {self.fill_values_[col]:.2f}")
        
        # Fill NaN before calculating outlier bounds
        for col, value in self.fill_values_.items():
            X[col].fillna(value, inplace=True)
        
        # 2. Learn outlier bounds (1st and 99th percentile)
        for col in X.columns:
            lower = X[col].quantile(0.01)
            upper = X[col].quantile(0.99)
            self.outlier_bounds_[col] = (lower, upper)
        
        # Apply outlier capping before scaling
        for col, (lower, upper) in self.outlier_bounds_.items():
            X[col] = X[col].clip(lower, upper)
        
        # 3. Fit scaler
        self.scaler.fit(X)
        self.feature_names_ = X.columns.tolist()
        
        logger.info("✓ Preprocessor fitted")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations"""
        X = X.copy()
        
        # 1. Fill missing values
        for col, value in self.fill_values_.items():
            if col in X.columns and X[col].isnull().any():
                X[col].fillna(value, inplace=True)
        
        # 2. Cap outliers
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower, upper)
        
        # 3. Scale features
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: str):
        """Save preprocessor to disk"""
        joblib.dump({
            'fill_values': self.fill_values_,
            'outlier_bounds': self.outlier_bounds_,
            'scaler': self.scaler,
            'feature_names': self.feature_names_
        }, filepath)
        logger.info(f"✓ Saved preprocessor to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.fill_values_ = data['fill_values']
        preprocessor.outlier_bounds_ = data['outlier_bounds']
        preprocessor.scaler = data['scaler']
        preprocessor.feature_names_ = data['feature_names']
        logger.info(f"✓ Loaded preprocessor from {filepath}")
        return preprocessor