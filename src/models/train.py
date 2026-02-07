import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and compare multiple models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def get_model_configs(self):
        """Get model configurations"""
        return {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=self.random_state),
                'params': {
                    'classifier__C': [0.01, 0.1, 1, 10],
                    'classifier__class_weight': ['balanced', None]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__class_weight': ['balanced', None]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.1],
                    'classifier__scale_pos_weight': [1, 3, 5]
                }
            },
            'lightgbm': {
                'model': LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.1]
                }
            }
        }
    
    def train_model(self, X_train, y_train, model_name, use_smote=True, tune=True):
        """Train a single model"""
        logger.info(f"\nTraining {model_name}...")
        
        config = self.get_model_configs()[model_name]
        
        # Create pipeline with optional SMOTE
        if use_smote:
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=self.random_state)),
                ('classifier', config['model'])
            ])
        else:
            pipeline = ImbPipeline([
                ('classifier', config['model'])
            ])
        
        # Hyperparameter tuning
        if tune:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            grid = GridSearchCV(
                pipeline,
                config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            logger.info(f"Best params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
            return pipeline
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test, models_to_train=None):
        """Train multiple models and compare"""
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        for model_name in models_to_train:
            # Train
            trained_model = self.train_model(X_train, y_train, model_name)
            
            # Evaluate
            metrics = self.evaluate_model(trained_model, X_test, y_test, model_name)
            
            # Store
            self.models[model_name] = trained_model
            self.results[model_name] = metrics
        
        # Find best model
        best_name = max(self.results.items(), key=lambda x: x[1]['roc_auc'])[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"\n Best Model: {best_name} (ROC-AUC: {self.results[best_name]['roc_auc']:.4f})")
        return self.results
