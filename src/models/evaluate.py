import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance with visualizations"""
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name, save_path=None):
        """Plot ROC curve"""
        from sklearn.metrics import roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        plt.close()
    
    def generate_report(self, y_test, y_pred, y_pred_proba, model_name):
        """Generate comprehensive evaluation report"""
        # Classification report
        report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])
        logger.info(f"\nClassification Report - {model_name}:\n{report}")
        
        # Confusion matrix details
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"TN: {cm[0,0]:5d} | FP: {cm[0,1]:5d}")
        logger.info(f"FN: {cm[1,0]:5d} | TP: {cm[1,1]:5d}")
