"""
Model Interpretability using SHAP and Feature Importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Any, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Generate interpretability insights for credit risk models"""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Args:
            model: Trained model (must have predict_proba method)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_shap_explainer(self, X_background: pd.DataFrame, 
                             max_samples: int = 100):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background data for SHAP (typically training sample)
            max_samples: Maximum samples to use for background
        """
        logger.info("Creating SHAP explainer...")
        
        # Sample background data if too large
        if len(X_background) > max_samples:
            X_background = X_background.sample(max_samples, random_state=42)
        
        # Extract the actual classifier from pipeline
        if hasattr(self.model, 'named_steps'):
            classifier = self.model.named_steps['classifier']
        else:
            classifier = self.model
        
        # Create explainer based on model type
        try:
            # Try TreeExplainer for tree-based models (faster)
            self.explainer = shap.TreeExplainer(classifier)
            logger.info("✓ Created TreeExplainer")
        except:
            # Fall back to KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                classifier.predict_proba, 
                X_background
            )
            logger.info("✓ Created KernelExplainer")
    
    def calculate_shap_values(self, X: pd.DataFrame):
        """Calculate SHAP values for dataset"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_shap_explainer first.")
        
        logger.info("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, get values for positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        logger.info("✓ SHAP values calculated")
        return self.shap_values
    
    def plot_feature_importance(self, X: pd.DataFrame, 
                               top_n: int = 20,
                               save_path: str = 'feature_importance_shap.png'):
        """Plot SHAP feature importance"""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, 
                         feature_names=self.feature_names,
                         max_display=top_n, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved SHAP summary plot to {save_path}")
        plt.close()
    
    def plot_shap_summary_bar(self, X: pd.DataFrame, 
                             top_n: int = 20,
                             save_path: str = 'feature_importance_bar.png'):
        """Plot mean absolute SHAP values as bar chart"""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', 
                   palette='viridis')
        plt.title(f'Top {top_n} Feature Importance (Mean |SHAP|)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved bar chart to {save_path}")
        plt.close()
        
        return importance_df
    
    def plot_dependence_plot(self, feature: str, X: pd.DataFrame,
                            interaction_feature: str = None,
                            save_path: str = None):
        """
        Plot SHAP dependence plot showing feature effect
        
        Args:
            feature: Feature to plot
            X: Data
            interaction_feature: Optional feature to color by
            save_path: Where to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        if save_path is None:
            save_path = f'dependence_{feature}.png'
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, self.shap_values, X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved dependence plot to {save_path}")
        plt.close()
    
    def explain_prediction(self, instance: pd.DataFrame, 
                          instance_name: str = "Instance",
                          save_path: str = None) -> Dict:
        """
        Explain a single prediction with SHAP force plot
        
        Args:
            instance: Single row DataFrame to explain
            instance_name: Name for the instance
            save_path: Where to save plot
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_shap_explainer first.")
        
        # Get prediction
        if hasattr(self.model, 'named_steps'):
            classifier = self.model.named_steps['classifier']
        else:
            classifier = self.model
            
        prediction_proba = classifier.predict_proba(instance)[0, 1]
        prediction = int(prediction_proba > 0.5)
        
        # Calculate SHAP values for instance
        shap_values = self.explainer.shap_values(instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get top contributing features
        shap_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'feature_value': instance.values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        explanation = {
            'prediction': prediction,
            'probability': prediction_proba,
            'top_features': shap_df.head(10).to_dict('records')
        }
        
        # Generate force plot
        if save_path:
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                self.explainer.expected_value if isinstance(self.explainer.expected_value, float) 
                else self.explainer.expected_value[1],
                shap_values[0],
                instance.iloc[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved force plot to {save_path}")
            plt.close()
        
        return explanation
    
    def generate_reason_codes(self, instance: pd.DataFrame, 
                             top_n: int = 4) -> List[str]:
        """
        Generate human-readable reason codes for a prediction
        (Required for regulatory compliance)
        
        Args:
            instance: Single row DataFrame
            top_n: Number of reason codes to generate
            
        Returns:
            List of reason code strings
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_shap_explainer first.")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get features with highest absolute SHAP values
        feature_impacts = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'feature_value': instance.values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        # Generate reason codes
        reason_codes = []
        reason_templates = {
            'rev_util': 'Revolving credit utilization: {:.1%}',
            'debt_ratio': 'Debt-to-income ratio: {:.2f}',
            'late_90': 'Number of 90+ day late payments: {:.0f}',
            'late_60_89': 'Number of 60-89 day late payments: {:.0f}',
            'late_30_59': 'Number of 30-59 day late payments: {:.0f}',
            'age': 'Age: {:.0f} years',
            'monthly_inc': 'Monthly income: ${:,.0f}',
            'open_credit': 'Number of open credit lines: {:.0f}',
            'total_late_payments': 'Total late payments: {:.0f}',
            'has_severe_delinquency': 'Severe delinquency history',
            'financial_stress': 'High financial stress indicator'
        }
        
        for _, row in feature_impacts.head(top_n).iterrows():
            feature = row['feature']
            value = row['feature_value']
            impact = 'increases' if row['shap_value'] > 0 else 'decreases'
            
            if feature in reason_templates:
                description = reason_templates[feature].format(value)
            else:
                description = f'{feature}: {value:.2f}'
            
            reason_codes.append(f"{description} ({impact} risk)")
        
        return reason_codes
    
    def create_comprehensive_report(self, X_test: pd.DataFrame, 
                                   output_dir: str = 'interpretability_report'):
        """Generate comprehensive interpretability report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating comprehensive interpretability report...")
        
        # 1. Feature importance
        self.plot_feature_importance(
            X_test, 
            save_path=f'{output_dir}/feature_importance_summary.png'
        )
        
        # 2. Bar chart
        importance_df = self.plot_shap_summary_bar(
            X_test,
            save_path=f'{output_dir}/feature_importance_bar.png'
        )
        
        # 3. Dependence plots for top features
        top_features = importance_df.head(5)['feature'].tolist()
        for feature in top_features:
            self.plot_dependence_plot(
                feature, X_test,
                save_path=f'{output_dir}/dependence_{feature}.png'
            )
        
        # 4. Example predictions
        sample_indices = X_test.sample(3, random_state=42).index
        for i, idx in enumerate(sample_indices):
            instance = X_test.loc[[idx]]
            self.explain_prediction(
                instance,
                save_path=f'{output_dir}/explanation_example_{i+1}.png'
            )
        
        logger.info(f"✓ Comprehensive report saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    import joblib
    
    # Load trained model and data
    model_data = joblib.load('models/best_model.pkl')
    model = model_data['model']
    
    df = pd.read_csv('processed_data.csv')
    X = df.drop('dlq_2yrs', axis=1)
    y = df['dlq_2yrs']
    
    # Initialize interpreter
    interpreter = ModelInterpreter(model, feature_names=X.columns.tolist())
    
    # Create explainer
    interpreter.create_shap_explainer(X.sample(100))
    
    # Generate interpretability report
    interpreter.create_comprehensive_report(X.sample(500))
    
    # Example: Explain single prediction
    instance = X.sample(1)
    explanation = interpreter.explain_prediction(
        instance, 
        save_path='single_prediction_explanation.png'
    )
    
    print("\nPrediction Explanation:")
    print(f"Predicted Class: {explanation['prediction']}")
    print(f"Probability: {explanation['probability']:.4f}")
    print("\nTop Contributing Features:")
    for feat in explanation['top_features'][:5]:
        print(f"  {feat['feature']}: {feat['feature_value']:.3f} "
              f"(SHAP: {feat['shap_value']:.3f})")
    
    # Generate reason codes
    reason_codes = interpreter.generate_reason_codes(instance)
    print("\nReason Codes for Decision:")
    for i, code in enumerate(reason_codes, 1):
        print(f"  {i}. {code}")