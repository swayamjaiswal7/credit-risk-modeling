import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create domain-specific features for credit risk
    
    Creates 17 new features:
    - 7 aggregate features (totals, ratios)
    - 6 interaction features (combined effects)
    - 4 binned features (categories)
    """
    
    def __init__(self, create_interactions=True, create_aggregates=True, create_bins=True):
        self.create_interactions = create_interactions
        self.create_aggregates = create_aggregates
        self.create_bins = create_bins
        
    def fit(self, X: pd.DataFrame, y=None):
        """No fitting needed - stateless transformer"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        X = X.copy()
        
        # Fill NaN BEFORE creating features
        X = X.fillna(X.median())
        
        if self.create_aggregates:
            X = self._create_aggregates(X)
        
        if self.create_interactions:
            X = self._create_interactions(X)
        
        if self.create_bins:
            X = self._create_bins(X)
        
        # Final cleanup - remove any NaN/Inf that were created
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"✓ Created features. Shape: {X.shape}")
        return X
    
    def _create_aggregates(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features - combine related columns
        Example: total_late_payments = late_30_59 + late_60_89 + late_90
        """
        # Total late payments
        X['total_late_payments'] = X['late_30_59'] + X['late_60_89'] + X['late_90']
        
        # Has severe delinquency (binary: yes/no)
        X['has_severe_delinquency'] = (X['late_90'] > 0).astype(int)
        
        # Number of different delinquency types
        X['delinquency_types'] = (
            (X['late_30_59'] > 0).astype(int) +
            (X['late_60_89'] > 0).astype(int) +
            (X['late_90'] > 0).astype(int)
        )
        
        # Total credit lines
        X['total_credit_lines'] = X['open_credit'] + X['real_estate']
        
        # Real estate ratio (prevent division by zero with +1)
        X['real_estate_ratio'] = X['real_estate'] / (X['total_credit_lines'] + 1)
        
        # Average late severity (weighted: 30-59 days = 1, 60-89 = 2, 90+ = 3)
        X['avg_late_severity'] = (
            X['late_30_59']*1 + X['late_60_89']*2 + X['late_90']*3
        ) / (X['total_late_payments'] + 1)
        
        # Dependents per income (financial burden)
        income_safe = X['monthly_inc'].clip(lower=1)  # Prevent division by zero
        X['dependents_per_income'] = X['dependents'] / income_safe
        
        return X
    
    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Interaction features - multiply/divide features to capture combined effects
        Example: financial_stress = rev_util * debt_ratio
        """
        # Financial stress (high utilization AND high debt = bad)
        X['financial_stress'] = X['rev_util'] * X['debt_ratio']
        
        # Risky borrower score
        X['risky_borrower_score'] = X['total_late_payments'] * X['rev_util']
        
        # Debt delinquency risk
        X['debt_delinquency_risk'] = X['debt_ratio'] * (X['late_90'] + 1)
        
        # Age-debt interaction
        X['age_debt_interaction'] = X['age'] * X['debt_ratio']
        
        # Credit management (prevent division by zero)
        credit_safe = X['open_credit'].clip(lower=1)
        X['credit_management'] = X['rev_util'] / credit_safe
        
        # Income stability (prevent division by zero)
        income_safe = X['monthly_inc'].clip(lower=1)
        age_safe = X['age'].clip(lower=1)
        dep_safe = X['dependents'].clip(lower=1)
        X['income_stability'] = income_safe / (age_safe * dep_safe)
        
        return X
    
    def _create_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Binned features - convert continuous to categories
        Example: age 25 -> age_group 0 (18-25)
        """
        # Age groups
        X['age_group'] = pd.cut(
            X['age'],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=[0, 1, 2, 3, 4, 5]
        )
        X['age_group'] = X['age_group'].astype(float).fillna(2)  # Default to middle
        
        # Utilization buckets
        X['util_bucket'] = pd.cut(
            X['rev_util'],
            bins=[-np.inf, 0.3, 0.5, 0.75, 1.0, np.inf],
            labels=[0, 1, 2, 3, 4]
        )
        X['util_bucket'] = X['util_bucket'].astype(float).fillna(1)
        
        # Debt categories
        X['debt_category'] = pd.cut(
            X['debt_ratio'],
            bins=[-np.inf, 0.2, 0.4, 0.6, np.inf],
            labels=[0, 1, 2, 3]
        )
        X['debt_category'] = X['debt_category'].astype(float).fillna(1)
        
        # Income brackets (using percentiles)
        try:
            q25 = X['monthly_inc'].quantile(0.25)
            q50 = X['monthly_inc'].quantile(0.50)
            q75 = X['monthly_inc'].quantile(0.75)
            
            X['income_bracket'] = pd.cut(
                X['monthly_inc'],
                bins=[-np.inf, q25, q50, q75, np.inf],
                labels=[0, 1, 2, 3]
            )
            X['income_bracket'] = X['income_bracket'].astype(float).fillna(1)
        except:
            X['income_bracket'] = 1  # If binning fails, use default
        
        return X
