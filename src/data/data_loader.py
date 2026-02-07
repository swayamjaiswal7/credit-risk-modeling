import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate credit risk data"""
    
    REQUIRED_COLUMNS = [
        'rev_util', 'age', 'late_30_59', 'debt_ratio', 'monthly_inc',
        'open_credit', 'late_90', 'real_estate', 'late_60_89', 'dependents'
    ]
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        return df
    
    def load_and_split(
        self, 
        filepath: str,
        target_column: str = 'dlq_2yrs',
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load data and split into train/test"""
        from sklearn.model_selection import train_test_split
        
        df = self.load_csv(filepath)
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        logger.info(f"Split data - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
