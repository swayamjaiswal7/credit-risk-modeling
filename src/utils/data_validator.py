"""
Data quality validation utilities
Catch NaN, Inf, and other data quality issues early
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validate data quality and fix common issues"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.issues = {}
        
    def validate(self, df: pd.DataFrame, stage: str = "unknown") -> Tuple[bool, Dict]:
        """
        Comprehensive data quality validation
        
        Args:
            df: DataFrame to validate
            stage: Stage of pipeline (for logging)
            
        Returns:
            (is_valid, issues_dict)
        """
        self.issues = {'stage': stage}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Data Quality Check - {stage}")
        logger.info(f"{'='*60}")
        
        # Check 1: NaN values
        self._check_nan(df)
        
        # Check 2: Infinity values
        self._check_inf(df)
        
        # Check 3: Data types
        self._check_dtypes(df)
        
        # Check 4: Duplicate rows
        self._check_duplicates(df)
        
        # Check 5: Constant columns
        self._check_constant_columns(df)
        
        # Check 6: Extreme values
        self._check_extreme_values(df)
        
        # Summary
        is_valid = (
            self.issues.get('nan_count', 0) == 0 and
            self.issues.get('inf_count', 0) == 0
        )
        
        if is_valid:
            logger.info("✓ Data quality check PASSED")
        else:
            logger.warning("⚠ Data quality check FAILED")
            
        return is_valid, self.issues
    
    def _check_nan(self, df: pd.DataFrame):
        """Check for NaN values"""
        nan_count = df.isna().sum().sum()
        nan_cols = df.columns[df.isna().any()].tolist()
        
        self.issues['nan_count'] = nan_count
        self.issues['nan_columns'] = nan_cols
        
        if nan_count > 0:
            logger.warning(f"⚠ Found {nan_count} NaN values")
            logger.warning(f"  Affected columns: {nan_cols}")
            
            # Show detailed counts
            if self.verbose:
                for col in nan_cols:
                    count = df[col].isna().sum()
                    pct = (count / len(df)) * 100
                    logger.warning(f"    {col}: {count} ({pct:.2f}%)")
        else:
            logger.info("✓ No NaN values found")
    
    def _check_inf(self, df: pd.DataFrame):
        """Check for infinity values"""
        numeric_df = df.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric_df).sum().sum()
        inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
        
        self.issues['inf_count'] = inf_count
        self.issues['inf_columns'] = inf_cols
        
        if inf_count > 0:
            logger.warning(f"⚠ Found {inf_count} Inf values")
            logger.warning(f"  Affected columns: {inf_cols}")
            
            if self.verbose:
                for col in inf_cols:
                    pos_inf = np.isposinf(df[col]).sum()
                    neg_inf = np.isneginf(df[col]).sum()
                    logger.warning(f"    {col}: +Inf={pos_inf}, -Inf={neg_inf}")
        else:
            logger.info("✓ No Inf values found")
    
    def _check_dtypes(self, df: pd.DataFrame):
        """Check data types"""
        dtypes = df.dtypes.value_counts().to_dict()
        self.issues['dtypes'] = {str(k): v for k, v in dtypes.items()}
        
        if self.verbose:
            logger.info(f"Data types: {self.issues['dtypes']}")
    
    def _check_duplicates(self, df: pd.DataFrame):
        """Check for duplicate rows"""
        dup_count = df.duplicated().sum()
        self.issues['duplicate_count'] = dup_count
        
        if dup_count > 0:
            pct = (dup_count / len(df)) * 100
            logger.warning(f"⚠ Found {dup_count} duplicate rows ({pct:.2f}%)")
        else:
            logger.info("✓ No duplicate rows")
    
    def _check_constant_columns(self, df: pd.DataFrame):
        """Check for columns with single value"""
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        self.issues['constant_columns'] = constant_cols
        
        if constant_cols:
            logger.warning(f"⚠ Found {len(constant_cols)} constant columns: {constant_cols}")
        else:
            logger.info("✓ No constant columns")
    
    def _check_extreme_values(self, df: pd.DataFrame):
        """Check for extreme outliers (beyond 5 std dev)"""
        numeric_df = df.select_dtypes(include=[np.number])
        extreme_cols = []
        
        for col in numeric_df.columns:
            mean = numeric_df[col].mean()
            std = numeric_df[col].std()
            
            if std > 0:
                z_scores = np.abs((numeric_df[col] - mean) / std)
                extreme_count = (z_scores > 5).sum()
                
                if extreme_count > 0:
                    extreme_cols.append({
                        'column': col,
                        'count': extreme_count,
                        'percentage': (extreme_count / len(df)) * 100
                    })
        
        self.issues['extreme_values'] = extreme_cols
        
        if extreme_cols and self.verbose:
            logger.info(f"Extreme outliers (>5σ): {len(extreme_cols)} columns")
            for item in extreme_cols[:5]:  # Show first 5
                logger.info(f"  {item['column']}: {item['count']} ({item['percentage']:.2f}%)")
    
    def fix_issues(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Automatically fix common data quality issues
        
        Args:
            df: DataFrame to fix
            strategy: 'auto', 'drop', 'fill', 'cap'
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        logger.info(f"\nFixing data quality issues (strategy: {strategy})...")
        
        # Fix infinity values
        if self.issues.get('inf_count', 0) > 0:
            logger.info("Replacing Inf values with NaN...")
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fix NaN values
        if self.issues.get('nan_count', 0) > 0 or df.isna().sum().sum() > 0:
            logger.info("Handling NaN values...")
            
            if strategy == 'drop':
                # Drop rows with any NaN
                df = df.dropna()
                logger.info(f"  Dropped rows with NaN. New shape: {df.shape}")
                
            elif strategy == 'fill' or strategy == 'auto':
                # Fill with median for numeric, mode for categorical
                for col in df.columns:
                    if df[col].isna().any():
                        if df[col].dtype in [np.float64, np.int64]:
                            fill_value = df[col].median()
                            if pd.isna(fill_value):  # If median is NaN
                                fill_value = 0
                        else:
                            fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                        
                        df[col] = df[col].fillna(fill_value)
                        logger.info(f"  Filled {col} with {fill_value}")
        
        # Remove constant columns
        if self.issues.get('constant_columns'):
            logger.info("Removing constant columns...")
            df = df.drop(columns=self.issues['constant_columns'])
            logger.info(f"  Removed {len(self.issues['constant_columns'])} columns")
        
        # Verify fixes
        final_check = DataQualityValidator(verbose=False)
        is_valid, _ = final_check.validate(df, "after_fix")
        
        if is_valid:
            logger.info("✓ All issues fixed successfully!")
        else:
            logger.warning("⚠ Some issues remain")
        
        return df


def validate_and_fix(df: pd.DataFrame, stage: str = "unknown", auto_fix: bool = True) -> pd.DataFrame:
    """
    Convenience function to validate and optionally fix data
    
    Args:
        df: DataFrame to validate
        stage: Pipeline stage name
        auto_fix: Whether to automatically fix issues
        
    Returns:
        Validated (and possibly fixed) DataFrame
    """
    validator = DataQualityValidator(verbose=True)
    is_valid, issues = validator.validate(df, stage)
    
    if not is_valid and auto_fix:
        logger.info("\nAuto-fixing data quality issues...")
        df = validator.fix_issues(df, strategy='auto')
    elif not is_valid:
        logger.warning("\n⚠ Data quality issues detected but auto_fix=False")
        logger.warning("Consider running: df = validator.fix_issues(df)")
    
    return df


def check_model_ready(df: pd.DataFrame) -> bool:
    """
    Check if data is ready for model training (no NaN/Inf)
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if ready, False otherwise
    """
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    if nan_count > 0:
        logger.error(f"❌ Cannot train model: {nan_count} NaN values present")
        logger.error(f"   Columns: {df.columns[df.isna().any()].tolist()}")
        return False
    
    if inf_count > 0:
        logger.error(f"❌ Cannot train model: {inf_count} Inf values present")
        logger.error(f"   Columns: {df.select_dtypes(include=[np.number]).columns[np.isinf(df.select_dtypes(include=[np.number])).any()].tolist()}")
        return False
    
    logger.info("✓ Data is ready for model training")
    return True


# Example usage
if __name__ == "__main__":
    # Create test data with issues
    test_data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [1.0, 2.0, 3.0, np.inf, 5.0],
        'feature3': [10, 10, 10, 10, 10],  # Constant
        'feature4': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    
    print("Test Data:")
    print(test_data)
    
    # Validate
    validator = DataQualityValidator()
    is_valid, issues = validator.validate(test_data, "test")
    
    print("\n" + "="*60)
    print("Issues found:")
    for key, value in issues.items():
        print(f"  {key}: {value}")
    
    # Fix
    if not is_valid:
        fixed_data = validator.fix_issues(test_data)
        print("\nFixed Data:")
        print(fixed_data)
        
        # Check if model ready
        check_model_ready(fixed_data)