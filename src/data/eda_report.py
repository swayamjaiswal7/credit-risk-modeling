"""
Comprehensive EDA Report for Credit Risk Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # no Tkinter, no GUI
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class CreditRiskEDA:
    def __init__(self, filepath):
        """Initialize EDA with data """
        self.df = pd.read_csv(filepath)
        self.target = 'dlq_2yrs'
        self.features = [col for col in self.df.columns if col != self.target]
        
    def generate_full_report(self):
        """Generate complete EDA report"""
        print("CREDIT RISK DATASET - EXPLORATORY DATA ANALYSIS REPORT")
        print("="*80)
        
        self.dataset_overview()
        self.missing_value_analysis()
        self.target_distribution()
        self.numerical_feature_analysis()
        self.correlation_analysis()
        self.outlier_detection()
        self.bivariate_analysis()
        self.key_insights()
        
    def dataset_overview(self):
        """Basic dataset information"""
        print("1. Dataset Overview")
        print("="*80)
        print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        
    def missing_value_analysis(self):
        """Analyze missing values"""
        print("2 Missing Value Analysis")
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Percentage', ascending=False)
        
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        if missing_df['Missing_Count'].sum() == 0:
            print("\n No missing values detected!")
        else:
            print(f"\n Total missing values: {missing_df['Missing_Count'].sum()}")
            
    def target_distribution(self):
        """Analyze target variable distribution"""
        print("3 delinquish_2yr distribution")
        print("="*80)
        
        target_counts = self.df[self.target].value_counts()
        target_pct = self.df[self.target].value_counts(normalize=True) * 100
        
        print(f"\nClass Distribution:")
        print(f"  Class 0 (No Delinquency): {target_counts[0]} ({target_pct[0]:.2f}%)")
        print(f"  Class 1 (Delinquency): {target_counts[1]} ({target_pct[1]:.2f}%)")
        
        imbalance_ratio = target_counts[0] / target_counts[1]
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print(" HIGHLY IMBALANCED - Consider SMOTE, class weights, or stratified sampling")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        target_counts.plot(kind='bar', ax=axes[0], color=['green', 'red'])
        axes[0].set_title('Target Variable Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['No Delinquency', 'Delinquency'], rotation=0)
        
        # Pie chart
        axes[1].pie(target_counts, labels=['No Delinquency', 'Delinquency'], 
                   autopct='%1.1f%%', colors=['green', 'red'])
        axes[1].set_title('Class Proportion')
        
        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        print("\n Saved: target_distribution.png")
        plt.close()
        
    def numerical_feature_analysis(self):
        """Analyze numerical features"""
        print("\n" + "="*80)
        print("4 Input Feature Analysis")
        print("="*80)
        
        for feature in self.features:
            print(f"\n--- {feature} ---")
            print(f"Mean: {self.df[feature].mean():.2f}")
            print(f"Median: {self.df[feature].median():.2f}")
            print(f"Std: {self.df[feature].std():.2f}")
            print(f"Min: {self.df[feature].min():.2f}")
            print(f"Max: {self.df[feature].max():.2f}")
            print(f"Skewness: {self.df[feature].skew():.2f}")
            print(f"Kurtosis: {self.df[feature].kurtosis():.2f}")
        
        # Distribution plots
        n_features = len(self.features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.features):
            self.df[feature].hist(bins=50, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'{feature} Distribution')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig('input_variable_dist.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: input_variables_distributions.png")
        plt.close()
        
    def correlation_analysis(self):
        """Correlation matrix and analysis
        threshold is >0.7"""
        print("5. Multicollinearity tests using Correlation")
        
        corr_matrix = self.df.corr()
        
        # Correlation with target
        target_corr = corr_matrix[self.target].sort_values(ascending=False)
        print(f"\nCorrelation with Target Variable ({self.target}):")
        print(target_corr)
        
        # Identify highly correlated features (multicollinearity)
        print("\n\nHighly Correlated Feature Pairs (|corr| > 0.7):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'Feature1': corr_matrix.columns[i],
                        'Feature2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            for pair in high_corr_pairs:
                print(f"  {pair['Feature1']} <-> {pair['Feature2']}: {pair['Correlation']:.3f}")
        else:
            print("  None found - Good!")
        
        # Heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: correlation_matrix.png")
        plt.close()
        
    def outlier_detection(self):
        """Detect outliers using IQR and Z-score methods"""
        print("6. OUTLIER DETECTION")
        print("="*80)
        
        outlier_summary = []
        
        for feature in self.features:
            # IQR method
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = ((self.df[feature] < lower_bound) | 
                           (self.df[feature] > upper_bound)).sum()
            outliers_pct = (outliers_iqr / len(self.df)) * 100
            
            outlier_summary.append({
                'Feature': feature,
                'Outliers_Count': outliers_iqr,
                'Outliers_Pct': outliers_pct,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })
        
        outlier_df = pd.DataFrame(outlier_summary).sort_values('Outliers_Pct', ascending=False)
        print(outlier_df)
        n_rows = self.df.shape[0]
        n_cols = self.df.shape[1]
        # Box plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.features):
            self.df.boxplot(column=feature, ax=axes[idx])
            axes[idx].set_title(f'{feature} - Boxplot')
            axes[idx].set_ylabel(feature)
            
        for idx in range(len(self.features), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig('outlier_boxplots.png', dpi=120, bbox_inches='tight')
        print("\n Saved: outlier_boxplots.png")
        plt.close()
        
    def bivariate_analysis(self):
        """Analyze relationship between features and target"""
        print("\n" + "="*80)
        print("7. BIVARIATE ANALYSIS (Features vs Target)")
        print("="*80)
        
        # Feature comparison by target class
        for feature in self.features:
            print(f"\n--- {feature} ---")
            
            # Statistics by class
            class_0_stats = self.df[self.df[self.target] == 0][feature].describe()
            class_1_stats = self.df[self.df[self.target] == 1][feature].describe()
            
            print(f"Class 0 (No Delinquency) - Mean: {class_0_stats['mean']:.2f}, "
                  f"Median: {class_0_stats['50%']:.2f}")
            print(f"Class 1 (Delinquency) - Mean: {class_1_stats['mean']:.2f}, "
                  f"Median: {class_1_stats['50%']:.2f}")
            
            # Statistical test (Mann-Whitney U test for non-normal distributions)
            stat, p_value = stats.mannwhitneyu(
                self.df[self.df[self.target] == 0][feature].dropna(),
                self.df[self.df[self.target] == 1][feature].dropna()
            )
            print(f"Mann-Whitney U test p-value: {p_value:.6f}")
            if p_value < 0.05:
                print("   Statistically significant difference between classes")
            else:
                print("  No significant difference")
        
        # Visualization - distributions by target
        n_features = len(self.features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.features):
            self.df[self.df[self.target] == 0][feature].hist(
                bins=30, alpha=0.5, label='No Delinquency', ax=axes[idx], color='green')
            self.df[self.df[self.target] == 1][feature].hist(
                bins=30, alpha=0.5, label='Delinquency', ax=axes[idx], color='red')
            axes[idx].set_title(f'{feature} by Target Class')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig('bivariate_distributions.png', dpi=300, bbox_inches='tight')
        print("\n Saved: bivariate_distributions.png")
        plt.close()
        
    def key_insights(self):
        """Summarize key insights and recommendations"""
        print("8. KEY INSIGHTS & RECOMMENDATIONS")
        print("="*80)
        
        print("\n DATA QUALITY:")
        missing_total = self.df.isnull().sum().sum()
        if missing_total > 0:
            print(f"  • Missing values detected - implement imputation strategy")
        else:
            print(f"  • ✓ No missing values")
            
        print("\n CLASS IMBALANCE:")
        target_counts = self.df[self.target].value_counts()
        imbalance_ratio = target_counts[0] / target_counts[1]
        print(f"  • Imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 3:
            print("  • Recommendation: Use SMOTE, class weights, or stratified sampling")
            print("  • Evaluation: Focus on AUC-ROC, Precision-Recall, F1-Score (not accuracy)")
        
        print("\n FEATURE ENGINEERING OPPORTUNITIES:")
        print("  • Create interaction terms (e.g., debt_ratio × late_90)")
        print("  • Aggregate late payment features (late_30_59 + late_60_89 + late_90)")
        print("  • Bin age into risk categories")
        print("  • Cap extreme values in rev_util and debt_ratio")
        
        print("\n MODELING RECOMMENDATIONS:")
        print("  • Baseline: Logistic Regression (interpretable)")
        print("  • High Performance: XGBoost, LightGBM")
        print("  • Use stratified k-fold cross-validation")
        print("  • Hyperparameter tuning with proper validation")
        

# Usage
if __name__ == "__main__":
    # REPLACE WITH YOUR FILE PATH
    filepath = r"D:\Credit_Risk_Project\data\raw\Credit_Risk_Benchmark.csv"
    
    eda = CreditRiskEDA(filepath)
    eda.generate_full_report()
    
    print("\n" + "="*80)
    print("EDA REPORT COMPLETED")
    print("="*80)
    print("\nGenerated files:")
    print("  • target_distribution.png")
    print("  • feature_distributions.png")
    print("  • correlation_matrix.png")
    print("  • outlier_boxplots.png")
    print("  • bivariate_distributions.png")