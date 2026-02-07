import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import our custom classes
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import CreditRiskPreprocessor
from src.features.feature_engineering import CreditRiskFeatureEngineer

print("="*60)
print("CREDIT RISK MODEL - TRAINING PIPELINE")
print("="*60)

# 1. LOAD DATA
print("\n1. Loading data...")
df = pd.read_csv("data/raw/Credit_Risk_Benchmark.csv")  # CHANGE PATH HERE
print(f"   Loaded: {df.shape}")
print(f"   Target: {df['dlq_2yrs'].value_counts().to_dict()}")

# 2. SPLIT DATA
print("\n2. Splitting data...")
X = df.drop('dlq_2yrs', axis=1)
y = df['dlq_2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# 3. PREPROCESS
print("\n3. Preprocessing (missing values, outliers, scaling)...")
preprocessor = CreditRiskPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"   ✓ Preprocessed")

# 4. FEATURE ENGINEERING
print("\n4. Feature engineering (creating 17 new features)...")
feature_engineer = CreditRiskFeatureEngineer()
X_train_eng = feature_engineer.fit_transform(X_train_processed)
X_test_eng = feature_engineer.transform(X_test_processed)
print(f"   Original: 10 features -> Engineered: {X_train_eng.shape[1]} features")

# 5. QUALITY CHECK
print("\n5. Data quality check...")
print(f"   X_train NaN: {X_train_eng.isnull().sum().sum()}")
print(f"   X_train Inf: {np.isinf(X_train_eng.select_dtypes(include=[np.number])).sum().sum()}")
print(f"   X_test NaN: {X_test_eng.isnull().sum().sum()}")
print(f"   X_test Inf: {np.isinf(X_test_eng.select_dtypes(include=[np.number])).sum().sum()}")

if X_train_eng.isnull().sum().sum() > 0 or np.isinf(X_train_eng.select_dtypes(include=[np.number])).sum().sum() > 0:
    print("   ⚠️ Fixing issues...")
    X_train_eng = X_train_eng.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_eng = X_test_eng.replace([np.inf, -np.inf], np.nan).fillna(0)

print("   ✓ Data is clean and ready")

# 6. HANDLE CLASS IMBALANCE
print("\n6. Applying SMOTE for class imbalance...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_eng, y_train)
print(f"   Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"   After SMOTE: {y_train_balanced.value_counts().to_dict()}")

# 7. TRAIN MODELS
print("\n7. Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

results = {}
for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = model.predict(X_test_eng)
    y_pred_proba = model.predict_proba(X_test_eng)[:, 1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"      ROC-AUC: {results[name]['ROC-AUC']:.4f}")
    print(f"      F1-Score: {results[name]['F1-Score']:.4f}")

# 8. RESULTS
print("\n" + "="*60)
print("RESULTS")
print("="*60)
results_df = pd.DataFrame(results).T
print(results_df.round(4))

# 9. SAVE BEST MODEL
best_model_name = results_df['ROC-AUC'].idxmax()
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save everything
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(feature_engineer, 'models/feature_engineer.pkl')

print("\n✓ Saved models to models/ directory")
print("="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)