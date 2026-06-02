import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib

def train_baseline_xgb():
    print("Loading ML dataset...")
    df = pd.read_parquet('data/ml_dataset.parquet')
    
    # Feature Engineering (if any beyond what's extracted)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Drop NaNs
    features = ['fip_z', 'mom20_z', 'mom60_z', 'vol20_z', 'beta', 'rsi14', 'structure_score']
    df = df.dropna(subset=features + ['target_fwd_ret'])
    
    # Optional: Clip extreme returns to avoid training on massive gap outliers
    df['target_fwd_ret'] = df['target_fwd_ret'].clip(-0.3, 0.5)
    
    # Train-test split (Time Series Walk Forward style: Train < 2025, Test >= 2025)
    train = df[df['year'] < 2025]
    test = df[df['year'] >= 2025]
    
    X_train, y_train = train[features], train['target_fwd_ret']
    X_test, y_test = test[features], test['target_fwd_ret']
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("Training XGBoost Regressor...")
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.5f}")
    
    # Rank Correlation is more important than MSE for long-short
    test_results = pd.DataFrame({'true': y_test, 'pred': preds})
    corr = test_results['true'].corr(test_results['pred'], method='spearman')
    print(f"Test Spearman Rank Correlation: {corr:.3f}")
    
    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(importance)
    
    # Save the model
    os.makedirs('engine/ml/models', exist_ok=True)
    model.save_model('engine/ml/models/xgb_baseline.json')
    print("\nModel saved to engine/ml/models/xgb_baseline.json")

if __name__ == "__main__":
    train_baseline_xgb()
