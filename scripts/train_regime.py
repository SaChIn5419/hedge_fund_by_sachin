"""
Train XGBoost Regime Classifier + HMM Transition Smoother
Replaces the hand-tuned logistic regime model with a learned ML pipeline.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_regime_classifier():
    print("Loading regime dataset...")
    df = pd.read_parquet('data/regime_dataset.parquet')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Features (exclude target columns and forward-looking data)
    exclude_cols = ['regime_label', 'fwd_ret_5']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Encode labels
    label_map = {'BULL': 0, 'CHOP': 1, 'BEAR': 2}
    reverse_map = {0: 'BULL', 1: 'CHOP', 2: 'BEAR'}
    df['label'] = df['regime_label'].map(label_map)
    
    # Time-based split: Train < 2025, Test >= 2025
    train = df[df.index.year < 2025]
    test = df[df.index.year >= 2025]
    
    X_train, y_train = train[feature_cols], train['label']
    X_test, y_test = test[feature_cols], test['label']
    
    print(f"Training: {len(X_train)} samples ({train.index.min().date()} to {train.index.max().date()})")
    print(f"Testing:  {len(X_test)} samples ({test.index.min().date()} to {test.index.max().date()})")
    print(f"\nTrain label distribution:")
    print(train['regime_label'].value_counts())
    print(f"\nTest label distribution:")
    print(test['regime_label'].value_counts())
    
    # Train XGBoost Multi-class Classifier
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,            # Shallow to prevent overfitting
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,          # L1 regularization
        reg_lambda=1.0,         # L2 regularization
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
    )
    
    print("\nTraining XGBoost Regime Classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    
    print("\n" + "=" * 60)
    print("  TRAIN SET CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_train, train_preds, target_names=['BULL', 'CHOP', 'BEAR']))
    
    print("=" * 60)
    print("  TEST SET CLASSIFICATION REPORT (2025+)")
    print("=" * 60)
    print(classification_report(y_test, test_preds, target_names=['BULL', 'CHOP', 'BEAR']))
    
    print("Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, test_preds)
    print(f"              Predicted")
    print(f"              BULL  CHOP  BEAR")
    for i, label in enumerate(['Actual BULL', 'Actual CHOP', 'Actual BEAR']):
        print(f"  {label}  {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}")
    
    # Feature Importance
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(f"\nTop 15 Feature Importance:")
    print(importance.head(15).to_string(index=False))
    
    # Save model
    os.makedirs('models/regime', exist_ok=True)
    model.save_model('models/regime/xgb_regime.json')
    joblib.dump(feature_cols, 'models/regime/regime_feature_cols.pkl')
    print(f"\nModel saved to models/regime/xgb_regime.json")
    
    # ==========================================
    # TRAIN HMM TRANSITION SMOOTHER
    # ==========================================
    print("\n" + "=" * 60)
    print("  TRAINING HMM TRANSITION SMOOTHER")
    print("=" * 60)
    
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("hmmlearn not installed. Installing...")
        os.system(f'{sys.executable} -m pip install hmmlearn -q')
        from hmmlearn.hmm import GaussianHMM
    
    # Feed the XGBoost probabilities into the HMM
    # The HMM will learn the natural transition dynamics
    all_probs = model.predict_proba(df[feature_cols])
    # Add small noise to break degeneracy (probabilities sum to 1, making covariance singular)
    all_probs_noisy = all_probs + np.random.RandomState(42).randn(*all_probs.shape) * 0.01
    all_probs_noisy = np.clip(all_probs_noisy, 0.001, 0.999)
    
    hmm = GaussianHMM(
        n_components=3,         # 3 hidden states (BULL, CHOP, BEAR)
        covariance_type='diag', # Diagonal covariance to avoid singular matrix
        n_iter=200,
        random_state=42,
        init_params='mc',       # Initialize means and covariances from data
        params='stmc',          # Learn start probs, transitions, means, covariances
    )
    
    # Fit on the probability sequences
    hmm.fit(all_probs_noisy)
    
    # Get smoothed regime predictions
    hmm_states = hmm.predict(all_probs)
    hmm_probs = hmm.predict_proba(all_probs)
    
    # Map HMM states to regime labels by matching to the most common XGBoost label
    state_to_regime = {}
    for state in range(3):
        mask = hmm_states == state
        if mask.sum() > 0:
            most_common_label = df['label'].values[mask]
            counts = np.bincount(most_common_label.astype(int), minlength=3)
            state_to_regime[state] = reverse_map[np.argmax(counts)]
        else:
            state_to_regime[state] = 'CHOP'
    
    print(f"HMM State Mapping: {state_to_regime}")
    print(f"\nHMM Transition Matrix:")
    trans = hmm.transmat_
    labels = [state_to_regime[i] for i in range(3)]
    print(f"        {'  '.join(f'{l:>6s}' for l in labels)}")
    for i in range(3):
        print(f"  {labels[i]:>5s}  {'  '.join(f'{trans[i][j]:6.3f}' for j in range(3))}")
    
    # Evaluate HMM smoothed predictions
    hmm_regime_preds = pd.Series([state_to_regime[s] for s in hmm_states], index=df.index)
    
    # Count flip-flops
    xgb_regimes = pd.Series([reverse_map[p] for p in model.predict(df[feature_cols])], index=df.index)
    xgb_flips = (xgb_regimes != xgb_regimes.shift(1)).sum()
    hmm_flips = (hmm_regime_preds != hmm_regime_preds.shift(1)).sum()
    
    print(f"\nRegime Flip-Flops:")
    print(f"  Raw XGBoost:    {xgb_flips} flips")
    print(f"  HMM Smoothed:   {hmm_flips} flips")
    print(f"  Reduction:      {(1 - hmm_flips/xgb_flips)*100:.0f}%")
    
    # Save HMM
    joblib.dump({
        'hmm': hmm,
        'state_to_regime': state_to_regime,
    }, 'models/regime/hmm_smoother.pkl')
    print(f"HMM saved to models/regime/hmm_smoother.pkl")
    
    # ==========================================
    # BACKTEST COMPARISON
    # ==========================================
    print("\n" + "=" * 60)
    print("  BACKTEST: OLD vs NEW REGIME CLASSIFIER")
    print("=" * 60)
    
    # Load old regime trace
    old_regime = pd.read_csv('data/regime_trace_chimera_fip.csv')
    old_regime['date'] = pd.to_datetime(old_regime['date'])
    
    # Compare year by year
    for year in sorted(df.index.year.unique()):
        yr_mask = df.index.year == year
        yr_old = old_regime[old_regime['date'].dt.year == year]
        
        old_flips = (yr_old['regime'] != yr_old['regime'].shift(1)).sum() if len(yr_old) > 0 else 0
        
        new_raw = xgb_regimes[yr_mask]
        new_hmm = hmm_regime_preds[yr_mask]
        new_raw_flips = (new_raw != new_raw.shift(1)).sum()
        new_hmm_flips = (new_hmm != new_hmm.shift(1)).sum()
        
        print(f"  {year}: Old={old_flips} flips | XGB={new_raw_flips} flips | HMM={new_hmm_flips} flips")


if __name__ == '__main__':
    train_regime_classifier()
