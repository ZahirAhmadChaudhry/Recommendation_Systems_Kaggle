"""
LightGBM model module for recommendation system.

This module contains functions for training and making predictions with a LightGBM model.
"""

import lightgbm as lgb
from lightgbm.callback import early_stopping
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 10,
    verbose: bool = True
) -> lgb.Booster:
    """
    Train a LightGBM model for recommendation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: LightGBM parameters (uses default GPU params if None)
        early_stopping_rounds: Number of rounds for early stopping
        verbose: Whether to print verbose output
        
    Returns:
        Trained LightGBM model
    """
    # Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val)
    
    # Default GPU parameters if none provided
    if params is None:
        params = {
            'objective': 'binary',               # Binary classification
            'metric': 'binary_logloss',          # Evaluation metric
            'learning_rate': 0.1,                # Learning rate
            'num_leaves': 31,                    # Number of leaves
            'max_depth': -1,                     # Unlimited depth
            'device': 'gpu',                     # Enable GPU
            'gpu_use_dp': True,                  # Enable double-precision
            'max_bin': 255,                      # Increased bin size for flexibility
            'min_data_in_leaf': 50,              # Avoid overly specific splits
            'min_split_gain': 0.0                # Relax split constraints
        }
    
    if verbose:
        print("Training LightGBM model with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # Train model
    model = lgb.train(
        params=params,
        train_set=train_dataset,
        valid_sets=[val_dataset],
        callbacks=[early_stopping(stopping_rounds=early_stopping_rounds)]
    )
    
    if verbose:
        print("Model training completed.")
        print(f"Best iteration: {model.best_iteration}")
    
    return model

def predict_and_rank(
    model: lgb.Booster,
    test_data: pd.DataFrame,
    feature_columns: List[str],
    top_k: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make predictions and rank them to get top K recommendations.
    
    Args:
        model: Trained LightGBM model
        test_data: Test data DataFrame
        feature_columns: Feature columns to use for prediction
        top_k: Number of top recommendations to return
        
    Returns:
        Tuple of (test_data with predictions, top_k_recommendations)
    """
    # Make predictions
    X_test = test_data[feature_columns].fillna(0)
    preds = model.predict(X_test)
    test_data['predicted_score'] = preds
    
    # Rank predictions by customer
    test_data = test_data.sort_values(
        by=['customer_id', 'predicted_score'], 
        ascending=[True, False]
    )
    test_data['rank'] = test_data.groupby('customer_id').cumcount() + 1
    
    # Extract top K recommendations
    top_k_recommendations = test_data[test_data['rank'] <= top_k][
        ['customer_id', 'product_id', 'rank']
    ]
    
    return test_data, top_k_recommendations

def get_feature_importance(model: lgb.Booster, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from the trained model.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    # Get feature importance
    importance = model.feature_importance(importance_type='split')
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values(
        by='Importance', 
        ascending=False
    ).reset_index(drop=True)
    
    return feature_importance