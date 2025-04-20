"""
Utility functions for the LightGBM recommendation system.

This module contains utility functions for file handling, submission formatting,
and other miscellaneous tasks.
"""

import pandas as pd
import os
from typing import Dict, Any, Optional

def save_model(model, path: str, verbose: bool = True) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: The trained model to save
        path: Path where to save the model
        verbose: Whether to print information about the save
    """
    model.save_model(path)
    if verbose:
        print(f"Model saved to {path}")

def load_model(path: str, verbose: bool = True) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        path: Path to the saved model
        verbose: Whether to print information about the load
        
    Returns:
        The loaded model
    """
    import lightgbm as lgb
    model = lgb.Booster(model_file=path)
    if verbose:
        print(f"Model loaded from {path}")
    return model

def save_to_csv(df: pd.DataFrame, path: str, index: bool = False, verbose: bool = True) -> None:
    """
    Save a DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        path: Path where to save the CSV file
        index: Whether to save the DataFrame index
        verbose: Whether to print information about the save
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    df.to_csv(path, index=index)
    if verbose:
        print(f"DataFrame saved to {path}")

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the recommendation system.
    
    Returns:
        Dictionary containing default configuration
    """
    return {
        'data': {
            'interaction_features_path': '../Enhanced_Features/interaction_features.parquet',
            'training_data_path': 'training_data_with_negative_samples_from_interaction_features_gzip',
            'test_data_path': '../test_clean.csv'
        },
        'features': {
            'numeric_features': [
                "transaction_count",
                "unique_transaction_count",
                "total_quantity",
                "avg_quantity",
                "promo_ratio",
                "days_since_first",
                "days_since_last",
                "purchase_interval"
            ],
            'datetime_cols': ["first_purchase", "last_purchase"]
        },
        'model': {
            'params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'device': 'gpu',
                'gpu_use_dp': True,
                'max_bin': 255,
                'min_data_in_leaf': 50,
                'min_split_gain': 0.0
            },
            'early_stopping_rounds': 10,
            'test_size': 0.2,
            'random_state': 42
        },
        'evaluation': {
            'top_k': 10
        },
        'submission': {
            'customer_range': (80001, 100000),
            'output_path': 'submission/submission_list.csv'
        }
    }

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Base configuration dictionary
        updates: Dictionary containing updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    # Deep copy the config to avoid modifying the original
    import copy
    result = copy.deepcopy(config)
    
    # Helper function to recursively update nested dictionaries
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    # Apply updates
    result = update_nested_dict(result, updates)
    
    return result