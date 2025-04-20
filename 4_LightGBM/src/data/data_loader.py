"""
Data loading and preprocessing module for LightGBM recommendation model.

This module contains functions for loading, merging, and preprocessing data for
the LightGBM recommendation model.
"""

import pandas as pd
import pyarrow.parquet as pq
from typing import List, Tuple, Optional
import numpy as np

def load_interaction_features(path: str) -> pd.DataFrame:
    """
    Load the interaction features from a parquet file.
    
    Args:
        path: Path to the interaction features parquet file
        
    Returns:
        DataFrame containing interaction features
    """
    print(f"Loading interaction features from {path}")
    interaction_features = pd.read_parquet(path)
    return interaction_features

def process_large_training_data(
    interaction_features: pd.DataFrame,
    training_data_path: str,
    numeric_features: List[str],
    datetime_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Process large training data file by reading it in chunks and merging with interaction features.
    
    Args:
        interaction_features: DataFrame containing interaction features
        training_data_path: Path to the training data parquet file
        numeric_features: List of numeric feature columns to keep
        datetime_cols: Optional list of datetime columns to drop
        
    Returns:
        Processed training data DataFrame
    """
    # For faster merging, set an index
    interaction_features.set_index(["customer_id", "product_id"], inplace=True)
    
    # Get Parquet metadata
    pf = pq.ParquetFile(training_data_path)
    num_row_groups = pf.metadata.num_row_groups
    print(f"Number of row groups in {training_data_path}: {num_row_groups}")
    
    # Process each chunk
    chunks = []
    for rg_index in range(num_row_groups):
        print(f"\nReading row group {rg_index + 1}/{num_row_groups}...")
        
        # Read just this row group, with only needed columns
        table = pf.read_row_group(
            rg_index,
            columns=["customer_id", "product_id", "target"]
        )
        chunk_df = table.to_pandas()
        
        # Merge chunk with interaction features
        chunk_merged = chunk_df.merge(
            interaction_features,
            how="left",
            left_on=["customer_id", "product_id"],
            right_index=True
        )
        
        # Fill numeric features with 0 for missing values
        chunk_merged[numeric_features] = chunk_merged[numeric_features].fillna(0)
        
        # Drop datetime columns if specified
        if datetime_cols:
            for col in datetime_cols:
                if col in chunk_merged.columns:
                    chunk_merged.drop(columns=[col], inplace=True)
        
        chunks.append(chunk_merged)
    
    # Concatenate all chunks
    training_data = pd.concat(chunks, ignore_index=True)
    print("Final merged shape:", training_data.shape)
    
    return training_data

def load_test_data(
    test_path: str,
    interaction_features_path: str,
    needed_columns: List[str]
) -> pd.DataFrame:
    """
    Load and preprocess test data by merging with interaction features.
    
    Args:
        test_path: Path to the test data file
        interaction_features_path: Path to the interaction features file
        needed_columns: List of feature columns needed for prediction
        
    Returns:
        Processed test data DataFrame
    """
    print(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path, compression='gzip')
    
    print(f"Loading interaction features from {interaction_features_path}")
    interaction_features = pd.read_parquet(interaction_features_path)
    
    # Merge test data with interaction features
    test_data = test_data.merge(
        interaction_features, 
        on=['customer_id', 'product_id'], 
        how='left'
    )
    
    # Fill missing feature values with 0
    test_data[needed_columns] = test_data[needed_columns].fillna(0)
    
    return test_data

def prepare_train_test_split(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare train-test split for model training.
    
    Args:
        data: DataFrame containing features and target
        feature_columns: List of feature columns to use
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    from sklearn.model_selection import train_test_split
    
    X = data[feature_columns]
    y = data[target_column].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val