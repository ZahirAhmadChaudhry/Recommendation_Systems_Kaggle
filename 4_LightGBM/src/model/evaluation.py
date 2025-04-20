"""
Evaluation module for recommendation system.

This module contains functions for evaluating recommendation systems,
including hitrate calculation and other metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional

def hitrate_at_k(
    true_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    k: int = 10
) -> float:
    """
    Calculate the hitrate at k for recommendations.
    
    This function assesses how relevant the k product recommendations are.
    It calculates the proportion of recommended products that are actually
    purchased by the customer.
    
    Args:
        true_data: DataFrame containing the true data with columns:
            - customer_id: the customer identifier
            - product_id: the product identifier that was purchased in the test set
        predicted_data: DataFrame containing the predicted data with columns:
            - customer_id: the customer identifier
            - product_id: the product identifier that was recommended
            - rank: the rank of the recommendation (should be between 1 and k)
        k: Number of recommendations to consider (between 1 and 10)
    
    Returns:
        The hitrate at k (float between 0 and 1)
    """
    # Merge true and predicted data
    data = pd.merge(
        left=true_data,
        right=predicted_data,
        how="left",
        on=["customer_id", "product_id"]
    )
    
    # Filter to only consider top k recommendations
    df = data[data["rank"] <= k]
    
    # Count non-null ranks for each customer (hits)
    non_null_counts = df.groupby('customer_id')['rank'] \
        .apply(lambda x: x.notna().sum()) \
        .reset_index(name='non_null_count')
    
    # Count distinct products per customer in true data
    distinct_products_per_customer = data.groupby('customer_id')['product_id'] \
        .nunique() \
        .reset_index(name='distinct_product_count')
    
    # Merge the counts
    df = pd.merge(
        left=distinct_products_per_customer,
        right=non_null_counts,
        how="left",
        on="customer_id"
    )
    
    # Calculate denominator as min(distinct products, k)
    df["denominator"] = [min(df.iloc[i].distinct_product_count, k) for i in range(len(df))]
    
    # Fill missing values with 0
    df = df.fillna(0)
    
    # Calculate hitrate
    return (df["non_null_count"] / df["denominator"]).mean()

def precision_at_k(
    true_data: pd.DataFrame,
    predicted_data: pd.DataFrame,
    k: int = 10
) -> float:
    """
    Calculate precision at k for recommendations.
    
    Args:
        true_data: DataFrame with true purchases
        predicted_data: DataFrame with predictions
        k: Number of recommendations to consider
    
    Returns:
        Precision at k
    """
    # Filter predictions to top k
    top_k_pred = predicted_data[predicted_data['rank'] <= k]
    
    # Merge with true data
    merged = pd.merge(
        top_k_pred,
        true_data,
        on=['customer_id', 'product_id'],
        how='left',
        indicator=True
    )
    
    # Mark hits (predictions that match true data)
    merged['hit'] = (merged['_merge'] == 'both').astype(int)
    
    # Calculate precision for each user
    user_precision = merged.groupby('customer_id')['hit'].mean()
    
    # Return average precision
    return user_precision.mean()

def format_submission(
    predictions: pd.DataFrame,
    customer_range: Optional[tuple] = None
) -> pd.DataFrame:
    """
    Format predictions for submission.
    
    Args:
        predictions: DataFrame with predictions (customer_id, product_id, rank)
        customer_range: Optional tuple of (min_id, max_id) to filter customers
    
    Returns:
        Formatted DataFrame ready for submission
    """
    # Make a copy to avoid modifying the original
    df = predictions.copy()
    
    # Filter by customer range if specified
    if customer_range:
        min_id, max_id = customer_range
        df = df[df.customer_id.between(min_id, max_id)]
    
    # Clean column names
    df.columns = df.columns.str.replace('+AF8-', '_', regex=False)
    
    # Clean string values
    df = df.replace(r'\\+AF8-', '_', regex=True)
    
    # Clean and convert IDs if needed
    for col in ['customer_id', 'product_id']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.extract('(\\d+)').fillna(11).astype(int)
    
    # Add ID column
    df['id'] = df.index
    df = df[['id'] + [col for col in df.columns if col != 'id']]
    
    # Group by customer_id to format as required for submission
    prediction_grouped = df.groupby('customer_id').agg({
        'product_id': lambda x: ','.join(map(str, x)),
        'rank': lambda x: ','.join(map(str, x))
    }).reset_index()
    
    # Add id column
    prediction_grouped.insert(0, 'id', range(len(prediction_grouped)))
    
    # Validation checks
    valid_submission = True
    for index, row in prediction_grouped.iterrows():
        # Check ranks
        ranks = list(map(int, row['rank'].split(',')))
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            print("Error: Ranks must be distinct and sequential.")
            valid_submission = False
            break
            
        # Check product duplicates
        products = row['product_id'].split(',')
        if len(products) != len(set(products)):
            print("Error: Products must be unique for each customer.")
            valid_submission = False
            break
    
    if not valid_submission:
        return None
    
    return prediction_grouped