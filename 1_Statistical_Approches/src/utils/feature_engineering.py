"""
Feature engineering module for generating features for recommendation models.
Includes functions for creating interaction features, product features, and temporal features.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional

class FeatureEngineer:
    """
    Utility class for generating features for recommendation systems.
    Creates user-product interaction features, product features, and temporal features.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize FeatureEngineer with logging configuration.
        
        Args:
            log_level: Logging level (default: INFO)
        """
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_interaction_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-product interaction features from train data.
        
        Args:
            train_df: Training DataFrame with user-product interactions
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Creating interaction features...")
        
        # Convert date to datetime if not already
        if train_df['date'].dtype != 'datetime64[ns]':
            train_df['date'] = pd.to_datetime(train_df['date'])
        
        # Group by customer and product
        interaction_features = (train_df
            .groupby(['customer_id', 'product_id'])
            .agg({
                'transaction_id': ['count', 'nunique'],  # frequency and unique transactions
                'quantity': ['sum', 'mean'],            # quantity patterns
                'is_promo': 'mean',                     # promo purchase ratio
                'date': ['min', 'max']                  # first and last purchase
            })
            .reset_index()
        )
        
        # Flatten column names
        interaction_features.columns = [
            'customer_id', 'product_id',
            'transaction_count', 'unique_transaction_count',
            'total_quantity', 'avg_quantity',
            'promo_ratio',
            'first_purchase', 'last_purchase'
        ]
        
        # Calculate days since first and last purchase
        latest_date = train_df['date'].max()
        interaction_features['days_since_first'] = (
            latest_date - interaction_features['first_purchase']).dt.days
        interaction_features['days_since_last'] = (
            latest_date - interaction_features['last_purchase']).dt.days
        
        # Calculate purchase frequency (in days)
        interaction_features['purchase_interval'] = (
            interaction_features['days_since_first'] / 
            np.maximum(interaction_features['unique_transaction_count'], 1)
        )
        
        # Calculate recency score (exponential decay)
        interaction_features['recency_score'] = np.exp(-0.1 * interaction_features['days_since_last'])
        
        # Calculate frequency score (normalized)
        interaction_features['frequency_score'] = interaction_features.groupby('customer_id')['transaction_count'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        
        # Create hybrid score combining recency and frequency
        interaction_features['hybrid_score'] = (
            0.7 * interaction_features['frequency_score'] + 
            0.3 * interaction_features['recency_score']
        )
        
        self.logger.info(f"Created interaction features: {len(interaction_features)} rows, {len(interaction_features.columns)} columns")
        return interaction_features
    
    def create_product_features(self, products_df: pd.DataFrame, interaction_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create product features combining categorical and binary information.
        
        Args:
            products_df: Products DataFrame
            interaction_features: Interaction features DataFrame
            
        Returns:
            DataFrame with product features
        """
        self.logger.info("Creating product features...")
        
        # Calculate product popularity metrics from interactions
        product_popularity = (interaction_features
            .groupby('product_id')
            .agg({
                'customer_id': 'count',          # number of unique customers
                'transaction_count': 'sum',      # total transactions
                'total_quantity': 'sum',         # total quantity sold
                'promo_ratio': 'mean'           # average promo ratio
            })
            .rename(columns={
                'customer_id': 'customer_count',
                'transaction_count': 'total_transactions',
                'total_quantity': 'total_quantity_sold',
                'promo_ratio': 'avg_promo_ratio'
            })
        )
        
        # Normalize popularity metrics
        for col in product_popularity.columns:
            product_popularity[col] = product_popularity[col] / (product_popularity[col].max() + 1e-8)
        
        # Create category embeddings using mean encoding
        self.logger.info("Creating category embeddings...")
        
        def create_category_embedding(df, column, interaction_stats):
            """Create embeddings for categorical columns using mean encoding"""
            # Handle null values in the categorical column
            if df[column].isnull().any():
                df = df.copy()
                df[column] = df[column].fillna('UNKNOWN')
                
            # Merge and aggregate to get category statistics
            embedding = (interaction_stats
                .reset_index()
                .merge(df[['product_id', column]], on='product_id', how='left')
                .groupby(column)
                .agg({
                    'customer_count': 'mean',
                    'total_transactions': 'mean',
                    'total_quantity_sold': 'mean',
                    'avg_promo_ratio': 'mean'
                })
            )
            
            # Normalize embeddings
            for col in embedding.columns:
                col_min = embedding[col].min()
                col_max = embedding[col].max()
                if col_max > col_min:
                    embedding[col] = (embedding[col] - col_min) / (col_max - col_min)
                else:
                    embedding[col] = 0.0
                    
            return embedding
        
        # Combine all features
        self.logger.info("Combining product features...")
        
        # Start with binary features
        binary_columns = ['bio', 'fresh', 'frozen', 'national_brand', 
                         'carrefour_brand', 'first_price_brand']
        
        # Ensure all expected binary columns exist
        for col in binary_columns:
            if col not in products_df.columns:
                self.logger.warning(f"Binary column {col} not found in products data")
                binary_columns.remove(col)
        
        product_features = products_df[['product_id'] + binary_columns].copy()
        
        # Add popularity metrics
        product_features = product_features.merge(
            product_popularity, 
            on='product_id', 
            how='left'
        )
        
        # Handle NaN values in popularity metrics
        for col in product_popularity.columns:
            product_features[col] = product_features[col].fillna(0)
        
        # Add category embeddings
        categorical_columns = ['department_key', 'class_key', 'subclass_key', 'brand_key', 'sector']
        
        # Ensure all expected categorical columns exist
        for col in categorical_columns.copy():
            if col not in products_df.columns:
                self.logger.warning(f"Categorical column {col} not found in products data")
                categorical_columns.remove(col)
        
        for col in categorical_columns:
            # Create embedding
            embedding = create_category_embedding(products_df, col, product_popularity)
            
            # Merge with product features
            temp_df = products_df[['product_id', col]].merge(
                embedding.reset_index(), 
                on=col, 
                how='left'
            )
            
            # Rename columns with prefix
            rename_dict = {
                c: f'{col}_{c}' for c in embedding.columns
            }
            temp_df = temp_df.rename(columns=rename_dict)
            
            # Merge with main features
            product_features = product_features.merge(
                temp_df.drop(columns=[col]), 
                on='product_id', 
                how='left'
            )
            
            # Handle NaN values
            for new_col in rename_dict.values():
                product_features[new_col] = product_features[new_col].fillna(0)
        
        self.logger.info(f"Created product features: {len(product_features)} rows, {len(product_features.columns)} columns")
        return product_features
    
    def create_temporal_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from transaction dates.
        
        Args:
            train_df: Training DataFrame with transaction dates
            
        Returns:
            DataFrame with temporal features
        """
        self.logger.info("Creating temporal features...")
        
        # Ensure date is in datetime format
        if train_df['date'].dtype != 'datetime64[ns]':
            train_df['date'] = pd.to_datetime(train_df['date'])
        
        # Create a copy to avoid modifying the original
        df = train_df.copy()
        
        # Extract basic date components
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day
        
        # Calculate days since first purchase per customer
        customer_first_purchase = df.groupby('customer_id')['date'].min().reset_index()
        customer_first_purchase.columns = ['customer_id', 'first_purchase_date']
        df = df.merge(customer_first_purchase, on='customer_id', how='left')
        df['days_since_first_purchase'] = (df['date'] - df['first_purchase_date']).dt.days
        
        # Calculate purchase sequence number for each customer
        df['purchase_seq_num'] = df.groupby('customer_id')['date'].rank(method='dense')
        
        # Calculate days since last purchase
        df = df.sort_values(['customer_id', 'date'])
        df['prev_purchase_date'] = df.groupby('customer_id')['date'].shift(1)
        df['days_since_prev_purchase'] = (df['date'] - df['prev_purchase_date']).dt.days
        
        # Fill NaN values for first purchases
        df['days_since_prev_purchase'] = df['days_since_prev_purchase'].fillna(0)
        
        # Calculate temporal statistics by customer
        temporal_features = (df
            .groupby('customer_id')
            .agg({
                'days_since_prev_purchase': ['mean', 'std', 'max'],
                'is_weekend': 'mean',
                'purchase_seq_num': 'max'
            })
            .reset_index()
        )
        
        # Flatten column names
        temporal_features.columns = [
            'customer_id',
            'avg_days_between_purchases', 'std_days_between_purchases', 'max_days_between_purchases',
            'weekend_purchase_ratio', 'total_purchases'
        ]
        
        self.logger.info(f"Created temporal features: {len(temporal_features)} rows, {len(temporal_features.columns)} columns")
        return temporal_features
    
    def analyze_data_dimensions(self, interaction_features: pd.DataFrame, product_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the dimensions and potential memory requirements of the data.
        
        Args:
            interaction_features: Interaction features DataFrame
            product_features: Product features DataFrame
            
        Returns:
            Dictionary with dimension analysis
        """
        self.logger.info("Analyzing data dimensions...")
        
        n_customers = interaction_features['customer_id'].nunique()
        n_products = product_features['product_id'].nunique()
        n_interactions = len(interaction_features)
        
        # Calculate potential memory usage
        matrix_size = n_customers * n_products * 8 / (1024**3)  # Size in GB if dense
        sparse_size = n_interactions * 16 / (1024**3)  # Approximate size in GB if sparse (8 bytes for data + 8 for indices)
        
        analysis = {
            "n_customers": n_customers,
            "n_products": n_products,
            "n_interactions": n_interactions,
            "density": n_interactions / (n_customers * n_products),
            "dense_matrix_size_gb": matrix_size,
            "sparse_matrix_size_gb": sparse_size
        }
        
        self.logger.info(f"Data dimensions: {n_customers:,} customers, {n_products:,} products, {n_interactions:,} interactions")
        self.logger.info(f"Matrix density: {(n_interactions / (n_customers * n_products)):.6%}")
        self.logger.info(f"Dense matrix size: {matrix_size:.2f} GB, Sparse matrix size: {sparse_size:.2f} GB")
        
        return analysis
    
    def save_features(self, features_df: pd.DataFrame, feature_type: str, output_dir: Union[str, Path]) -> Path:
        """
        Save generated features to parquet file.
        
        Args:
            features_df: Features DataFrame to save
            feature_type: Type of features (interaction, product, temporal)
            output_dir: Directory to save the file
            
        Returns:
            Path to saved file
        """
        # Convert output_dir to Path if it's a string
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file path
        file_path = output_dir / f"{feature_type}_features.parquet"
        
        # Save to parquet
        self.logger.info(f"Saving {feature_type} features to {file_path}")
        features_df.to_parquet(file_path, index=False)
        
        return file_path
    
    def generate_all_features(self, train_df: pd.DataFrame, products_df: pd.DataFrame, 
                             output_dir: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate and save all features in one go.
        
        Args:
            train_df: Training DataFrame
            products_df: Products DataFrame
            output_dir: Directory to save features (optional)
            
        Returns:
            Dictionary with all generated feature DataFrames
        """
        self.logger.info("Generating all features...")
        
        # Generate interaction features
        interaction_features = self.create_interaction_features(train_df)
        
        # Generate product features
        product_features = self.create_product_features(products_df, interaction_features)
        
        # Generate temporal features
        temporal_features = self.create_temporal_features(train_df)
        
        # Analyze data dimensions
        self.analyze_data_dimensions(interaction_features, product_features)
        
        # Save features if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
            self.save_features(interaction_features, "interaction", output_dir)
            self.save_features(product_features, "product", output_dir)
            self.save_features(temporal_features, "temporal", output_dir)
        
        # Return all feature DataFrames
        return {
            "interaction": interaction_features,
            "product": product_features,
            "temporal": temporal_features
        }