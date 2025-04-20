"""
Statistical recommendation models module for generating recommendations.
Implements various statistical approaches to product recommendations.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

class StatisticalRecommender:
    """
    Collection of statistical recommendation models.
    Includes frequency-based, recency-based, and hybrid approaches.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize StatisticalRecommender with logging configuration.
        
        Args:
            log_level: Logging level (default: INFO)
        """
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def frequency_based_recommendations(self, 
                                        train_df: pd.DataFrame, 
                                        customers: Union[List[int], np.ndarray],
                                        top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations based on purchase frequency.
        
        Args:
            train_df: Training DataFrame with purchase history
            customers: List of customer IDs to generate recommendations for
            top_k: Number of recommendations per customer
            
        Returns:
            DataFrame with customer_id, product_id, and rank columns
        """
        self.logger.info(f"Generating frequency-based recommendations for {len(customers)} customers...")
        
        # Calculate purchase frequency
        freq_df = (train_df
                  .groupby(['customer_id', 'product_id'])
                  .size()
                  .reset_index(name='frequency'))
        
        # Rank products by frequency for each customer
        freq_df['rank'] = freq_df.groupby('customer_id')['frequency'].rank(
            method='dense', ascending=False).astype(int)
        
        # Filter top-k recommendations for each customer
        recommendations = freq_df[freq_df['rank'] <= top_k]
        
        # Filter for specified customers
        if customers is not None:
            recommendations = recommendations[recommendations['customer_id'].isin(customers)]
        
        self.logger.info(f"Generated {len(recommendations)} recommendations for {recommendations['customer_id'].nunique()} customers")
        return recommendations[['customer_id', 'product_id', 'rank']]
    
    def recency_based_recommendations(self,
                                     train_df: pd.DataFrame,
                                     customers: Union[List[int], np.ndarray],
                                     top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations based on purchase recency.
        
        Args:
            train_df: Training DataFrame with purchase history
            customers: List of customer IDs to generate recommendations for
            top_k: Number of recommendations per customer
            
        Returns:
            DataFrame with customer_id, product_id, and rank columns
        """
        self.logger.info(f"Generating recency-based recommendations for {len(customers)} customers...")
        
        # Convert date to datetime if not already
        if train_df['date'].dtype != 'datetime64[ns]':
            train_df = train_df.copy()
            train_df['date'] = pd.to_datetime(train_df['date'])
        
        # Get most recent purchase for each customer-product pair
        recency_df = (train_df
                     .sort_values('date')
                     .groupby(['customer_id', 'product_id'])
                     .agg({'date': 'max'})
                     .reset_index())
        
        # Rank products by recency for each customer (most recent first)
        recency_df['rank'] = recency_df.groupby('customer_id')['date'].rank(
            method='dense', ascending=False).astype(int)
        
        # Filter top-k recommendations for each customer
        recommendations = recency_df[recency_df['rank'] <= top_k]
        
        # Filter for specified customers
        if customers is not None:
            recommendations = recommendations[recommendations['customer_id'].isin(customers)]
        
        self.logger.info(f"Generated {len(recommendations)} recommendations for {recommendations['customer_id'].nunique()} customers")
        return recommendations[['customer_id', 'product_id', 'rank']]
    
    def popularity_based_recommendations(self,
                                        train_df: pd.DataFrame,
                                        customers: Union[List[int], np.ndarray],
                                        top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations based on overall product popularity.
        
        Args:
            train_df: Training DataFrame with purchase history
            customers: List of customer IDs to generate recommendations for
            top_k: Number of recommendations per customer
            
        Returns:
            DataFrame with customer_id, product_id, and rank columns
        """
        self.logger.info(f"Generating popularity-based recommendations for {len(customers)} customers...")
        
        # Calculate product popularity (number of unique customers who bought it)
        popularity_df = (train_df
                        .groupby('product_id')
                        ['customer_id'].nunique()
                        .reset_index(name='popularity')
                        .sort_values('popularity', ascending=False)
                        .head(top_k)
                        .reset_index(drop=True))
        
        # Set ranks
        popularity_df['rank'] = popularity_df.index + 1
        
        # Create recommendations for all customers
        recommendations = []
        for customer_id in customers:
            for _, row in popularity_df.iterrows():
                recommendations.append({
                    'customer_id': customer_id,
                    'product_id': row['product_id'],
                    'rank': row['rank']
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        self.logger.info(f"Generated {len(recommendations_df)} recommendations for {len(customers)} customers")
        return recommendations_df
    
    def hybrid_recommendations(self,
                              train_df: pd.DataFrame,
                              customers: Union[List[int], np.ndarray],
                              freq_weight: float = 0.7,
                              recency_weight: float = 0.3,
                              top_k: int = 10) -> pd.DataFrame:
        """
        Generate hybrid recommendations combining frequency and recency.
        
        Args:
            train_df: Training DataFrame with purchase history
            customers: List of customer IDs to generate recommendations for
            freq_weight: Weight to assign to frequency score
            recency_weight: Weight to assign to recency score
            top_k: Number of recommendations per customer
            
        Returns:
            DataFrame with customer_id, product_id, and rank columns
        """
        self.logger.info(f"Generating hybrid recommendations for {len(customers)} customers...")
        
        # Ensure date is datetime
        if train_df['date'].dtype != 'datetime64[ns]':
            train_df = train_df.copy()
            train_df['date'] = pd.to_datetime(train_df['date'])
        
        # Calculate frequency metrics
        freq_df = (train_df
                  .groupby(['customer_id', 'product_id'])
                  ['transaction_id'].count()
                  .reset_index(name='frequency'))
        
        # Calculate recency metrics
        recency_df = (train_df
                     .sort_values('date')
                     .groupby(['customer_id', 'product_id'])
                     .agg({'date': 'max'})
                     .reset_index())
        
        # Calculate days since last purchase
        latest_date = train_df['date'].max()
        recency_df['days_since'] = (latest_date - recency_df['date']).dt.days
        
        # Combine metrics
        combined_df = freq_df.merge(
            recency_df[['customer_id', 'product_id', 'days_since']],
            on=['customer_id', 'product_id']
        )
        
        # Normalize scores within each customer
        combined_df['freq_norm'] = combined_df.groupby('customer_id')['frequency'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        
        combined_df['recency_norm'] = combined_df.groupby('customer_id')['days_since'].transform(
            lambda x: 1 - (x - x.min()) / (x.max() - x.min() + 1e-8))
        
        # Calculate weighted score
        combined_df['score'] = (
            freq_weight * combined_df['freq_norm'] +
            recency_weight * combined_df['recency_norm']
        )
        
        # Rank by score
        combined_df['rank'] = combined_df.groupby('customer_id')['score'].rank(
            method='dense', ascending=False).astype(int)
        
        # Filter to top-k recommendations
        recommendations = combined_df[combined_df['rank'] <= top_k]
        
        # Filter to specified customers
        if customers is not None:
            recommendations = recommendations[recommendations['customer_id'].isin(customers)]
        
        self.logger.info(f"Generated {len(recommendations)} recommendations for {recommendations['customer_id'].nunique()} customers")
        return recommendations[['customer_id', 'product_id', 'rank']]
    
    def customer_segmented_recommendations(self,
                                          train_df: pd.DataFrame,
                                          customers: Union[List[int], np.ndarray],
                                          products_df: Optional[pd.DataFrame] = None,
                                          n_segments: int = 3,
                                          top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations based on customer segments.
        
        Args:
            train_df: Training DataFrame with purchase history
            customers: List of customer IDs to generate recommendations for
            products_df: Optional products DataFrame for additional features
            n_segments: Number of customer segments to create
            top_k: Number of recommendations per customer
            
        Returns:
            DataFrame with customer_id, product_id, and rank columns
        """
        self.logger.info(f"Generating segmented recommendations for {len(customers)} customers...")
        
        # Create customer purchase profiles
        customer_profiles = (train_df
                           .groupby('customer_id')
                           .agg({
                               'transaction_id': 'nunique',
                               'product_id': 'nunique',
                               'quantity': ['sum', 'mean'],
                               'is_promo': 'mean'
                           })
                           .reset_index())
        
        # Flatten column names
        customer_profiles.columns = [
            'customer_id', 'num_transactions', 'num_unique_products', 
            'total_quantity', 'avg_quantity', 'promo_ratio'
        ]
        
        # Add category purchase ratios if products_df is provided
        if products_df is not None:
            # Merge product categories with purchases
            category_purchases = train_df.merge(
                products_df[['product_id', 'department_key']],
                on='product_id',
                how='left'
            )
            
            # Calculate category purchase ratios
            category_ratios = category_purchases.groupby(
                ['customer_id', 'department_key']
            ).size().unstack(fill_value=0)
            
            # Normalize to get ratios
            category_ratios = category_ratios.div(category_ratios.sum(axis=1), axis=0)
            
            # Merge with customer profiles
            customer_profiles = customer_profiles.merge(
                category_ratios.reset_index(),
                on='customer_id',
                how='left'
            )
            
            # Fill NaN values
            category_cols = category_ratios.columns.tolist()
            customer_profiles[category_cols] = customer_profiles[category_cols].fillna(0)
        
        # Create simple segments based on purchase frequency and volume
        customer_profiles['purchase_volume'] = (
            customer_profiles['num_transactions'] * customer_profiles['avg_quantity']
        )
        
        # Segment customers based on purchase volume
        customer_profiles['segment'] = pd.qcut(
            customer_profiles['purchase_volume'],
            q=n_segments,
            labels=[f'segment_{i}' for i in range(1, n_segments+1)]
        )
        
        # Generate recommendations for each segment
        recommendations = []
        
        for segment in customer_profiles['segment'].unique():
            # Get customers in this segment
            segment_customers = customer_profiles[
                customer_profiles['segment'] == segment
            ]['customer_id'].values
            
            # Get purchases for this segment
            segment_purchases = train_df[
                train_df['customer_id'].isin(segment_customers)
            ]
            
            # Get popular products for this segment
            segment_products = (segment_purchases
                              .groupby('product_id')
                              .size()
                              .reset_index(name='count')
                              .sort_values('count', ascending=False)
                              .head(top_k))
            
            segment_products['rank'] = segment_products.index + 1
            
            # Generate recommendations for customers in the target list
            for customer_id in set(customers) & set(segment_customers):
                for _, row in segment_products.iterrows():
                    recommendations.append({
                        'customer_id': customer_id,
                        'product_id': row['product_id'],
                        'rank': row['rank']
                    })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        # If we missed any customers (could happen with a small number of segments),
        # fall back to hybrid recommendations for those customers
        missing_customers = set(customers) - set(recommendations_df['customer_id'].unique())
        
        if missing_customers:
            self.logger.warning(f"Missing recommendations for {len(missing_customers)} customers. Using fallback.")
            fallback_recs = self.hybrid_recommendations(
                train_df, 
                list(missing_customers),
                top_k=top_k
            )
            recommendations_df = pd.concat([recommendations_df, fallback_recs], ignore_index=True)
        
        self.logger.info(f"Generated {len(recommendations_df)} recommendations for {recommendations_df['customer_id'].nunique()} customers")
        return recommendations_df[['customer_id', 'product_id', 'rank']]
    
    def weighted_hybrid_model(self,
                             train_df: pd.DataFrame,
                             customers: Union[List[int], np.ndarray],
                             models: Dict[str, float] = None,
                             top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations using a weighted combination of multiple models.
        
        Args:
            train_df: Training DataFrame with purchase history
            customers: List of customer IDs to generate recommendations for
            models: Dictionary mapping model names to weights
                    (e.g., {'frequency': 0.4, 'recency': 0.3, 'popularity': 0.3})
            top_k: Number of recommendations per customer
            
        Returns:
            DataFrame with customer_id, product_id, and rank columns
        """
        # Default model weights if none provided
        if models is None:
            models = {
                'frequency': 0.4,
                'recency': 0.4,
                'popularity': 0.2
            }
        
        self.logger.info(f"Generating weighted hybrid recommendations with models: {models}")
        
        # Generate recommendations from each model
        all_recommendations = {}
        
        for model_name, weight in models.items():
            if model_name == 'frequency':
                model_recs = self.frequency_based_recommendations(
                    train_df, customers, top_k=top_k*2)  # Get more candidates
            elif model_name == 'recency':
                model_recs = self.recency_based_recommendations(
                    train_df, customers, top_k=top_k*2)
            elif model_name == 'popularity':
                model_recs = self.popularity_based_recommendations(
                    train_df, customers, top_k=top_k*2)
            elif model_name == 'hybrid':
                model_recs = self.hybrid_recommendations(
                    train_df, customers, top_k=top_k*2)
            elif model_name == 'segmented':
                model_recs = self.customer_segmented_recommendations(
                    train_df, customers, top_k=top_k*2)
            else:
                self.logger.warning(f"Unknown model: {model_name}. Skipping.")
                continue
            
            # Store recommendations
            all_recommendations[model_name] = model_recs
        
        # Combine recommendations with weighted scores
        combined_scores = defaultdict(float)
        
        for model_name, recs in all_recommendations.items():
            weight = models[model_name]
            
            # Calculate score for each recommendation
            for _, row in recs.iterrows():
                customer_id = row['customer_id']
                product_id = row['product_id']
                rank = row['rank']
                
                # Convert rank to score (higher score for lower rank)
                score = 1.0 / max(rank, 1)
                
                # Add weighted score
                key = (customer_id, product_id)
                combined_scores[key] += weight * score
        
        # Convert combined scores to DataFrame
        combined_df = []
        for (customer_id, product_id), score in combined_scores.items():
            combined_df.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'score': score
            })
        
        combined_df = pd.DataFrame(combined_df)
        
        # Rank by score for each customer
        if not combined_df.empty:
            combined_df['rank'] = combined_df.groupby('customer_id')['score'].rank(
                method='dense', ascending=False).astype(int)
            
            # Filter to top-k recommendations
            recommendations = combined_df[combined_df['rank'] <= top_k]
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for {recommendations['customer_id'].nunique()} customers")
            return recommendations[['customer_id', 'product_id', 'rank']]
        else:
            self.logger.warning("No recommendations generated.")
            return pd.DataFrame(columns=['customer_id', 'product_id', 'rank'])
    
    def save_recommendations(self,
                             recommendations: pd.DataFrame,
                             output_path: Union[str, Path],
                             suffix: str = "") -> Path:
        """
        Save recommendations to CSV file.
        
        Args:
            recommendations: DataFrame with recommendations
            output_path: Path to save directory
            suffix: Optional string to append to filename
            
        Returns:
            Path to saved file
        """
        # Convert output_path to Path if it's a string
        output_path = Path(output_path) if isinstance(output_path, str) else output_path
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add suffix if provided
        if suffix:
            filename = output_path.stem + "_" + suffix + output_path.suffix
            output_path = output_path.parent / filename
        
        # Save to CSV
        self.logger.info(f"Saving recommendations to {output_path}")
        recommendations.to_csv(output_path, index=False)
        
        return output_path
    
    def format_for_submission(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Format recommendations for Kaggle submission.
        
        Args:
            recommendations: DataFrame with recommendations
            
        Returns:
            DataFrame formatted for submission
        """
        # Ensure required columns exist
        required_cols = ['customer_id', 'product_id', 'rank']
        for col in required_cols:
            if col not in recommendations.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Basic column renaming if needed
        submission = recommendations.copy()
        
        # Ensure each customer has exactly top_k recommendations
        # Group by customer_id and keep only the top-k rows based on rank
        submission = (submission
                     .sort_values(['customer_id', 'rank'])
                     .groupby('customer_id')
                     .head(10)
                     .reset_index(drop=True))
        
        # Check that all customers have 10 recommendations
        recs_per_customer = submission.groupby('customer_id').size()
        if (recs_per_customer != 10).any():
            self.logger.warning("Not all customers have exactly 10 recommendations!")
            
            # Adjust if needed...
            # For now, just log the warning
        
        return submission[['customer_id', 'product_id', 'rank']]