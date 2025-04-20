"""
Negative Sampling module for LightGBM recommendation model.

This module contains the HybridNegativeSampler class which is responsible for generating
negative samples for training the recommendation model using a hybrid weighting scheme.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import gc
from typing import Tuple, Dict, Set

class HybridNegativeSampler:
    def __init__(
        self, 
        interaction_features: pd.DataFrame,
        popularity_weight: float = 0.5,
        recency_weight: float = 0.3,
        promo_weight: float = 0.2,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """Initialize the HybridNegativeSampler.
        
        Args:
            interaction_features: DataFrame containing interactions and features.
            popularity_weight: Weight for the popularity component.
            recency_weight: Weight for the recency component.
            promo_weight: Weight for the promo component.
            device: 'cuda' if GPU is available, otherwise 'cpu'.
            verbose: If False, suppress all debug prints (but keep tqdm progress bars).
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.interaction_features = interaction_features
        self.weights = {
            'popularity': popularity_weight,
            'recency': recency_weight,
            'promo': promo_weight
        }
        self.verbose = verbose
        
        # Validate weights sum to 1
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"
        
        if self.verbose:
            print(f"Using device: {self.device}")
            print(f"Weight configuration: {self.weights}")

    def _print(self, *args, **kwargs):
        """Utility function to print only if verbose is True."""
        if self.verbose:
            print(*args, **kwargs)

    def compute_sampling_weights(self) -> Tuple[torch.Tensor, np.ndarray]:
        """Compute hybrid weights for negative sampling.
        
        Returns: 
            Tuple containing (weights tensor, product_ids array)
        """
        self._print("Computing sampling weights...")
        
        # Group by product_id for aggregation
        product_stats = self.interaction_features.groupby('product_id').agg({
            'transaction_count': 'sum',    # Popularity
            'days_since_last': 'min',      # Recency
            'promo_ratio': 'mean'          # Promo usage
        })

        # Compute normalized components
        popularity = torch.tensor(product_stats['transaction_count'].values, device=self.device)
        popularity = popularity / (popularity.max() if popularity.max() > 0 else 1)

        recency = torch.tensor(product_stats['days_since_last'].values, device=self.device)
        recency = 1 / (1 + recency)  # Inverse of days since last purchase
        recency = recency / (recency.max() if recency.max() > 0 else 1)

        promo = torch.tensor(product_stats['promo_ratio'].values, device=self.device)
        promo = promo / (promo.max() if promo.max() > 0 else 1)

        # Apply weights
        final_weights = (
            self.weights['popularity'] * popularity + 
            self.weights['recency'] * recency + 
            self.weights['promo'] * promo
        )
        total_sum = final_weights.sum()
        if total_sum > 0:
            final_weights = final_weights / total_sum

        self._print(f"Computed weights for {len(final_weights)} products.")
        return final_weights, product_stats.index.values

    def build_customer_purchase_dict(self) -> Dict[int, Set[int]]:
        """Build dictionary mapping each customer to the products they have purchased."""
        self._print("Building customer purchase dictionary...")
        customer_purchases = {}
        
        # Show progress bar with a description, always enabled
        for cid in tqdm(self.interaction_features['customer_id'].unique(), desc="Building purchase dictionary"):
            customer_purchases[cid] = set(
                self.interaction_features[
                    self.interaction_features['customer_id'] == cid
                ]['product_id'].values
            )
        
        self._print(f"Built purchase dictionary for {len(customer_purchases)} customers.")
        return customer_purchases

    def generate_samples(
        self,
        batch_size: int = 10000,
        neg_ratio: int = 1,
        min_samples_per_user: int = 1
    ) -> pd.DataFrame:
        """Generate negative samples using a hybrid weighting scheme.
        
        Args:
            batch_size: Number of customers to process at once
            neg_ratio: Ratio of negative to positive samples
            min_samples_per_user: Minimum number of negative samples per user
            
        Returns:
            DataFrame containing both positive and negative samples with 'target' column
        """
        weights, product_ids = self.compute_sampling_weights()
        
        # Prepare positive samples (all (customer, product) pairs in the dataset)
        positive_samples = self.interaction_features[['customer_id', 'product_id']].values
        self._print(f"Number of positive samples: {positive_samples.shape[0]}")

        # Get unique customers to ensure we sample negatives only once per customer
        unique_customers = self.interaction_features['customer_id'].unique()
        customer_ids = torch.tensor(unique_customers, device=self.device)
        
        # Build customer -> purchased products map
        customer_purchases = self.build_customer_purchase_dict()
        
        negative_samples = []
        total_customers = len(customer_ids)
        num_batches = (total_customers + batch_size - 1) // batch_size
        
        self._print(f"Generating negative samples with {neg_ratio}:1 ratio")
        self._print(f"Processing {total_customers:,} unique customers in {num_batches} batches")
        
        # Always show progress bar with a description
        for batch_start in tqdm(range(0, total_customers, batch_size), desc="Generating negative samples"):
            batch_customers = customer_ids[batch_start:batch_start + batch_size]
            batch_negatives = []
            
            for cid in batch_customers.cpu().numpy():
                purchased = customer_purchases[cid]
                self._print(f"Customer {cid} purchased {len(purchased)} products")
                
                # Create mask for unpurchased products
                mask = torch.ones_like(weights, dtype=torch.bool)
                indices_to_mask = [i for i, pid in enumerate(product_ids) if pid in purchased]
                mask[np.array(indices_to_mask)] = False
                self._print(f"Mask size (unpurchased products): {mask.sum().item()} out of {len(mask)} total products")
                
                # If the customer has all products, skip
                if not mask.any():
                    self._print(f"Customer {cid} has purchased all products, skipping.")
                    continue
                
                # Get valid weights for unpurchased products
                valid_weights = weights[mask]
                valid_weights_sum = valid_weights.sum()
                if valid_weights_sum > 0:
                    valid_weights = valid_weights / valid_weights_sum
                
                # Number of positive interactions for this customer
                n_positive = len(purchased)
                # Number of negative samples to draw
                n_samples = max(
                    min(n_positive * neg_ratio, (weights.size(0) - len(purchased))),
                    min_samples_per_user
                )
                self._print(f"Customer {cid} will generate {n_samples} negative samples")
                
                if n_samples > 0:
                    negative_indices = torch.multinomial(
                        valid_weights,
                        n_samples,
                        replacement=False
                    )
                    valid_product_ids = product_ids[mask.cpu().numpy()]
                    selected_products = valid_product_ids[negative_indices.cpu().numpy()]
                    
                    batch_negatives.extend([
                        (cid, prod_id, 0) for prod_id in selected_products
                    ])
            
            negative_samples.extend(batch_negatives)
            
            # Memory management for GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            self._print(f"Batch {(batch_start // batch_size) + 1}: Negative samples generated: {len(batch_negatives)}")
        
        # Construct final dataset
        self._print("Creating final dataset...")
        negative_df = pd.DataFrame(negative_samples, columns=['customer_id', 'product_id', 'target'])
        
        # Positive dataset
        positive_df = pd.DataFrame(positive_samples, columns=['customer_id', 'product_id'])
        positive_df['target'] = 1
        
        self._print(f"Total positive samples: {positive_df.shape[0]}")
        self._print(f"Total negative samples: {negative_df.shape[0]}")
        
        training_data = pd.concat([positive_df, negative_df]).reset_index(drop=True)
        
        # Final stats
        total_samples = len(training_data)
        pos_ratio = (training_data['target'] == 1).mean()
        neg_ratio_actual = 1.0 - pos_ratio  # or (training_data['target'] == 0).mean()
        
        self._print("\nFinal dataset statistics:")
        self._print(f"Total samples: {total_samples:,}")
        self._print(f"Positive samples: {pos_ratio * 100:.1f}%")
        self._print(f"Negative samples: {neg_ratio_actual * 100:.1f}%")
        if pos_ratio > 0:  # Avoid division by zero if no positives
            self._print(f"Achieved negative:positive ratio = {neg_ratio_actual / pos_ratio:.2f}:1")
        
        return training_data