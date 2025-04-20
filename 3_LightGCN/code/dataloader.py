"""
Data loading and processing for Carrefour dataset
"""
import os
from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import scipy.sparse as sp
import world
from time import time
import gc

class CarrefourDataset(Dataset):
    def __init__(self):
        # The path suffix should be fixed
        self.path = "../Enhanced_data/"
        self.split_ratio = 0.10  # Changed to 20% sampling
        self._load_and_sample_data()
        self._build_mappings()
        self._create_sparse_interactions()
        self._build_test()
        self.Graph = None
        
    def _load_and_sample_data(self):
        """Load data and sample 10% of customers"""
        print(f"Loading data from {self.path}")
        
        # Load products with correct format
        self.products = pd.read_parquet(join(self.path, "optimized_products.parquet"))
        # Reduced log
        # print("Products loaded successfully")
        
        # Load training data with correct format
        self.train_data = pd.read_parquet(join(self.path, "optimized_train.parquet"))
        # Reduced log
        # print("Training data loaded successfully")
        
        # Load test data with correct format
        self.test_data = pd.read_parquet(join(self.path, "optimized_test.parquet"))
        # Reduced log
        # print("Test data loaded successfully")
        
        # Reduced log
        # print("Sampling 10% of customers...")
        # Sample 10% of unique customers
        unique_customers = self.train_data['customer_id'].unique()
        num_samples = int(len(unique_customers) * self.split_ratio)
        sampled_customers = np.random.choice(unique_customers, size=num_samples, replace=False)
        
        # Filter data for sampled customers
        self.train_data = self.train_data[self.train_data['customer_id'].isin(sampled_customers)]
        self.test_data = self.test_data[self.test_data['customer_id'].isin(sampled_customers)]
        
        print(f"Sampled dataset size: {len(sampled_customers)} users ({self.split_ratio*100}% of original)")
    
    def _build_mappings(self):
        """Create user and item ID mappings"""
        print("Creating user-item mappings...")
        unique_users = sorted(self.train_data['customer_id'].unique())
        unique_items = sorted(self.products['product_id'].unique())
        
        self.user2id = {user: idx for idx, user in enumerate(unique_users)}
        self.item2id = {item: idx for idx, item in enumerate(unique_items)}
        
        self.id2user = {idx: user for user, idx in self.user2id.items()}
        self.id2item = {idx: item for item, idx in self.item2id.items()}
        
        self.n_users = len(self.user2id)
        self.m_items = len(self.item2id)
        
        print("Converting to internal IDs...")
        self.train_data['user_idx'] = self.train_data['customer_id'].map(self.user2id)
        self.train_data['item_idx'] = self.train_data['product_id'].map(self.item2id)
        self.test_data['user_idx'] = self.test_data['customer_id'].map(self.user2id)
        self.test_data['item_idx'] = self.test_data['product_id'].map(self.item2id)
        
        print(f"Max user index: {max(self.user2id.values())}, Max item index: {max(self.item2id.values())}")
    
    def _create_sparse_interactions(self):
        """Create user-item interaction sparse matrix"""
        print("Creating interaction matrix...")
        self.trainDataSize = len(self.train_data)
        
        # Pre-calculating user-item interactions
        self.allPos = self._get_user_pos_items(self.train_data)
        
        # Create sparse matrix in COO format
        rows = self.train_data['user_idx'].values
        cols = self.train_data['item_idx'].values
        data = np.ones_like(rows)
        
        self.UserItemNet = sp.coo_matrix((data, (rows, cols)), 
                                       shape=(self.n_users, self.m_items),
                                       dtype=np.float32)
        
        # Convert to CSR format for efficient operations
        self.UserItemNet = self.UserItemNet.tocsr()
        
        print(f"Carrefour Dataset loaded. n_users: {self.n_users}, n_items: {self.m_items}")
        print(f"Number of interactions: {len(self.train_data)}")
    
    def _build_test(self):
        """Build test dictionary"""
        self.testDict = self._get_test_dict()
    
    def _get_user_pos_items(self, data):
        """Get positive items for each user"""
        pos_items = {}
        for user, items in data.groupby('user_idx')['item_idx']:
            pos_items[user] = items.values.tolist()
        return pos_items
    
    def _get_test_dict(self):
        """Create test dictionary"""
        test_dict = {}
        for user, items in self.test_data.groupby('user_idx')['item_idx']:
            test_dict[user] = items.values.tolist()
        return test_dict
    
    def getSparseGraph(self):
        """Build sparse graph in normalized format"""
        if self.Graph is not None:
            return self.Graph

        try:
            print("Building graph in sparse format...")
            
            # Pre-allocate memory for indices and values
            n_nodes = self.n_users + self.m_items
            n_edges = len(self.train_data) * 2  # bidirectional edges
            
            # Build indices more efficiently
            user_indices = self.train_data['user_idx'].values
            item_indices = self.train_data['item_idx'].values + self.n_users
            
            row_indices = torch.from_numpy(np.concatenate([user_indices, item_indices]))
            col_indices = torch.from_numpy(np.concatenate([item_indices, user_indices]))
            
            indices = torch.stack([row_indices, col_indices])
            values = torch.ones(n_edges, dtype=torch.float32)
            
            # Create sparse tensor with pre-allocated memory
            self.Graph = torch.sparse_coo_tensor(
                indices,
                values,
                torch.Size([n_nodes, n_nodes]),
                device='cpu'  # Build on CPU first
            ).coalesce()
            
            # Calculate degrees efficiently
            degrees = torch.sparse.sum(self.Graph, dim=1).to_dense()
            degrees[degrees == 0] = 1
            deg_inv_sqrt = torch.pow(degrees, -0.5)
            
            # Create diagonal matrix efficiently
            diag_indices = torch.arange(n_nodes)
            diag_indices = torch.stack([diag_indices, diag_indices])
            deg_inv_sqrt_mat = torch.sparse_coo_tensor(
                diag_indices, 
                deg_inv_sqrt,
                torch.Size([n_nodes, n_nodes])
            )
            
            # Normalize with better memory management
            self.Graph = torch.sparse.mm(
                torch.sparse.mm(deg_inv_sqrt_mat, self.Graph),
                deg_inv_sqrt_mat
            ).coalesce()
            
            # Move to GPU after all operations are done
            self.Graph = self.Graph.to(world.device)
            
            print("Graph built successfully!")
            return self.Graph
            
        except Exception as e:
            print(f"Error building graph: {str(e)}")
            raise
    
    def __len__(self):
        return self.n_users
    
    def __getitem__(self, idx):
        return idx
