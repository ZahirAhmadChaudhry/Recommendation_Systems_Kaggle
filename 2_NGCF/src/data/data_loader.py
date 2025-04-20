"""
Data processing utilities for NGCF.

This module provides functions for loading, preprocessing, and preparing data
for training and evaluation with the NGCF model.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pickle


def load_data(train_path, products_path, test_path):
    """
    Load and preprocess the training, product, and test data.
    
    Args:
        train_path (str): Path to the training data
        products_path (str): Path to the product data
        test_path (str): Path to the test data
        
    Returns:
        tuple: Preprocessed training data, product data, and test data
    """
    # Load training data (potentially from multiple files)
    if os.path.isdir(train_path):
        # If train_path is a directory, load all parts and concatenate
        train_dataframes = []
        for i in tqdm(range(1, 11)):  # Assuming 10 parts
            file_path = os.path.join(train_path, f"train_data_part_{i}.csv")
            if os.path.exists(file_path):
                train_dataframes.append(pd.read_csv(file_path))
        
        train_data = pd.concat(train_dataframes, ignore_index=True)
        del train_dataframes  # Free memory
    else:
        # Otherwise, load a single file
        train_data = pd.read_csv(train_path)
    
    # Load product data
    products_data = pd.read_csv(products_path)
    
    # Load test data
    test_data = pd.read_csv(test_path)
    
    return train_data, products_data, test_data


def transform_data(train_data, products_data):
    """
    Transform raw data into a format suitable for NGCF.
    
    Args:
        train_data (pd.DataFrame): Raw training data
        products_data (pd.DataFrame): Raw product data
        
    Returns:
        pd.DataFrame: Transformed and processed purchase data
    """
    # Aggregate customer purchase data
    customer_data = train_data.groupby(['customer_id', 'product_id']).agg({
        'quantity': 'sum'  # Total quantity purchased per product per customer
    }).reset_index()
    
    customer_data['quantity'] = customer_data['quantity'].astype(int)
    customer_data = customer_data.sort_values(by=['customer_id', 'quantity'], ascending=[True, False])
    
    # Select relevant product features
    features_to_keep = [
        'product_id', 'brand_key', 'shelf_level1', 'shelf_level2', 'shelf_level3',
        'bio', 'sugar_free', 'gluten_free', 'halal', 'reduced_sugar', 'vegetarian', 'vegan',
        'pesticide_free', 'no_added_sugar', 'salt_reduced', 'no_added_salt', 'no_artificial_flavours', 
        'porc', 'frozen', 'fat_free', 'reduced_fats', 'fresh', 'alcool', 'lactose_free'
    ]
    
    # Select only the required columns from the products table for efficiency
    products_reduced = products_data[features_to_keep]
    
    # Merge customer_data with the filtered products table on 'product_id'
    purchase_data = customer_data.merge(products_reduced, on='product_id', how='inner')
    
    # Convert customer_id and product_id to integer IDs (removing non-numeric characters)
    purchase_data['customer_id'] = purchase_data['customer_id'].str.replace('Household_', '').astype(int)
    purchase_data['product_id'] = purchase_data['product_id'].str.replace('Product_', '').astype(int)
    
    purchase_data = purchase_data.sort_values(by=['customer_id', 'quantity'], ascending=[True, False])
    
    # Encode categorical features
    purchase_data['brand_key_encoded'] = frequency_encode(purchase_data, 'brand_key')
    purchase_data['shelf_level1_encoded'] = frequency_encode(purchase_data, 'shelf_level1')
    purchase_data['shelf_level2_encoded'] = frequency_encode(purchase_data, 'shelf_level2')
    purchase_data['shelf_level3_encoded'] = frequency_encode(purchase_data, 'shelf_level3')
    
    # Drop the original categorical columns
    purchase_data.drop(['brand_key', 'shelf_level1', 'shelf_level2', 'shelf_level3'], axis=1, inplace=True)
    
    # Normalize numerical features
    numerical_columns = ['quantity', 'brand_key_encoded', 'shelf_level1_encoded', 
                         'shelf_level2_encoded', 'shelf_level3_encoded']
    
    scaler = MinMaxScaler()
    purchase_data[numerical_columns] = scaler.fit_transform(purchase_data[numerical_columns])
    
    return purchase_data


def frequency_encode(df, column):
    """
    Encode a categorical column using frequency encoding.
    
    Args:
        df (pd.DataFrame): DataFrame containing the column
        column (str): Name of the column to encode
        
    Returns:
        pd.Series: Encoded column values
    """
    freq = df[column].value_counts()  # Calculate frequencies of each category
    return df[column].map(freq)  # Map categories to their corresponding frequencies


def preprocess_test_data(test_data):
    """
    Preprocess test data for evaluation.
    
    Args:
        test_data (pd.DataFrame): Raw test data
        
    Returns:
        pd.DataFrame: Preprocessed test data
    """
    test_set = test_data.drop(columns=["transaction_id"])
    
    if not pd.api.types.is_numeric_dtype(test_set['customer_id']):
        test_set['customer_id'] = test_set['customer_id'].str.replace('Household_', '').astype(int)
    
    if not pd.api.types.is_numeric_dtype(test_set['product_id']):
        test_set['product_id'] = test_set['product_id'].str.replace('Product_', '').astype(int)
    
    test_set = test_set.sort_values(by=['customer_id', 'product_id'], ascending=[True, True])
    
    return test_set


def create_graph(purchase_data):
    """
    Create a bipartite graph from purchase data.
    
    Args:
        purchase_data (pd.DataFrame): Processed purchase data
        
    Returns:
        nx.Graph: NetworkX graph representing the bipartite interactions
    """
    B = nx.Graph()
    
    # Add nodes for customers and products with the appropriate types
    for _, row in purchase_data.iterrows():
        # Ensure the IDs are integers before creating the nodes
        customer_id = int(row['customer_id'])
        product_id = int(row['product_id'])
        
        customer_node = f"Customer_{customer_id}"
        product_node = f"Product_{product_id}"
        
        # Add customer node (if not already in graph)
        if customer_node not in B:
            B.add_node(customer_node, type='customer')
        
        # Add product node (if not already in graph)
        if product_node not in B:
            B.add_node(product_node, type='product')
        
        # Add edge between customer and product (purchase relationship)
        B.add_edge(customer_node, product_node, weight=row['quantity'])
    
    return B


def save_graph(graph, output_dir):
    """
    Save the graph to disk in chunks.
    
    Args:
        graph (nx.Graph): NetworkX graph to save
        output_dir (str): Directory to save the graph to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save nodes with attributes to a file
    with open(os.path.join(output_dir, "nodes.pkl"), "wb") as f:
        pickle.dump(dict(graph.nodes(data=True)), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    chunk_size = 9_000_000
    edges = list(graph.edges(data=True))  # Get all edges with attributes
    
    # Save edges in chunks
    for i in range(0, len(edges), chunk_size):
        chunk = edges[i:i + chunk_size]
        with open(os.path.join(output_dir, f"edges_{i // chunk_size}.pkl"), "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(input_dir):
    """
    Load a graph from disk.
    
    Args:
        input_dir (str): Directory containing the saved graph
        
    Returns:
        nx.Graph: Loaded NetworkX graph
    """
    with open(os.path.join(input_dir, "nodes.pkl"), "rb") as f:
        nodes = pickle.load(f)
    
    graph = nx.Graph()
    graph.add_nodes_from(nodes.items())
    
    i = 0
    while True:
        edge_file = os.path.join(input_dir, f"edges_{i}.pkl")
        if not os.path.exists(edge_file):
            break
        
        with open(edge_file, "rb") as f:
            edges = pickle.load(f)
            graph.add_edges_from(edges)
        i += 1
    
    return graph


def initialize_embeddings(graph, purchase_data, embedding_dim=128):
    """
    Initialize embeddings for users and items based on graph structure and features.
    
    Args:
        graph (nx.Graph): NetworkX graph
        purchase_data (pd.DataFrame): Processed purchase data
        embedding_dim (int): Dimension of embeddings
        
    Returns:
        tuple: Customer embeddings, product embeddings, customer node indices, product node indices, 
               customer to int mapping, product to int mapping
    """
    # Extract customer and product nodes
    customer_nodes = [n for n in graph.nodes if graph.nodes[n].get('type') == 'customer']
    product_nodes = [n for n in graph.nodes if graph.nodes[n].get('type') == 'product']
    
    # Extract product features for initialization
    product_features = purchase_data.drop_duplicates(subset="product_id").set_index("product_id")
    product_feature_columns = product_features.columns.drop(["customer_id", "quantity"])
    product_features = product_features[product_feature_columns].to_dict(orient="index")
    
    # Create mappings from nodes to indices
    customer_to_int = {node: idx for idx, node in enumerate(customer_nodes)}
    product_to_int = {node: idx for idx, node in enumerate(product_nodes)}
    
    # Function to handle dimensionality mismatch for product embeddings
    def get_product_embedding(product_id):
        features = np.array(list(product_features[product_id].values()))
        if len(features) < embedding_dim:
            padding = np.random.rand(embedding_dim - len(features))
            return np.concatenate([features, padding])
        elif len(features) > embedding_dim:
            return features[:embedding_dim]
        return features
    
    # Initialize customer embeddings as random tensors and normalize them
    customer_embeddings = torch.randn(len(customer_nodes), embedding_dim)
    customer_embeddings = torch.nn.functional.normalize(customer_embeddings, p=2, dim=1)
    
    # Initialize product embeddings using product features
    product_embeddings = torch.zeros(len(product_nodes), embedding_dim)
    
    for idx, product_node in enumerate(product_nodes):
        product_id = int(product_node.split('_')[1])
        product_embedding = get_product_embedding(product_id)
        product_embeddings[idx] = torch.tensor(product_embedding, dtype=torch.float32)
    
    product_embeddings = torch.nn.functional.normalize(product_embeddings, p=2, dim=1)
    
    # Create node indices tensors
    customer_node_indices = torch.tensor([i for i, n in enumerate(graph.nodes) if graph.nodes[n].get('type') == 'customer'])
    product_node_indices = torch.tensor([i for i, n in enumerate(graph.nodes) if graph.nodes[n].get('type') == 'product'])
    
    return (customer_embeddings, product_embeddings, customer_node_indices, product_node_indices,
            customer_to_int, product_to_int)


def hard_negative_sampling(user_idx, positive_items, all_items, product_embeddings, 
                          num_negatives=5, subset_size=1000):
    """
    Perform hard negative sampling by finding items similar to those the user has interacted with.
    
    Args:
        user_idx (int): User index
        positive_items (list): Items the user has interacted with
        all_items (list): All available items
        product_embeddings (torch.Tensor): Item embeddings
        num_negatives (int): Number of negative samples to generate
        subset_size (int): Size of the subset to sample from
        
    Returns:
        list: Hard negative samples
    """
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    product_embeddings = product_embeddings.to(device)
    
    # Efficiently sample a subset of all_items
    if isinstance(all_items, range):
        subset_indices = torch.randint(0, len(all_items), (min(subset_size, len(all_items)),), device=device)
    else:
        subset_indices = torch.tensor(
            random.sample(all_items, min(subset_size, len(all_items))),
            device=device,
            dtype=torch.long
        )
    
    # Exclude positive items from the subset
    positive_items_tensor = torch.tensor(positive_items, device=device, dtype=torch.long)
    mask = ~torch.isin(subset_indices, positive_items_tensor)
    filtered_subset = subset_indices[mask]
    
    # Ensure at least `num_negatives` items are available
    if len(filtered_subset) < num_negatives:
        print(f"Warning: Not enough filtered negatives, reducing to {len(filtered_subset)}")
        num_negatives = len(filtered_subset)
    
    # Get embeddings for positive items and the filtered subset
    positive_embeddings = product_embeddings[positive_items_tensor]
    subset_embeddings = product_embeddings[filtered_subset]
    
    # Normalize embeddings for cosine similarity calculation
    positive_embeddings = torch.nn.functional.normalize(positive_embeddings, dim=1)
    subset_embeddings = torch.nn.functional.normalize(subset_embeddings, dim=1)
    
    # Compute cosine similarities between subset and positive items
    similarities = torch.mm(subset_embeddings, positive_embeddings.T)  # (|subset|, |positive_items|)
    
    # Compute mean similarity for each sampled item in the subset
    mean_similarities = similarities.mean(dim=1)  # (|subset|,)
    
    # Select the top-k hardest negatives (lowest mean similarity)
    hard_negatives_indices = torch.topk(mean_similarities, num_negatives, largest=False).indices
    
    # Map back to the original item indices
    hard_negatives = filtered_subset[hard_negatives_indices]
    
    return hard_negatives.tolist()


def parallel_negative_sampling(user_indices, item_indices, all_item_indices, 
                             max_neg_samples=5, device='cuda'):
    """
    Generate negative samples efficiently using GPU-based tensor operations.
    
    Args:
        user_indices (torch.Tensor): User indices
        item_indices (torch.Tensor): Item indices
        all_item_indices (torch.Tensor): All item indices
        max_neg_samples (int): Maximum number of negative samples per user-item pair
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: Negative item indices
    """
    # Ensure the data is on the correct device (GPU or CPU)
    user_indices = user_indices.to(device)
    item_indices = item_indices.to(device)
    all_item_indices = all_item_indices.to(device)
    
    def generate_negatives(user_idx, item_idx):
        # Sample random negatives (ensuring no overlap with the positive item)
        neg_samples = all_item_indices[torch.randint(len(all_item_indices), (max_neg_samples,)).to(device)]
        neg_samples = neg_samples[neg_samples != item_idx]  # Exclude the positive item
        
        # If less than required, keep sampling until we fill max_neg_samples
        while len(neg_samples) < max_neg_samples:
            additional_neg_samples = all_item_indices[torch.randint(len(all_item_indices), (max_neg_samples - len(neg_samples),)).to(device)]
            neg_samples = torch.cat((neg_samples, additional_neg_samples), dim=0)
            neg_samples = neg_samples[neg_samples != item_idx]
        
        return neg_samples[:max_neg_samples]
    
    # Parallelizing the negative sampling for each user-item pair using multiprocessing
    def negative_sampling_worker(start_idx, end_idx):
        negatives_batch = []
        for idx in range(start_idx, end_idx):
            user_idx = user_indices[idx]
            item_idx = item_indices[idx]
            neg_samples = generate_negatives(user_idx, item_idx)
            negatives_batch.append(neg_samples)
        return negatives_batch
    
    num_workers = mp.cpu_count()  # Use all available CPU cores
    chunk_size = len(user_indices) // num_workers
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
    
    # Use multiprocessing to distribute the negative sampling across the workers
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(negative_sampling_worker, chunks)
    
    # Flatten the results
    all_negatives = torch.cat([torch.cat(res, dim=0) for res in results], dim=0)
    
    return all_negatives


def prepare_data(graph, batch_size=10000, model=None, item_embeddings=None):
    """
    Prepare training data with positive and negative samples.
    
    Args:
        graph (nx.Graph): NetworkX graph
        batch_size (int): Batch size
        model (torch.nn.Module): NGCF model (if available)
        item_embeddings (torch.Tensor): Item embeddings
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for training
    """
    if item_embeddings is None:
        raise ValueError("Item embeddings are None. Ensure that the embeddings are properly initialized.")
    
    # Map user and item IDs to integers
    user_to_int = {user: idx for idx, user in enumerate(n for n in graph.nodes if graph.nodes[n].get('type') == 'customer')}
    item_to_int = {item: idx for idx, item in enumerate(n for n in graph.nodes if graph.nodes[n].get('type') == 'product')}
    
    # Extract users and items
    users = list(user_to_int.keys())
    all_items = torch.tensor(list(item_to_int.values()), dtype=torch.long)  # All item IDs as tensor
    
    # Cache neighbors for all users
    user_neighbors = {user: list(graph.neighbors(user)) for user in users}
    
    # Initialize lists to collect results
    positive_users, positive_items = [], []
    negative_users, negative_items = [], []
    
    # Process users in batches
    for i in range(0, len(users), batch_size):
        batch_users = users[i:i + batch_size]
        
        # Create positive interactions (observed user-item pairs)
        batch_positive_users, batch_positive_items = [], []
        for user in batch_users:
            neighbors = user_neighbors[user]
            for item in neighbors:
                if graph[user][item].get('weight', 0) > 0:  # Valid interaction
                    batch_positive_users.append(user_to_int[user])
                    batch_positive_items.append(item_to_int[item])
        
        positive_users.extend(batch_positive_users)
        positive_items.extend(batch_positive_items)
        
        # Efficient negative sampling for the batch
        for user in batch_users:
            # Get positive items for the user (those they've interacted with)
            positive_items_for_user = [item_to_int[item] for item in user_neighbors[user] if graph[user][item].get('weight', 0) > 0]
            
            # Get unobserved items efficiently by creating a mask
            positive_items_set = set(positive_items_for_user)
            unobserved_items = all_items[~torch.isin(all_items, torch.tensor(positive_items_for_user, dtype=torch.long))]
            
            # Adjust the indices for negative sampling to match embeddings
            unobserved_items_adjusted = unobserved_items + 99999  # Add 99,999 to product indices
            
            # Sample hard negatives for the entire batch of users at once
            sampled_negatives = hard_negative_sampling(user_to_int[user], positive_items_for_user, unobserved_items_adjusted.tolist(), item_embeddings)
            
            # Collect the negative samples
            for neg_item in sampled_negatives:
                negative_users.append(user_to_int[user])
                negative_items.append(neg_item)
    
    # Ensure there are sufficient positive and negative samples
    if not positive_users or not negative_users:
        raise ValueError("Insufficient positive or negative samples generated.")
    
    # Convert to tensors for use in training
    positive_users_tensor = torch.tensor(positive_users, dtype=torch.long)
    positive_items_tensor = torch.tensor(positive_items, dtype=torch.long)
    negative_users_tensor = torch.tensor(negative_users, dtype=torch.long)
    negative_items_tensor = torch.tensor(negative_items, dtype=torch.long)
    
    # Return the dataset
    return data_loader(positive_users_tensor, positive_items_tensor, negative_items_tensor, batch_size)


def prepare_test_data(graph, batch_size=10000):
    """
    Prepare test data for evaluation.
    
    Args:
        graph (nx.Graph): NetworkX graph
        batch_size (int): Batch size
        
    Returns:
        tuple: User indices and item indices for testing
    """
    # Map user and item IDs to integers
    user_to_int = {user: idx for idx, user in enumerate(n for n in graph.nodes if graph.nodes[n].get('type') == 'customer')}
    item_to_int = {item: idx for idx, item in enumerate(n for n in graph.nodes if graph.nodes[n].get('type') == 'product')}
    
    # Extract users as integers
    users = list(user_to_int.keys())
    
    # Initialize lists to collect results
    test_users, test_items = [], []
    
    # Precompute neighbors for all users in the graph
    user_neighbors = {user: list(graph.neighbors(user)) for user in users}
    
    # Process users in batches
    for i in range(0, len(users), batch_size):
        batch_users = users[i:i + batch_size]
        
        # Create test samples for the current batch
        for user in batch_users:
            neighbors = user_neighbors[user]
            for item in neighbors:
                # Check if the interaction is valid (positive weight)
                if graph[user][item].get('weight', 0) > 0:
                    test_users.append(user_to_int[user])
                    test_items.append(item_to_int[item])
    
    if not test_users:
        raise ValueError("No test samples found in the dataset.")
    
    # Convert to tensors
    test_user_indices = torch.tensor(test_users, dtype=torch.long)
    test_item_indices = torch.tensor(test_items, dtype=torch.long)
    
    return test_user_indices, test_item_indices


def data_loader(user_indices, pos_item_indices, neg_item_indices, batch_size):
    """
    Create a DataLoader for training.
    
    Args:
        user_indices (torch.Tensor): User indices
        pos_item_indices (torch.Tensor): Positive item indices
        neg_item_indices (torch.Tensor): Negative item indices
        batch_size (int): Batch size
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for training
    """
    dataset = TensorDataset(user_indices, pos_item_indices, neg_item_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)