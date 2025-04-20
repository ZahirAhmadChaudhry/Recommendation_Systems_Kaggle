"""
Training and evaluation utilities for the NGCF model.

This module provides functions for training and evaluating the NGCF model,
as well as for calculating metrics like hit rate.
"""

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from tqdm import tqdm


def train(model, train_loader, optimizer, epoch, device):
    """
    Train the NGCF model for one epoch.
    
    Args:
        model (torch.nn.Module): NGCF model
        train_loader (torch.utils.data.DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): Current epoch number
        device (torch.device): Device to train on
        
    Returns:
        float: Average loss for this epoch
    """
    model.train()
    total_loss = 0.0
    scaler = GradScaler()  # For mixed precision training
    
    # Use tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for user_indices, item_indices, negative_item_indices in progress_bar:
        # Move data to the device in advance if it's not already there
        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        negative_item_indices = negative_item_indices.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        with autocast():  # Enable mixed precision
            # Forward pass
            positive_preds, negative_preds = model(user_indices, item_indices, negative_item_indices)
            
            # Compute BPR loss
            loss = bpr_loss(positive_preds, negative_preds, lambda_reg=1e-5)
        
        # Backpropagate and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update the progress bar
        current_loss = loss.item()
        total_loss += current_loss
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, k=10):
    """
    Evaluate the NGCF model.
    
    Args:
        model (torch.nn.Module): NGCF model
        test_loader (torch.utils.data.DataLoader): Test data loader
        device (torch.device): Device to evaluate on
        k (int): Number of top items to consider for hit rate
        
    Returns:
        float: Hit rate at k
    """
    model.eval()
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Unpack the batch depending on its structure
            if len(batch) == 3:  # If it's from a TensorDataset with negative samples
                user_indices, item_indices, _ = batch
            elif len(batch) == 2:  # If it's from a TensorDataset without negative samples
                user_indices, item_indices = batch
            else:
                raise ValueError(f"Unexpected batch structure: {len(batch)} elements")
            
            # Move data to device
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            
            # Get all item embeddings for ranking
            all_users = model.user_embedding.weight
            all_items = model.item_embedding.weight
            
            # Get embeddings for the current batch of users
            user_emb = model.user_embedding(user_indices)
            
            # Compute scores for all items for each user
            batch_size = user_emb.size(0)
            scores = torch.matmul(user_emb, all_items.t())  # [batch_size, num_items]
            
            # For each user, store the true item and the scores for all items
            for i in range(batch_size):
                all_predictions.append(scores[i])
                all_ground_truth.append(item_indices[i])
    
    # Calculate hit rate
    hit_rate = hit_at_k(all_predictions, all_ground_truth, k)
    
    return hit_rate


def hit_at_k(predictions, ground_truth, k=10):
    """
    Calculate Hit@K metric for recommendations.
    
    Args:
        predictions (list): List of tensors with predicted scores for all items
        ground_truth (list): List of tensors with true item indices
        k (int): Number of top items to consider
        
    Returns:
        float: Hit rate at k
    """
    hits = 0
    for i in range(len(predictions)):
        # Get the top-k item indices based on predictions
        _, top_indices = torch.topk(predictions[i], k=min(k, len(predictions[i])))
        
        # Check if the ground truth item is in the top-k
        if ground_truth[i] in top_indices:
            hits += 1
    
    return hits / len(predictions)


def bpr_loss(y_pred_pos, y_pred_neg, lambda_reg=1e-5):
    """
    Bayesian Personalized Ranking (BPR) loss function.
    
    Args:
        y_pred_pos (torch.Tensor): Predicted scores for positive items
        y_pred_neg (torch.Tensor): Predicted scores for negative items
        lambda_reg (float): L2 regularization strength
        
    Returns:
        torch.Tensor: BPR loss value
    """
    loss = -torch.log(torch.sigmoid(y_pred_pos - y_pred_neg)).mean()
    reg_loss = lambda_reg * (y_pred_pos.norm(2) + y_pred_neg.norm(2))
    return loss + reg_loss


def hitrate_at_k(true_data, predicted_data, k=10):
    """
    Calculate hitrate at k for the Kaggle competition evaluation metric.
    
    Hitrate@K is defined as the proportion of recommended products 
    that are actually purchased by the customer.
    
    Args:
        true_data (pd.DataFrame): DataFrame with columns ['customer_id', 'product_id']
        predicted_data (pd.DataFrame): DataFrame with columns ['customer_id', 'product_id', 'rank']
        k (int): Number of recommendations to consider
        
    Returns:
        float: Hitrate at k
    """
    # Merge true data with predictions
    data = pd.merge(left=true_data, right=predicted_data, how="left", on=["customer_id", "product_id"])
    
    # Filter recommendations within the specified rank
    df = data[data["rank"] <= k]
    
    # Count non-null ranks (successful hits) per customer
    non_null_counts = df.groupby('customer_id')['rank'].apply(lambda x: x.notna().sum()).reset_index(name='non_null_count')
    
    # Count distinct products per customer in the true data
    distinct_products_per_customer = data.groupby('customer_id')['product_id'].nunique().reset_index(name='distinct_product_count')
    
    # Merge the counts
    df = pd.merge(left=distinct_products_per_customer, right=non_null_counts, how="left", on="customer_id")
    
    # Calculate denominator as min(distinct_product_count, k)
    df["denominator"] = df.apply(lambda row: min(row.distinct_product_count, k), axis=1)
    
    # Fill NaN values with 0 (customers with no hits)
    df = df.fillna(0)
    
    # Calculate hitrate
    return (df["non_null_count"] / df["denominator"]).mean()


def generate_recommendations(model, user_indices, num_items, top_k=10, device='cuda'):
    """
    Generate top-k recommendations for each user.
    
    Args:
        model (torch.nn.Module): Trained NGCF model
        user_indices (torch.Tensor): User indices to generate recommendations for
        num_items (int): Total number of items
        top_k (int): Number of recommendations to generate
        device (str): Device to use
        
    Returns:
        list: List of tuples (user_idx, [recommended_item_indices])
    """
    model.eval()
    recommendations = []
    
    with torch.no_grad():
        # Process users in batches to avoid memory issues
        batch_size = 128
        for start_idx in range(0, len(user_indices), batch_size):
            end_idx = min(start_idx + batch_size, len(user_indices))
            batch_users = user_indices[start_idx:end_idx].to(device)
            
            # Get embeddings for the batch of users
            user_emb = model.user_embedding(batch_users)
            
            # Get all item embeddings
            all_items = model.item_embedding.weight
            
            # Compute scores for all items for each user
            scores = torch.matmul(user_emb, all_items.t())  # [batch_size, num_items]
            
            # Get top-k items for each user
            _, top_indices = torch.topk(scores, k=top_k)
            
            # Store recommendations
            for i, user_idx in enumerate(batch_users.cpu().numpy()):
                recommendations.append((user_idx, top_indices[i].cpu().numpy()))
    
    return recommendations


def create_submission_file(recommendations, user_mapping, item_mapping, output_file):
    """
    Create a submission file for the Kaggle competition.
    
    Args:
        recommendations (list): List of tuples (user_idx, [recommended_item_indices])
        user_mapping (dict): Mapping from user indices to original IDs
        item_mapping (dict): Mapping from item indices to original IDs
        output_file (str): Path to save the submission file
        
    Returns:
        pd.DataFrame: Submission DataFrame
    """
    # Create inverse mappings
    inv_user_mapping = {v: k for k, v in user_mapping.items()}
    inv_item_mapping = {v: k for k, v in item_mapping.items()}
    
    # Initialize lists for the DataFrame
    customer_ids = []
    product_ids = []
    ranks = []
    
    # Process each recommendation
    for user_idx, item_indices in recommendations:
        user_id = inv_user_mapping.get(user_idx, f"User_{user_idx}")
        
        # Add each recommended item with its rank
        for rank, item_idx in enumerate(item_indices, 1):
            customer_ids.append(user_id)
            product_ids.append(inv_item_mapping.get(item_idx, f"Product_{item_idx}"))
            ranks.append(rank)
    
    # Create DataFrame
    submission_df = pd.DataFrame({
        "customer_id": customer_ids,
        "product_id": product_ids,
        "rank": ranks
    })
    
    # Save to file
    submission_df.to_csv(output_file, index=False)
    
    return submission_df