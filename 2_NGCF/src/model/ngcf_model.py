"""
Neural Graph Collaborative Filtering (NGCF) model implementation.

This module contains the implementation of the NGCF model as described in the paper:
"Neural Graph Collaborative Filtering" by Wang et al. (SIGIR 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingPropagationLayer(nn.Module):
    """
    Implementation of a single embedding propagation layer in NGCF.
    
    This layer performs message passing and aggregation between users and items
    in the bipartite graph.
    """
    def __init__(self, embedding_dim, alpha=0.2):
        """
        Initialize the embedding propagation layer.
        
        Args:
            embedding_dim (int): Dimension of embeddings
            alpha (float): LeakyReLU coefficient
        """
        super(EmbeddingPropagationLayer, self).__init__()
        # Trainable weight matrices for embedding transformation and interaction
        self.W1 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.W2 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.alpha = alpha  # LeakyReLU coefficient

    def forward(self, user_embeddings, item_embeddings, adjacency_matrix):
        """
        Perform one layer of embedding propagation between users and items.
        
        Args:
            user_embeddings (torch.Tensor): User embeddings
            item_embeddings (torch.Tensor): Item embeddings
            adjacency_matrix (torch.Tensor): Adjacency matrix representing the bipartite graph
            
        Returns:
            tuple: Updated user and item embeddings
        """
        # Normalize adjacency matrix (graph Laplacian scaling factor)
        row_sum = adjacency_matrix.sum(dim=1, keepdim=True)
        col_sum = adjacency_matrix.sum(dim=0, keepdim=True)
        scaling_factor = torch.sqrt(row_sum * col_sum) + 1e-8
        normalized_adj = adjacency_matrix / scaling_factor

        # Compute messages from items to users
        message_item_to_user = self.compute_message(item_embeddings, user_embeddings, normalized_adj.T)

        # Aggregate messages for users
        updated_user_embeddings = self.aggregate_embeddings(user_embeddings, message_item_to_user)

        # Compute messages from users to items
        message_user_to_item = self.compute_message(user_embeddings, item_embeddings, normalized_adj)

        # Aggregate messages for items
        updated_item_embeddings = self.aggregate_embeddings(item_embeddings, message_user_to_item)

        return updated_user_embeddings, updated_item_embeddings

    def compute_message(self, source_embeddings, target_embeddings, adjacency_matrix):
        """
        Compute messages from source nodes (e.g., items) to target nodes (e.g., users).
        
        Args:
            source_embeddings (torch.Tensor): Embeddings of source nodes
            target_embeddings (torch.Tensor): Embeddings of target nodes
            adjacency_matrix (torch.Tensor): Adjacency matrix
            
        Returns:
            torch.Tensor: Computed messages
        """
        # Interaction term between source and target embeddings (element-wise multiplication)
        interaction_term = source_embeddings.unsqueeze(1) * target_embeddings.unsqueeze(0)

        # Message encoding (Equation 3 in the paper)
        message = torch.matmul(source_embeddings, self.W1) + torch.matmul(interaction_term, self.W2)
        
        # Aggregate the messages with the adjacency matrix
        message = torch.bmm(adjacency_matrix.unsqueeze(2), message)  # Adjust shape for broadcasting

        return message

    def aggregate_embeddings(self, embeddings, message):
        """
        Aggregate messages and update embeddings with LeakyReLU (Equation 4 in the paper).
        
        Args:
            embeddings (torch.Tensor): Original embeddings
            message (torch.Tensor): Computed messages
            
        Returns:
            torch.Tensor: Updated embeddings
        """
        aggregated_message = message.sum(dim=1) + embeddings  # Self-connection included
        updated_embeddings = F.leaky_relu(aggregated_message, negative_slope=self.alpha)

        return updated_embeddings


class GraphEmbeddingPropagation(nn.Module):
    """
    Implementation of multi-layer graph embedding propagation for NGCF.
    
    This class stacks multiple EmbeddingPropagationLayer instances to perform
    multi-layer message passing in the user-item bipartite graph.
    """
    def __init__(self, user_embeddings, item_embeddings, adjacency_matrix, 
                 embedding_dim=128, num_layers=3, sample_size=50, alpha=0.2):
        """
        Initialize the graph embedding propagation module.
        
        Args:
            user_embeddings (torch.Tensor): Initial user embeddings
            item_embeddings (torch.Tensor): Initial item embeddings
            adjacency_matrix (torch.sparse.Tensor): Sparse adjacency matrix
            embedding_dim (int): Dimension of embeddings
            num_layers (int): Number of propagation layers
            sample_size (int): Number of neighbors to sample
            alpha (float): LeakyReLU coefficient
        """
        super(GraphEmbeddingPropagation, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.sample_size = sample_size  # Number of neighbors to sample
        self.alpha = alpha  # LeakyReLU coefficient

        # Initialize user and item embeddings as trainable parameters
        self.user_embeddings = nn.Parameter(torch.tensor(user_embeddings, dtype=torch.float32))
        self.item_embeddings = nn.Parameter(torch.tensor(item_embeddings, dtype=torch.float32))

        # Store adjacency matrix as sparse for memory efficiency
        self.adjacency_matrix = adjacency_matrix.coalesce()  # Convert to sparse format

        # Stack embedding propagation layers
        self.layers = nn.ModuleList([
            EmbeddingPropagationLayer(embedding_dim, alpha) for _ in range(num_layers)
        ])

    def sample_neighbors(self, node_idx, num_neighbors):
        """
        Sample a fixed number of neighbors for a given node index.
        
        Args:
            node_idx (int): Index of the node
            num_neighbors (int): Number of neighbors to sample
            
        Returns:
            list: Sampled neighbor indices
        """
        neighbors = self.adjacency_matrix.indices()[1][self.adjacency_matrix.indices()[0] == node_idx]
        if len(neighbors) > num_neighbors:
            import random
            neighbors = random.sample(neighbors.tolist(), num_neighbors)
        return neighbors

    def create_sampled_adjacency(self, batch_nodes):
        """
        Create a sampled adjacency matrix for the batch nodes.
        
        Args:
            batch_nodes (list): List of node indices in the batch
            
        Returns:
            torch.sparse.Tensor: Sampled adjacency matrix
        """
        indices = []
        values = []

        for node in batch_nodes:
            neighbors = self.sample_neighbors(node, self.sample_size)
            for neighbor in neighbors:
                indices.append([node, neighbor])
                values.append(self.adjacency_matrix[node, neighbor].item())

        # Convert to sparse adjacency matrix
        indices = torch.tensor(indices, dtype=torch.long).t()  # Transpose for PyTorch format
        values = torch.tensor(values, dtype=torch.float32)
        sampled_adj = torch.sparse_coo_tensor(indices, values, self.adjacency_matrix.size())
        return sampled_adj.coalesce()

    def forward(self, batch_nodes):
        """
        Perform multi-layer embedding propagation with neighbor sampling and message passing.
        
        Args:
            batch_nodes (list): List of node indices in the batch
            
        Returns:
            tuple: Updated user and item embeddings
        """
        # Initialize embeddings
        user_embeddings, item_embeddings = self.user_embeddings, self.item_embeddings

        # Create sampled adjacency matrix
        sampled_adj = self.create_sampled_adjacency(batch_nodes)

        # Sequentially apply each embedding propagation layer
        for layer in self.layers:
            user_embeddings, item_embeddings = layer(user_embeddings, item_embeddings, sampled_adj)

        return user_embeddings, item_embeddings


class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering (NGCF) model.
    
    This is the main NGCF model that leverages graph neural networks for
    collaborative filtering by exploiting the user-item interaction graph.
    """
    def __init__(self, num_users, num_items, embedding_size=64, num_layers=3, 
                 dropout_ratio=0.1, p1=0.1, p2=0.0):
        """
        Initialize the NGCF model.
        
        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_size (int): Size of embeddings
            num_layers (int): Number of graph convolutional layers
            dropout_ratio (float): Dropout rate
            p1 (float): Dropout rate for layer embeddings
            p2 (float): Dropout rate for final embeddings
        """
        super(NGCF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # Define the layer transformations
        self.weights = nn.ModuleList([
            nn.Linear(embedding_size, embedding_size) for _ in range(num_layers)
        ])
        
        # Initialize with Xavier initialization
        self.apply(self.xavier_init)
        
        # Dropout rates
        self.p1 = p1
        self.p2 = p2
        
        # Final linear layer to map concatenated embeddings back to the embedding size
        self.final_user_linear = nn.Linear(embedding_size * (num_layers + 1), embedding_size)
        self.final_item_linear = nn.Linear(embedding_size * (num_layers + 1), embedding_size)

    def xavier_init(self, m):
        """
        Xavier initialization for the model weights.
        
        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, user_indices, item_indices, negative_item_indices):
        """
        Forward pass through the NGCF model.
        
        Args:
            user_indices (torch.Tensor): User indices
            item_indices (torch.Tensor): Item indices
            negative_item_indices (torch.Tensor): Negative item indices for BPR loss
            
        Returns:
            tuple: Prediction scores for positive and negative items
        """
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        negative_item_emb = self.item_embedding(negative_item_indices)

        all_user_embeddings = [user_emb]
        all_item_embeddings = [item_emb]
        
        # Propagate through layers
        for layer in range(self.num_layers):
            user_emb = F.dropout(user_emb, p=self.p1, training=self.training)
            item_emb = F.dropout(item_emb, p=self.p1, training=self.training)
            
            user_emb = F.relu(self.weights[layer](user_emb))
            item_emb = F.relu(self.weights[layer](item_emb))
            
            all_user_embeddings.append(user_emb)
            all_item_embeddings.append(item_emb)
        
        # Concatenate embeddings
        user_final_emb = torch.cat(all_user_embeddings, dim=-1)  # Shape: (batch_size, embedding_size * (num_layers + 1))
        item_final_emb = torch.cat(all_item_embeddings, dim=-1)  # Shape: (batch_size, embedding_size * (num_layers + 1))

        # Final projection to embedding size
        user_final_emb = self.final_user_linear(user_final_emb)
        item_final_emb = self.final_item_linear(item_final_emb)

        # Apply dropout to final embeddings
        if self.p2 > 0:
            user_final_emb = F.dropout(user_final_emb, p=self.p2, training=self.training)
            item_final_emb = F.dropout(item_final_emb, p=self.p2, training=self.training)

        # Compute predictions: positive items and negative items
        prediction_pos = torch.sum(user_final_emb * item_final_emb, dim=-1)
        prediction_neg = torch.sum(user_final_emb * negative_item_emb, dim=-1)

        return prediction_pos, prediction_neg


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