# LightGCN Architecture Overview

This document provides a detailed explanation of the LightGCN architecture implemented in this codebase.

## Core Concept

LightGCN simplifies traditional Graph Convolutional Networks (GCNs) by removing feature transformation and non-linear activation components, focusing solely on neighborhood aggregation for collaborative filtering in recommendation systems.

## Mathematical Formulation

The key operation in LightGCN is the light graph convolution, defined as:

For users:
$$e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|} \cdot \sqrt{|\mathcal{N}_i|}} e_i^{(k)}$$

For items:
$$e_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i|} \cdot \sqrt{|\mathcal{N}_u|}} e_u^{(k)}$$

Where:
- $e_u^{(k)}$ and $e_i^{(k)}$ are the embeddings of user $u$ and item $i$ at the $k$-th layer
- $\mathcal{N}_u$ is the set of items interacted by user $u$
- $\mathcal{N}_i$ is the set of users who interacted with item $i$
- $\frac{1}{\sqrt{|\mathcal{N}_u|} \cdot \sqrt{|\mathcal{N}_i|}}$ is the symmetric normalization term

## Layer Structure

1. **Initial Embedding Layer**:
   - User embedding matrix $\mathbf{E}^{(0)}_{\text{user}} \in \mathbb{R}^{n \times d}$
   - Item embedding matrix $\mathbf{E}^{(0)}_{\text{item}} \in \mathbb{R}^{m \times d}$
   - Where $n$ is the number of users, $m$ is the number of items, and $d$ is the embedding dimension

2. **Light Graph Convolution Layers**:
   - Each layer performs message passing between users and items
   - No weight matrix or non-linear transformation
   - Only normalized sum aggregation of neighbor embeddings

3. **Layer Combination**:
   - Final embeddings are weighted sum of embeddings from all layers:
   $$e_u = \sum_{k=0}^{K} \alpha_k e_u^{(k)}, \quad e_i = \sum_{k=0}^{K} \alpha_k e_i^{(k)}$$
   - Where $\alpha_k$ is the weight for the $k$-th layer (defaults to $\frac{1}{K+1}$ for all layers)

4. **Prediction Layer**:
   - User-item affinity calculated by inner product: $\hat{y}_{ui} = e_u^T \cdot e_i$

## Implementation Details

In the codebase:

- The model is implemented in `model.py` with the `LightGCN` class inheriting from `BasicModel`
- The graph convolution is implemented using sparse matrix operations
- The adjacency matrix is normalized as $\hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$
- The model uses Bayesian Personalized Ranking (BPR) loss for optimization

## Training Procedure

1. Initialize user and item embeddings randomly
2. For each epoch:
   - Sample mini-batches of (user, positive item, negative item) triplets
   - Perform forward pass through the LightGCN model
   - Calculate BPR loss
   - Update embeddings via backpropagation
3. Evaluate on validation set periodically
4. Save the best model based on validation performance

## Inference Procedure

1. Load the trained model
2. Obtain final user and item embeddings by forward pass
3. For each user, calculate scores with all candidate items
4. Rank items by scores and recommend top-K items

## Advantages Over Traditional GCN

1. **Simplicity**: No feature transformation or non-linear activation
2. **Efficiency**: Fewer parameters and lower computational cost
3. **Effectiveness**: Better performance for recommendation tasks
4. **Interpretability**: Clearer connection to collaborative filtering principles