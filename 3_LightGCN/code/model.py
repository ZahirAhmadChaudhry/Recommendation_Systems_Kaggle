"""
LightGCN Model Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import world
import numpy as np
import utils

class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()
        self.sparse_graph = config['sparse_graph']
        self.use_sparse_tensors = config['use_sparse_tensors']
        
    def _init_weight(self):
        """Initialize weights using Xavier initialization"""
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        
        # Initialize embeddings
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        
        self.f = nn.Sigmoid()
        self.Graph = None

    def computer(self):
        """
        Propagate embeddings through the graph
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        # Get graph structure
        if self.Graph is None:
            self.Graph = self.dataset.getSparseGraph()
            
            if self.Graph is None:
                raise ValueError("Failed to build graph. Check graph construction in dataset.")
            
            if self.sparse_graph and self.use_sparse_tensors:
                self.Graph = self.Graph.coalesce()
        
        embs = [all_emb]
        
        # Graph propagation
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = torch.zeros_like(all_emb)
                for h in self.Graph:
                    temp_emb = temp_emb + torch.sparse.mm(h, all_emb)
            else:
                temp_emb = torch.sparse.mm(self.Graph, all_emb)
            
            if world.config['cuda_empty_cache']:
                # We still keep it once per layer if memory is a concern
                torch.cuda.empty_cache()
            
            all_emb = temp_emb
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        """Get ratings for users"""
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        """Get embeddings for users and items"""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        """Calculate BPR loss"""
        (users_emb, pos_emb, neg_emb, 
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2) + 
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        pos_scores = torch.mul(users_emb, pos_emb)
        neg_scores = torch.mul(users_emb, neg_emb)
        
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
    
    def forward(self, users, items):
        """Forward pass"""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
