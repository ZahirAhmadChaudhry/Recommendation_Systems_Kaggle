"""
Training and testing procedures for LightGCN
"""
import world
import numpy as np
import torch
import utils
import gc
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

def train_one_epoch(dataset, model, bpr, epoch, w=None):
    """Train for one epoch with memory optimizations"""
    model.train()
    with utils.timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    
    # Shuffle entire sample once
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    
    with utils.timer(name="Train"):
        pbar = tqdm(range(total_batch), desc=f'Epoch {epoch}')
        for (batch_i,
             (batch_users,
              batch_pos,
              batch_neg)) in enumerate(
                utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
            
            loss = bpr.stageOne(batch_users, batch_pos, batch_neg)
            aver_loss += loss
            
            if w:
                w.add_scalar(f'BPRLoss/BPR', loss, epoch * total_batch + batch_i)
            
            # Show minimal log on progress bar
            pbar.set_postfix(loss=f"{loss:.4f}")
            pbar.update(1)

            # [Removed gc.collect() and torch.cuda.empty_cache() calls from inside the batch loop]
        
        pbar.close()
    
    # Final memory cleanup at the end of the epoch
    del S, users, posItems, negItems
    gc.collect()
    if world.config['cuda_empty_cache']:
        torch.cuda.empty_cache()
    
    return f"loss{aver_loss/total_batch:.3f}-{utils.timer.dict()}"

def test_one_epoch(dataset, model, epoch, w=None, multicore=0):
    """Evaluate model with memory optimizations"""
    model.eval()
    max_K = max(world.topks)
    
    with torch.no_grad():
        users = list(dataset.testDict.keys())
        users_list = []
        ratings_list = []
        groundTrue_list = []
        
        # Process users in optimized chunks
        chunk_size = world.config['evaluation_load_size']
        total_chunks = (len(users) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, len(users))
            chunk_users = users[chunk_start:chunk_end]
            
            # Move user tensor to GPU once
            user_tensor = torch.Tensor(chunk_users).long().to(world.device)
            ratings = model.getUsersRating(user_tensor)
            
            # Optimize exclusion process
            exclude_index = []
            exclude_items = []
            for idx, user in enumerate(chunk_users):
                train_items = set(dataset.allPos[user]) - set(dataset.testDict[user])
                exclude_index.extend([idx] * len(train_items))
                exclude_items.extend(train_items)
            
            if exclude_index:
                ratings[exclude_index, exclude_items] = -(1<<10)
            
            # Get top K efficiently
            _, ratings_K = torch.topk(ratings, k=max_K)
            
            users_list.extend(chunk_users)
            ratings_list.append(ratings_K.cpu())
            groundTrue_list.extend([dataset.testDict[u] for u in chunk_users])
            
            # Clear memory
            del ratings, user_tensor
            if world.config['cuda_empty_cache']:
                gc.collect()
                torch.cuda.empty_cache()
        
        ratings_list = torch.cat(ratings_list, dim=0)
        hitrate = 0
        
        # Reduced logs: removing excessive debug prints
        # print("\nDEBUG: Checking predictions and ground truth")
        # print(f"Number of users being evaluated: {len(users)}")
        
        for i, (rating_K, groundTrue) in enumerate(zip(ratings_list, groundTrue_list)):
            pred_items = rating_K[:world.topks[0]].tolist()
            # Removed detailed per-user debug
            
            non_null_count = len(set(pred_items) & set(groundTrue))
            distinct_product_count = len(set(groundTrue))
            denominator = min(distinct_product_count, world.topks[0])
            hitrate += non_null_count / denominator if denominator > 0 else 0
        
        hitrate = hitrate / len(users)
        print(f"Final average hitrate (epoch {epoch}): {hitrate:.4f}")
        
        if w:
            w.add_scalar(f'Hitrate/HR_{world.topks[0]}', hitrate, epoch)
    
    return {f'hitrate@{world.topks[0]}': hitrate}
