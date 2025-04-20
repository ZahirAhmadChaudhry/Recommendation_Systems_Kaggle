"""
Utility functions for LightGCN implementation
"""
import world
import torch
import numpy as np
from torch import nn, optim
import os
import gc
from time import time
import random

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    # Reduced log
    # print("Using Python implementation for sampling")
    sample_ext = False

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def getFileName():
    """Generate filename for model checkpoint"""
    file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}"
    return os.path.join(world.FILE_PATH, file + '.pth.tar')

def save_checkpoint(model, epoch, optimizer, best_hitrate):
    """Save model checkpoint"""
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_hitrate': best_hitrate
    }
    torch.save(state, getFileName())
    print(f"Checkpoint saved: {getFileName()}")

def load_checkpoint(model, optimizer=None):
    """Load model checkpoint"""
    filename = getFileName()
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_hitrate = checkpoint.get('best_hitrate', 0.0)
        print(f"Checkpoint loaded: epoch {epoch}, best hitrate: {best_hitrate:.4f}")
        return epoch + 1, best_hitrate
    return 0, 0.0

def clear_memory():
    """Clear memory and CUDA cache"""
    if world.config['cuda_empty_cache']:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class timer:
    """Time context manager for code block"""
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}
    
    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1
    
    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint
    
    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0
    
    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE
    
    def __enter__(self):
        self.start = time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += time() - self.start
        else:
            self.tape.append(time() - self.start)

def UniformSample_original(dataset, neg_ratio = 1):
    """Original implementation of uniform sampling"""
    allPos = dataset.allPos
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                   dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """Python implementation of uniform sampling with vectorization"""
    total_start = time()
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    
    # Vectorized operations
    batch_size = 10000
    for i in range(0, user_num, batch_size):
        batch_users = users[i:i+batch_size]
        batch_S = []
        
        for user in batch_users:
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
                
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            
            # Generate multiple negatives at once
            neg_items = np.random.randint(0, dataset.m_items, size=10)
            for negitem in neg_items:
                if negitem not in posForUser:
                    batch_S.append([user, positem, negitem])
                    break
            else:
                # If no valid negative found in batch, try one more time
                negitem = np.random.randint(0, dataset.m_items)
                while negitem in posForUser:
                    negitem = np.random.randint(0, dataset.m_items)
                batch_S.append([user, positem, negitem])
        
        S.extend(batch_S)
    
    return np.array(S)

class BPRLoss:
    """BPR Loss implementation"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.opt = optim.Adam(model.parameters(), lr=config['lr'])
    
    def stageOne(self, users, pos, neg):
        """Training step"""
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        loss = loss + self.config['decay'] * reg_loss
        
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        
        clear_memory()
        return loss.item()

def minibatch(*tensors, **kwargs):
    """Generate minibatches from tensors"""
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):
    """Shuffle arrays in unison"""
    require_indices = kwargs.get('indices', False)
    
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')
    
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    
    if require_indices:
        return result, shuffle_indices
    else:
        return result

def hitrate_at_k(scores, ground_truth, k):
    """
    Calculate hit rate at k exactly matching pandas implementation:
    
    Original pandas code:
    data = pd.merge(left=true_data, right=predicted_data, how="left", on=["customer_id", "product_id"])
    df = data[data["rank"] <= k]
    non_null_counts = df.groupby('customer_id')['rank'].apply(lambda x: x.notna().sum())
    distinct_products_per_customer = data.groupby('customer_id')['product_id'].nunique()
    denominator = min(distinct_product_count, k)
    return non_null_count / denominator
    
    Args:
        scores: tensor of top k predictions
        ground_truth: list of ground truth items
        k: number of recommendations to consider
    """
    # Get top k predictions (equivalent to rank <= k)
    pred_items = scores[:k].tolist()
    
    # Count matches (equivalent to merge and count non-null)
    non_null_count = sum(1 for idx, item in enumerate(pred_items) if item in ground_truth)
    
    # Get distinct products (like distinct_products_per_customer)
    distinct_product_count = len(set(ground_truth))
    
    # Calculate denominator exactly as in pandas
    denominator = min(distinct_product_count, k)
    
    return non_null_count / denominator if denominator > 0 else 0
