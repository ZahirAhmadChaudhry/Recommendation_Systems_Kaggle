'''
Global configuration and settings for Carrefour LightGCN Implementation
'''

import os
from os.path import join
import torch
import multiprocessing
from enum import Enum
from parse import parse_args
import logging
from warnings import simplefilter
import sys

# Basic path setup before anything else
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
LOG_PATH = join(CODE_PATH, 'logs')

# Create necessary directories
for path in [FILE_PATH, BOARD_PATH, LOG_PATH]:
    os.makedirs(path, exist_ok=True)

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(join(LOG_PATH, 'lightgcn.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# ------------------------------------------------------------------------
# Environment / Speed Tweaks
# ------------------------------------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# You can try increasing these if you see benefit or if your GPU has enough memory
# Setting max_split_size_mb to 512 or 1024 can help reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.9'

# Limit CPU threading to reduce overhead from too many threads
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

# Set PyTorch threads
torch.set_num_threads(4)           # how many threads for CPU ops
torch.set_num_interop_threads(4)   # for inter-op parallelism

# For GPUs with TensorFloat32 (Ampere+), enabling TF32 can speed up training:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add sources to path
sys.path.append(join(CODE_PATH, 'sources'))

# Suppress warnings
simplefilter(action="ignore", category=FutureWarning)

class Config:
    """Configuration class to manage all settings."""
    def __init__(self):
        # 1. Parse CLI arguments
        self.args = parse_args()
        
        # 2. Hardware setup
        self.setup_hardware_configs()
        
        # 3. Core checks
        self.setup_core_configs()
        
        # 4. Model configs
        self.setup_model_configs()
        
        # 5. Training configs
        self.setup_training_configs()

    def setup_hardware_configs(self):
        """Setup hardware-specific configurations."""
        # CUDA check
        self.GPU = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU else "cpu")
        
        # Use all CPU cores, or limit them to your environment setting:
        self.CORES = multiprocessing.cpu_count()
        
        if self.GPU:
            gpu_properties = torch.cuda.get_device_properties(0)
            logging.info(f"Using GPU: {gpu_properties.name}")
            logging.info(f"GPU Memory: {gpu_properties.total_memory / 1024**3:.2f} GB")
            
            total_memory = gpu_properties.total_memory / (1024**3)
            self.max_memory = int(total_memory * 1.0)
            
            # Clear CUDA cache once at init
            torch.cuda.empty_cache()

    def setup_core_configs(self):
        """Setup core dataset/model checks."""
        self.all_dataset = ['carrefour']
        self.all_models = ['lgn']
        
        self.dataset = self.args.dataset
        self.model_name = self.args.model
        
        if self.dataset not in self.all_dataset:
            raise NotImplementedError(f"Dataset {self.dataset} not supported.")
        if self.model_name not in self.all_models:
            raise NotImplementedError(f"Model {self.model_name} not supported.")

    def setup_model_configs(self):
        """Setup LightGCN (or other model) configurations."""
        self.config = {
            # No longer forced min(...) for these
            'bpr_batch_size': self.args.bpr_batch, 
            'latent_dim_rec': self.args.recdim,
            'lightGCN_n_layers': self.args.layer,
            'dropout': self.args.dropout,
            'keep_prob': self.args.keepprob,
            'A_n_fold': self.args.a_fold,
            'test_u_batch_size': self.args.testbatch,  # Also no forced min
            # Training
            'lr': self.args.lr,
            'decay': self.args.decay,
            'multicore': self.args.multicore,
            # Carrefour specifics
            'n_users': 20000,
            'n_items': 82966,
            'top_k': 10,
            # Memory optimization
            'sparse_graph': True,
            'use_sparse_tensors': True,
            'test_batch_size': 800,
            'evaluation_load_size': 4000,
            'use_cpu_eval': False,
            'cuda_empty_cache': True,
            'pin_memory': True,
            'optimize_memory': True,
            'batch_split_size': 8,
            'max_memory_per_batch': 256,
            'build_graph_in_chunks': True,
            'graph_chunk_size': 8000,
            'cache_refresh_rate': 10,
            # Performance
            # Adjust workers/prefetch_factor if you see it helps or hurts
            'num_workers': 7,
            'prefetch_factor': 2,
            'persistent_workers': True,
            # Graph parameters
            'A_split': False,
            'pretrain': 0,
            # Checkpointing / early stopping
            'checkpoint_freq': 2,
            'save_best_only': True,
            'patience': 3,
            'min_delta': 0.0001
        }

    def setup_training_configs(self):
        """Setup training configs (epochs, load, etc.)."""
        self.TRAIN_epochs = self.args.epochs
        self.LOAD = self.args.load
        self.PATH = self.args.path
        self.topks = eval(self.args.topks)
        self.tensorboard = self.args.tensorboard
        self.comment = self.args.comment
        self.seed = self.args.seed

    def print_config(self):
        """Print out the config details."""
        logging.info("\n=== Configuration ===")
        logging.info(f"Dataset: {self.dataset}")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Epochs: {self.TRAIN_epochs}")
        logging.info(f"Batch Size: {self.config['bpr_batch_size']}")
        logging.info(f"Learning Rate: {self.config['lr']}")
        logging.info(f"Memory Optimizations: Enabled")
        logging.info(f"Test Batch Size: {self.config['test_batch_size']}")
        logging.info(f"Evaluation Chunk Size: {self.config['evaluation_load_size']}")
        logging.info(f"CPU Cores: {self.CORES}")
        if self.GPU:
            logging.info(f"Max Memory Usage: {self.max_memory}GB")
            logging.info(f"Memory Efficient Features: Enabled")
        logging.info("==================\n")

def cprint(words: str):
    """Colored print function."""
    print(f"\033[0;30;43m{words}\033[0m")

# ASCII art logo
logo = r"""
██╗     ██╗ ██████╗ ██╗  ██╗████████╗ ██████╗  ██████╗███╗   ██╗
██║     ██║██╔════╝ ██║  ██║╚══██╔══╝██╔════╝ ██╔════╝████╗  ██║
██║     ██║██║  ███╗███████║   ██║   ██║  ███╗██║     ██╔██╗ ██║
██║     ██║██║   ██║██╔══██║   ██║   ██║   ██║██║     ██║╚██╗██║
███████╗██║╚██████╔╝██║  ██║   ██║   ╚██████╔╝╚██████╗██║ ╚████║
╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═══╝
"""

try:
    # Initialize global configuration
    world_config = Config()

    # Export all configurations as module-level variables
    config = world_config.config
    device = world_config.device
    GPU = world_config.GPU
    CORES = world_config.CORES
    seed = world_config.seed
    dataset = world_config.dataset
    model_name = world_config.model_name
    TRAIN_epochs = world_config.TRAIN_epochs
    LOAD = world_config.LOAD
    PATH = world_config.PATH
    topks = world_config.topks
    tensorboard = world_config.tensorboard
    comment = world_config.comment

    # Print configuration at import
    world_config.print_config()

except Exception as e:
    logging.error(f"Error initializing configuration: {str(e)}")
    raise
