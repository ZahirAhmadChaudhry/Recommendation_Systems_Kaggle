import world
import utils
import torch
import numpy as np
from tensorboardX import SummaryWriter
from time import time
import Procedure
from os.path import join
import os
import logging
import gc
import datetime

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(join(world.LOG_PATH, 'training.log')),
        logging.StreamHandler()
    ]
)

def initialize_model_and_optimizer(dataset):
    """Initialize model, optimizer, and load checkpoint if needed"""
    try:
        # Set random seeds for reproducibility
        utils.set_seed(world.seed)
        logging.info(f"Random seed set to: {world.seed}")
        
        # Import and initialize model
        from register import MODELS
        model = MODELS[world.model_name](world.config, dataset)
        model = model.to(world.device)
        logging.info(f"Model initialized and moved to device: {world.device}")

        # Initialize optimizer and loss
        bpr = utils.BPRLoss(model, world.config)
        logging.info("BPR Loss initialized")

        # Load checkpoint if needed
        start_epoch = 0
        best_hitrate = 0.0
        if world.LOAD:
            start_epoch, best_hitrate = utils.load_checkpoint(model, bpr.opt)
            logging.info(f"Loaded checkpoint: epoch {start_epoch}, best hitrate: {best_hitrate:.4f}")
        else:
            logging.info(f"Training from scratch. Model will be saved to: {utils.getFileName()}")

        return model, bpr, start_epoch, best_hitrate

    except Exception as e:
        logging.error(f"Error initializing model and optimizer: {str(e)}")
        raise

def setup_tensorboard():
    """Setup tensorboard writer"""
    if world.tensorboard:
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        w = SummaryWriter(
            f"{world.BOARD_PATH}/{world.model_name}/{current_time}"
            + f"_{world.dataset}_{world.comment}")
        return w
    return None

def train_model(dataset, model, bpr, start_epoch, best_hitrate, writer=None):
    """
    Main training loop with partial evaluation:
    We'll evaluate less by only doing it every 10 epochs
    and also at the very end.
    """
    try:
        for epoch in range(start_epoch, world.TRAIN_epochs):
            try:
                start_time = time()
                output_information = Procedure.train_one_epoch(dataset, model, bpr, epoch, w=writer)
                logging.info(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                logging.info(f"Epoch time: {time() - start_time:.2f}s")
                
                # Evaluate less frequently: every 10 epochs
                if (epoch + 1) % 10 == 0:
                    test_start_time = time()
                    results = Procedure.test_one_epoch(dataset, model, epoch, w=writer, multicore=world.CORES)
                    logging.info(f'Evaluation time: {time() - test_start_time:.2f}s')
                    hitrate = results.get(f'hitrate@{world.topks[0]}', 0.0)
                    # If you want to track best hitrate in these partial evals:
                    if hitrate > best_hitrate:
                        best_hitrate = hitrate
                        if world.config['save_best_only']:
                            utils.save_checkpoint(model, epoch, bpr.opt, best_hitrate)
                
                # Save checkpoint every 10 epochs anyway
                if (epoch + 1) % 10 == 0 and not world.config['save_best_only']:
                    utils.save_checkpoint(model, epoch, bpr.opt, best_hitrate)
                    logging.info(f"Checkpoint saved at epoch {epoch+1}")
                
                # End of epoch memory cleanup
                if world.config['cuda_empty_cache']:
                    gc.collect()
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error during epoch {epoch}: {str(e)}")
                utils.save_checkpoint(model, epoch, bpr.opt, best_hitrate)
                raise
        
        # Final test after all epochs
        logging.info(f"Training completed for {world.TRAIN_epochs} epochs. Starting final evaluation...")
        final_test_time = time()
        results = Procedure.test_one_epoch(dataset, model, world.TRAIN_epochs, w=writer, multicore=world.CORES)
        logging.info(f'Final testing after {world.TRAIN_epochs} epochs: {results}')
        hitrate = results.get(f'hitrate@{world.topks[0]}', 0.0)
        
        # Save final model
        utils.save_checkpoint(model, world.TRAIN_epochs, bpr.opt, hitrate)
        logging.info(f"Final model saved with hitrate: {hitrate:.4f}")
        logging.info(f"Final evaluation time: {time() - final_test_time:.2f}s")
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        utils.save_checkpoint(model, epoch, bpr.opt, best_hitrate)
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise
    finally:
        if writer is not None:
            writer.close()

def main():
    """Main function with error handling and memory management"""
    try:
        from register import dataset
        logging.info("Starting LightGCN training for Carrefour dataset")
        logging.info(world.logo)
        
        # Initialize model and optimizer
        model, bpr, start_epoch, best_hitrate = initialize_model_and_optimizer(dataset)
        
        # Setup tensorboard
        writer = setup_tensorboard()
        
        # Train model with less-frequent evaluation
        train_model(dataset, model, bpr, start_epoch, best_hitrate, writer)
        
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise
    finally:
        if world.config['cuda_empty_cache'] and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
