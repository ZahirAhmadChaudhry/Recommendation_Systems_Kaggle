"""
Main script for training and using the NGCF model.

This script provides a command-line interface for training and evaluating
the Neural Graph Collaborative Filtering (NGCF) model.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import time

from src.data.data_loader import (
    load_data, transform_data, preprocess_test_data,
    create_graph, save_graph, load_graph, initialize_embeddings,
    prepare_data, prepare_test_data
)
from src.model.ngcf_model import NGCF
from src.utils.training import (
    train, evaluate, hitrate_at_k, generate_recommendations, 
    create_submission_file
)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="NGCF: Neural Graph Collaborative Filtering")
    
    # Data arguments
    parser.add_argument('--train_path', type=str, default='data-train',
                        help='Path to training data directory or file')
    parser.add_argument('--products_path', type=str, default='data-train/products_data.csv',
                        help='Path to product data file')
    parser.add_argument('--test_path', type=str, default='data-train/test_data.csv',
                        help='Path to test data file')
    parser.add_argument('--graph_path', type=str, default='graph_data',
                        help='Path to save/load graph data')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Dimension of embeddings')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of graph convolutional layers')
    parser.add_argument('--dropout_ratio', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--layer_dropout', type=float, default=0.1,
                        help='Dropout rate for layer embeddings')
    parser.add_argument('--final_dropout', type=float, default=0.0,
                        help='Dropout rate for final embeddings')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for L2 regularization')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='Number of epochs for early stopping')
    
    # Evaluation arguments
    parser.add_argument('--k', type=int, default=10,
                        help='Number of top items to recommend')
    
    # Operational arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How many batches to wait before logging')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the model after training')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'recommend'],
                        default='train', help='Mode to run the script in')
    
    return parser.parse_args()


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main function to run the script.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        # Load and preprocess data
        print("Loading and preprocessing data...")
        train_data, products_data, test_data = load_data(
            args.train_path, args.products_path, args.test_path
        )
        
        purchase_data = transform_data(train_data, products_data)
        test_set = preprocess_test_data(test_data)
        
        # Create or load graph
        if os.path.exists(os.path.join(args.graph_path, "nodes.pkl")):
            print("Loading existing graph...")
            graph = load_graph(args.graph_path)
        else:
            print("Creating graph from data...")
            graph = create_graph(purchase_data)
            os.makedirs(args.graph_path, exist_ok=True)
            save_graph(graph, args.graph_path)
        
        # Initialize embeddings
        print("Initializing embeddings...")
        (customer_embeddings, product_embeddings, 
         customer_node_indices, product_node_indices,
         customer_to_int, product_to_int) = initialize_embeddings(
            graph, purchase_data, embedding_dim=args.embedding_dim
        )
        
        # Calculate model parameters
        num_users = len(customer_to_int)
        num_items = len(product_to_int)
        print(f"Number of users: {num_users}, Number of items: {num_items}")
        
        # Create model
        model = NGCF(
            num_users=num_users,
            num_items=num_items,
            embedding_size=args.embedding_dim,
            num_layers=args.num_layers,
            dropout_ratio=args.dropout_ratio,
            p1=args.layer_dropout,
            p2=args.final_dropout
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # Prepare training data
        print("Preparing training data...")
        train_loader = prepare_data(
            graph, 
            batch_size=args.batch_size,
            model=model,
            item_embeddings=product_embeddings
        )
        
        # Train model
        print("Starting training...")
        best_hit_rate = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            # Train for one epoch
            loss = train(model, train_loader, optimizer, epoch, device)
            
            # Evaluate on test set
            test_hit_rate = evaluate(model, train_loader, device, k=args.k)
            
            print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}, Hit Rate@{args.k}: {test_hit_rate:.4f}")
            
            # Check if we have a new best model
            if test_hit_rate > best_hit_rate:
                best_hit_rate = test_hit_rate
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                if args.save_model:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'hit_rate': test_hit_rate,
                        'customer_to_int': customer_to_int,
                        'product_to_int': product_to_int
                    }, os.path.join(args.output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Training completed. Best Hit Rate@{args.k}: {best_hit_rate:.4f} at epoch {best_epoch}")
        
        # Generate recommendations for test users
        print("Generating recommendations for test users...")
        test_user_indices = torch.tensor(
            [customer_to_int[f"Customer_{customer_id}"] for customer_id in test_set['customer_id'].unique()],
            dtype=torch.long
        )
        
        recommendations = generate_recommendations(
            model, test_user_indices, num_items, top_k=args.k, device=device
        )
        
        # Create submission file
        submission_df = create_submission_file(
            recommendations, 
            customer_to_int,
            product_to_int,
            os.path.join(args.output_dir, 'ngcf_submission.csv')
        )
        
        print(f"Recommendations generated and saved to {os.path.join(args.output_dir, 'ngcf_submission.csv')}")
    
    elif args.mode == 'evaluate':
        # Load test data
        print("Loading test data...")
        _, _, test_data = load_data(
            args.train_path, args.products_path, args.test_path
        )
        test_set = preprocess_test_data(test_data)
        
        # Load the saved model
        if args.load_model is None:
            args.load_model = os.path.join(args.output_dir, 'best_model.pth')
        
        checkpoint = torch.load(args.load_model, map_location=device)
        customer_to_int = checkpoint['customer_to_int']
        product_to_int = checkpoint['product_to_int']
        
        num_users = len(customer_to_int)
        num_items = len(product_to_int)
        
        # Create model and load state
        model = NGCF(
            num_users=num_users,
            num_items=num_items,
            embedding_size=args.embedding_dim,
            num_layers=args.num_layers,
            dropout_ratio=args.dropout_ratio,
            p1=args.layer_dropout,
            p2=args.final_dropout
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate recommendations
        print("Generating recommendations for test users...")
        test_user_indices = torch.tensor(
            [customer_to_int.get(f"Customer_{customer_id}", 0) for customer_id in test_set['customer_id'].unique()],
            dtype=torch.long
        )
        
        recommendations = generate_recommendations(
            model, test_user_indices, num_items, top_k=args.k, device=device
        )
        
        # Create submission file
        submission_df = create_submission_file(
            recommendations, 
            customer_to_int,
            product_to_int,
            os.path.join(args.output_dir, 'ngcf_submission.csv')
        )
        
        print(f"Recommendations generated and saved to {os.path.join(args.output_dir, 'ngcf_submission.csv')}")
    
    elif args.mode == 'recommend':
        # Load the saved model
        if args.load_model is None:
            args.load_model = os.path.join(args.output_dir, 'best_model.pth')
        
        checkpoint = torch.load(args.load_model, map_location=device)
        customer_to_int = checkpoint['customer_to_int']
        product_to_int = checkpoint['product_to_int']
        
        num_users = len(customer_to_int)
        num_items = len(product_to_int)
        
        # Create model and load state
        model = NGCF(
            num_users=num_users,
            num_items=num_items,
            embedding_size=args.embedding_dim,
            num_layers=args.num_layers,
            dropout_ratio=args.dropout_ratio,
            p1=args.layer_dropout,
            p2=args.final_dropout
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get user input
        while True:
            user_input = input("Enter a customer ID (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            try:
                # Convert input to the expected format
                if not user_input.startswith("Customer_"):
                    user_input = f"Customer_{user_input}"
                
                if user_input not in customer_to_int:
                    print(f"Customer {user_input} not found in the training data.")
                    continue
                
                user_idx = customer_to_int[user_input]
                user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
                
                # Generate recommendations
                recs = generate_recommendations(model, user_tensor, num_items, top_k=args.k, device=device)
                
                # Display recommendations
                print(f"Recommendations for {user_input}:")
                for idx, (_, item_indices) in enumerate(recs):
                    for rank, item_idx in enumerate(item_indices, 1):
                        # Get the original product ID
                        for prod_id, prod_idx in product_to_int.items():
                            if prod_idx == item_idx:
                                print(f"  {rank}. {prod_id}")
                                break
            
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()