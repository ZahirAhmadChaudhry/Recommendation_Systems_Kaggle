"""
Main script for running the statistical approaches recommendation pipeline.
Handles data loading, feature engineering, model training, and evaluation.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, Any, List, Union, Tuple, Optional

# Import components from the data module
from src.data.data_loader import DataLoader
from src.data.data_optimizer import DataOptimizer

# Import components from the utils module
from src.utils.data_validator import DataValidator
from src.utils.feature_engineering import FeatureEngineer
from src.utils.evaluation import RecommendationEvaluator

# Import components from the models module
from src.models.statistical_models import StatisticalRecommender


def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("recommendation_pipeline")
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Statistical Approaches Recommendation Pipeline')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='../cleaned_data',
                        help='Directory containing the cleaned data files')
    parser.add_argument('--enhanced-data-dir', type=str, default='Enhanced_data',
                        help='Directory containing enhanced/optimized data files')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results and visualizations')
    
    # Pipeline flags
    parser.add_argument('--optimize-data', action='store_true',
                        help='Optimize data before processing')
    parser.add_argument('--use-optimized', action='store_true',
                        help='Use previously optimized data')
    parser.add_argument('--feature-engineering', action='store_true',
                        help='Generate and save features')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['frequency', 'recency', 'hybrid', 'segmented'],
                        help='Models to train and evaluate')
    
    # Evaluation settings
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='k values for evaluation metrics')
    
    # Output settings
    parser.add_argument('--save-recommendations', action='store_true',
                        help='Save recommendations to file')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save evaluation plots')
    parser.add_argument('--analyze-recommendations', action='store_true',
                        help='Analyze recommendation patterns')
    
    # Logging settings
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file')
    
    return parser.parse_args()


def main():
    """Main function to run the recommendation pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logger = setup_logging(log_levels[args.log_level], args.log_file)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamps for files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    data_loader = DataLoader(args.data_dir, args.enhanced_data_dir)
    data_optimizer = DataOptimizer()
    feature_engineer = FeatureEngineer()
    recommender = StatisticalRecommender()
    evaluator = RecommendationEvaluator()
    
    # Load data
    logger.info("Loading data...")
    if args.use_optimized:
        logger.info("Using optimized data...")
        train_df, products_df, test_df = data_loader.load_all_data(use_optimized=True)
    else:
        logger.info("Loading original data...")
        train_df, products_df, test_df = data_loader.load_all_data(use_optimized=False)
    
    # Optimize data if requested
    if args.optimize_data:
        logger.info("Optimizing data...")
        if not args.use_optimized:
            train_df = data_optimizer.optimize_train_data(train_df)
            products_df = data_optimizer.optimize_products_data(products_df)
            test_df = data_optimizer.optimize_test_data(test_df)
        
        # Save optimized data
        optimized_dir = Path(args.enhanced_data_dir)
        optimized_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving optimized data...")
        data_optimizer.save_optimized_data(train_df, "train", optimized_dir)
        data_optimizer.save_optimized_data(products_df, "products", optimized_dir)
        data_optimizer.save_optimized_data(test_df, "test", optimized_dir)
    
    # Generate features if requested
    if args.feature_engineering:
        logger.info("Generating features...")
        features_dir = output_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        features = feature_engineer.generate_all_features(
            train_df, products_df, output_dir=features_dir
        )
        
        # Print feature info
        for feature_type, feature_df in features.items():
            logger.info(f"{feature_type.capitalize()} features shape: {feature_df.shape}")
    
    # Get test customers
    test_customers = test_df['customer_id'].unique()
    logger.info(f"Test set has {len(test_customers)} unique customers")
    
    # Train and evaluate models
    logger.info("Training and evaluating models...")
    model_recommendations = {}
    
    # Configure model parameters
    model_params = {
        'frequency': {'top_k': 10},
        'recency': {'top_k': 10},
        'popularity': {'top_k': 10},
        'hybrid': {'freq_weight': 0.7, 'recency_weight': 0.3, 'top_k': 10},
        'segmented': {'n_segments': 3, 'top_k': 10},
        'weighted': {
            'models': {
                'frequency': 0.4, 
                'recency': 0.3, 
                'popularity': 0.3
            },
            'top_k': 10
        }
    }
    
    # Train selected models
    for model_name in args.models:
        logger.info(f"Training model: {model_name}")
        
        if model_name == 'frequency':
            recs = recommender.frequency_based_recommendations(
                train_df, test_customers, **model_params['frequency']
            )
        elif model_name == 'recency':
            recs = recommender.recency_based_recommendations(
                train_df, test_customers, **model_params['recency']
            )
        elif model_name == 'popularity':
            recs = recommender.popularity_based_recommendations(
                train_df, test_customers, **model_params['popularity']
            )
        elif model_name == 'hybrid':
            recs = recommender.hybrid_recommendations(
                train_df, test_customers, **model_params['hybrid']
            )
        elif model_name == 'segmented':
            recs = recommender.customer_segmented_recommendations(
                train_df, test_customers, products_df, **model_params['segmented']
            )
        elif model_name == 'weighted':
            recs = recommender.weighted_hybrid_model(
                train_df, test_customers, **model_params['weighted']
            )
        else:
            logger.warning(f"Unknown model: {model_name}. Skipping.")
            continue
        
        # Save recommendations
        if args.save_recommendations:
            recs_path = output_dir / f"{model_name}_recommendations_{timestamp}.csv"
            recommender.save_recommendations(recs, recs_path)
        
        # Store for evaluation
        model_recommendations[model_name] = recs
    
    # Evaluate and compare models
    logger.info("Evaluating models...")
    comparison_results = evaluator.compare_models(
        test_df, model_recommendations, args.k_values
    )
    
    # Save comparison results
    comparison_path = output_dir / f"model_comparison_{timestamp}.csv"
    comparison_results.to_csv(comparison_path, index=False)
    logger.info(f"Saved model comparison to {comparison_path}")
    
    # Create visualizations if requested
    if args.save_plots:
        logger.info("Creating visualizations...")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot model comparison
        comparison_plot_path = plots_dir / f"model_comparison_{timestamp}.png"
        evaluator.plot_model_comparison(
            comparison_results, 'hitrate@10', output_path=comparison_plot_path
        )
        
        # Plot metrics by k
        metrics_plot_path = plots_dir / f"metrics_by_k_{timestamp}.png"
        evaluator.plot_metrics_by_k(
            comparison_results, model_name=None, output_path=metrics_plot_path
        )
    
    # Analyze recommendations if requested
    if args.analyze_recommendations:
        logger.info("Analyzing recommendations...")
        
        # Find best model based on hit rate at 10
        best_model = comparison_results.loc[
            comparison_results['hitrate@10'].idxmax()
        ]['model']
        
        logger.info(f"Best model: {best_model}")
        
        # Analyze best model's recommendations
        best_recs = model_recommendations[best_model]
        analysis_results = evaluator.analyze_recommendations(
            best_recs, test_df, products_df
        )
        
        # Save analysis results
        analysis_path = output_dir / f"recommendation_analysis_{timestamp}.json"
        with open(analysis_path, 'w') as f:
            # Convert any non-serializable values to strings
            serializable_results = {}
            for k, v in analysis_results.items():
                if isinstance(v, dict):
                    serializable_results[k] = {str(k2): float(v2) for k2, v2 in v.items()}
                elif isinstance(v, (int, float, str, bool)):
                    serializable_results[k] = v
                else:
                    serializable_results[k] = str(v)
            
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Saved analysis results to {analysis_path}")
        
        # Create visualizations of analysis
        analysis_plots_dir = output_dir / "analysis_plots"
        analysis_plots_dir.mkdir(parents=True, exist_ok=True)
        
        visualization_paths = evaluator.visualize_analysis(
            analysis_results, analysis_plots_dir
        )
        
        logger.info(f"Created {len(visualization_paths)} analysis visualizations")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()