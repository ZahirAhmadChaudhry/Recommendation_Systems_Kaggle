"""
Evaluation module for recommendation systems.
Implements metrics and visualization tools for evaluating recommendation performance.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from tqdm import tqdm

class RecommendationEvaluator:
    """
    Evaluates the performance of recommendation systems.
    Provides metrics calculation and visualization tools.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize RecommendationEvaluator with logging configuration.
        
        Args:
            log_level: Logging level (default: INFO)
        """
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def hitrate_at_k(self, 
                     test_df: pd.DataFrame, 
                     recommendations_df: pd.DataFrame, 
                     k: int = 10) -> float:
        """
        Calculate hit rate at k.
        
        Args:
            test_df: Test DataFrame with ground truth purchases
            recommendations_df: DataFrame with recommendations
            k: Number of top recommendations to consider
            
        Returns:
            Hit rate score (between 0 and 1)
        """
        self.logger.info(f"Calculating hit rate at {k}...")
        
        # Check if recommendations dataframe is empty
        if recommendations_df.empty:
            self.logger.warning("Recommendations DataFrame is empty. Returning hit rate of 0.")
            return 0.0
        
        # Create a set of customer-product pairs from test data
        test_pairs = set(zip(test_df['customer_id'], test_df['product_id']))
        
        # Filter recommendations to top-k for each customer
        top_k_recs = (recommendations_df
                      .sort_values(['customer_id', 'rank'])
                      .groupby('customer_id')
                      .head(k))
        
        # Create a set of recommended customer-product pairs
        rec_pairs = set(zip(top_k_recs['customer_id'], top_k_recs['product_id']))
        
        # Count the number of hits
        num_hits = len(test_pairs.intersection(rec_pairs))
        
        # Calculate hit rate
        hit_rate = num_hits / len(test_pairs)
        
        self.logger.info(f"Hit rate at {k}: {hit_rate:.4f}")
        return hit_rate
    
    def precision_at_k(self, 
                        test_df: pd.DataFrame, 
                        recommendations_df: pd.DataFrame, 
                        k: int = 10) -> float:
        """
        Calculate precision at k.
        
        Args:
            test_df: Test DataFrame with ground truth purchases
            recommendations_df: DataFrame with recommendations
            k: Number of top recommendations to consider
            
        Returns:
            Precision score (between 0 and 1)
        """
        self.logger.info(f"Calculating precision at {k}...")
        
        # Check if recommendations dataframe is empty
        if recommendations_df.empty:
            self.logger.warning("Recommendations DataFrame is empty. Returning precision of 0.")
            return 0.0
        
        # Create a set of customer-product pairs from test data
        test_pairs = set(zip(test_df['customer_id'], test_df['product_id']))
        
        # Filter recommendations to top-k for each customer
        top_k_recs = (recommendations_df
                      .sort_values(['customer_id', 'rank'])
                      .groupby('customer_id')
                      .head(k))
        
        # Create a set of recommended customer-product pairs
        rec_pairs = set(zip(top_k_recs['customer_id'], top_k_recs['product_id']))
        
        # Count the number of hits
        num_hits = len(test_pairs.intersection(rec_pairs))
        
        # Calculate precision
        precision = num_hits / len(rec_pairs)
        
        self.logger.info(f"Precision at {k}: {precision:.4f}")
        return precision
    
    def evaluate_model(self, 
                       test_df: pd.DataFrame, 
                       recommendations_df: pd.DataFrame, 
                       ks: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            test_df: Test DataFrame with ground truth purchases
            recommendations_df: DataFrame with recommendations
            ks: List of k values for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Evaluating model performance...")
        
        results = {}
        
        # Calculate hit rate and precision at different k values
        for k in ks:
            hr = self.hitrate_at_k(test_df, recommendations_df, k)
            precision = self.precision_at_k(test_df, recommendations_df, k)
            
            results[f'hitrate@{k}'] = hr
            results[f'precision@{k}'] = precision
        
        # Calculate average metrics
        avg_hr = np.mean([results[f'hitrate@{k}'] for k in ks])
        avg_precision = np.mean([results[f'precision@{k}'] for k in ks])
        
        results['avg_hitrate'] = avg_hr
        results['avg_precision'] = avg_precision
        
        self.logger.info(f"Evaluation results: {results}")
        return results
    
    def compare_models(self, 
                       test_df: pd.DataFrame, 
                       model_recommendations: Dict[str, pd.DataFrame], 
                       ks: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """
        Compare multiple recommendation models.
        
        Args:
            test_df: Test DataFrame with ground truth purchases
            model_recommendations: Dictionary mapping model names to recommendation DataFrames
            ks: List of k values for evaluation
            
        Returns:
            DataFrame with comparison results
        """
        self.logger.info(f"Comparing {len(model_recommendations)} models...")
        
        # Initialize results
        results = []
        
        # Evaluate each model
        for model_name, recs in model_recommendations.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Evaluate model
            metrics = self.evaluate_model(test_df, recs, ks)
            
            # Add model name to metrics
            metrics['model'] = model_name
            
            # Add to results
            results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Reorder columns
        metric_cols = [col for col in results_df.columns if col != 'model']
        results_df = results_df[['model'] + metric_cols]
        
        self.logger.info("Model comparison complete")
        return results_df
    
    def plot_model_comparison(self, 
                             comparison_df: pd.DataFrame, 
                             metric: str = 'hitrate@10', 
                             figsize: Tuple[int, int] = (10, 6), 
                             output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create visualization of model comparison.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to use for comparison
            figsize: Figure size
            output_path: Optional path to save the figure
        """
        self.logger.info(f"Creating model comparison plot for {metric}...")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Sort by metric value
        plot_df = comparison_df.sort_values(metric, ascending=False)
        
        # Create bar chart
        ax = sns.barplot(x='model', y=metric, data=plot_df)
        
        # Add title and labels
        plt.title(f'Model Comparison - {metric}', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(plot_df[metric]):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            # Convert output_path to Path if it's a string
            output_path = Path(output_path) if isinstance(output_path, str) else output_path
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {output_path}")
        
        plt.close()
    
    def plot_metrics_by_k(self, 
                         comparison_df: pd.DataFrame, 
                         model_name: Optional[str] = None, 
                         figsize: Tuple[int, int] = (12, 8), 
                         output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot metrics (hit rate and precision) by k.
        
        Args:
            comparison_df: DataFrame with model comparison results
            model_name: Optional model name to filter (if None, plot all models)
            figsize: Figure size
            output_path: Optional path to save the figure
        """
        self.logger.info("Creating metrics by k plot...")
        
        # Filter by model if provided
        if model_name:
            plot_df = comparison_df[comparison_df['model'] == model_name].copy()
            self.logger.info(f"Plotting metrics for model: {model_name}")
        else:
            plot_df = comparison_df.copy()
            self.logger.info(f"Plotting metrics for all models")
        
        # Get k values
        hitrate_cols = [col for col in plot_df.columns if col.startswith('hitrate@')]
        ks = [int(col.split('@')[1]) for col in hitrate_cols]
        
        # Extract metrics by k
        hr_data = []
        precision_data = []
        
        for k in ks:
            for _, row in plot_df.iterrows():
                model = row['model']
                hr_data.append({
                    'model': model,
                    'k': k,
                    'metric': 'Hit Rate',
                    'value': row[f'hitrate@{k}']
                })
                precision_data.append({
                    'model': model,
                    'k': k,
                    'metric': 'Precision',
                    'value': row[f'precision@{k}']
                })
        
        metrics_df = pd.DataFrame(hr_data + precision_data)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot hit rate
        hr_df = metrics_df[metrics_df['metric'] == 'Hit Rate']
        sns.lineplot(x='k', y='value', hue='model', marker='o', data=hr_df, ax=ax1)
        ax1.set_title('Hit Rate by k', fontsize=14)
        ax1.set_xlabel('k', fontsize=12)
        ax1.set_ylabel('Hit Rate', fontsize=12)
        
        # Plot precision
        prec_df = metrics_df[metrics_df['metric'] == 'Precision']
        sns.lineplot(x='k', y='value', hue='model', marker='o', data=prec_df, ax=ax2)
        ax2.set_title('Precision by k', fontsize=14)
        ax2.set_xlabel('k', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            # Convert output_path to Path if it's a string
            output_path = Path(output_path) if isinstance(output_path, str) else output_path
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved metrics by k plot to {output_path}")
        
        plt.close()
    
    def analyze_recommendations(self, 
                               recommendations_df: pd.DataFrame, 
                               test_df: pd.DataFrame, 
                               products_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze recommendations for patterns and insights.
        
        Args:
            recommendations_df: DataFrame with recommendations
            test_df: Test DataFrame with ground truth purchases
            products_df: Optional products DataFrame for additional analysis
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Analyzing recommendations...")
        
        # Check for empty dataframes
        if recommendations_df.empty:
            self.logger.warning("Recommendations DataFrame is empty.")
            return {"error": "Empty recommendations DataFrame"}
        
        analysis = {}
        
        # Basic statistics
        analysis['num_customers'] = recommendations_df['customer_id'].nunique()
        analysis['num_products'] = recommendations_df['product_id'].nunique()
        analysis['total_recommendations'] = len(recommendations_df)
        
        # Average recommendations per customer
        recs_per_customer = recommendations_df.groupby('customer_id').size()
        analysis['avg_recs_per_customer'] = recs_per_customer.mean()
        analysis['min_recs_per_customer'] = recs_per_customer.min()
        analysis['max_recs_per_customer'] = recs_per_customer.max()
        
        # Product popularity in recommendations
        product_counts = recommendations_df['product_id'].value_counts()
        analysis['top_products'] = product_counts.head(10).to_dict()
        analysis['unique_products_ratio'] = len(product_counts) / len(recommendations_df)
        
        # Rank distribution
        rank_counts = recommendations_df['rank'].value_counts().sort_index()
        analysis['rank_distribution'] = rank_counts.to_dict()
        
        # Hit analysis - recommendations that appear in test data
        test_pairs = set(zip(test_df['customer_id'], test_df['product_id']))
        rec_pairs = set(zip(recommendations_df['customer_id'], recommendations_df['product_id']))
        hits = test_pairs.intersection(rec_pairs)
        
        analysis['num_hits'] = len(hits)
        analysis['hit_ratio'] = len(hits) / len(test_pairs)
        
        # Analyze product categories if products_df is provided
        if products_df is not None:
            # Merge recommendations with product categories
            if 'department_key' in products_df.columns:
                recs_with_categories = recommendations_df.merge(
                    products_df[['product_id', 'department_key']],
                    on='product_id',
                    how='left'
                )
                
                # Category distribution in recommendations
                cat_dist = recs_with_categories['department_key'].value_counts(normalize=True)
                analysis['category_distribution'] = cat_dist.head(10).to_dict()
                
                # Get test purchases with categories
                test_with_categories = test_df.merge(
                    products_df[['product_id', 'department_key']],
                    on='product_id',
                    how='left'
                )
                
                # Category distribution in test data
                test_cat_dist = test_with_categories['department_key'].value_counts(normalize=True)
                analysis['test_category_distribution'] = test_cat_dist.head(10).to_dict()
                
                # Compare category distributions
                common_cats = set(cat_dist.index) & set(test_cat_dist.index)
                cat_diff = {cat: cat_dist.get(cat, 0) - test_cat_dist.get(cat, 0) for cat in common_cats}
                analysis['category_bias'] = {k: v for k, v in sorted(cat_diff.items(), key=lambda item: abs(item[1]), reverse=True)[:10]}
        
        self.logger.info("Recommendation analysis complete")
        return analysis
    
    def visualize_analysis(self, 
                          analysis_results: Dict[str, Any], 
                          output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Create visualizations from recommendation analysis.
        
        Args:
            analysis_results: Dictionary with analysis results
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        self.logger.info("Creating visualizations from analysis...")
        
        # Convert output_dir to Path if it's a string
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {}
        
        # 1. Rank distribution visualization
        if 'rank_distribution' in analysis_results:
            rank_dist = analysis_results['rank_distribution']
            
            plt.figure(figsize=(10, 6))
            plt.bar(rank_dist.keys(), rank_dist.values())
            plt.title('Distribution of Recommendation Ranks', fontsize=14)
            plt.xlabel('Rank', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            output_path = output_dir / 'rank_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths['rank_distribution'] = output_path
        
        # 2. Category distribution comparison
        if 'category_distribution' in analysis_results and 'test_category_distribution' in analysis_results:
            rec_cats = analysis_results['category_distribution']
            test_cats = analysis_results['test_category_distribution']
            
            # Get common categories
            common_cats = set(rec_cats.keys()) & set(test_cats.keys())
            
            if common_cats:
                # Create DataFrame for plotting
                cat_data = []
                for cat in common_cats:
                    cat_data.append({
                        'category': cat,
                        'recommendations': rec_cats.get(cat, 0),
                        'test_data': test_cats.get(cat, 0)
                    })
                
                cat_df = pd.DataFrame(cat_data)
                cat_df = cat_df.sort_values('test_data', ascending=False).head(10)
                
                plt.figure(figsize=(12, 8))
                
                # Plot both distributions as grouped bars
                x = range(len(cat_df))
                width = 0.35
                
                plt.bar(x, cat_df['recommendations'], width, label='Recommendations')
                plt.bar([i + width for i in x], cat_df['test_data'], width, label='Test Data')
                
                plt.xlabel('Category', fontsize=12)
                plt.ylabel('Proportion', fontsize=12)
                plt.title('Category Distribution: Recommendations vs Test Data', fontsize=14)
                plt.xticks([i + width/2 for i in x], cat_df['category'], rotation=45, ha='right')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                output_path = output_dir / 'category_comparison.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                output_paths['category_comparison'] = output_path
        
        # 3. Product popularity visualization
        if 'top_products' in analysis_results:
            top_products = analysis_results['top_products']
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(top_products)), list(top_products.values()))
            plt.title('Top Products in Recommendations', fontsize=14)
            plt.xlabel('Product Rank', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(range(len(top_products)), [f'P{i+1}' for i in range(len(top_products))])
            plt.grid(True, linestyle='--', alpha=0.7)
            
            output_path = output_dir / 'top_products.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths['top_products'] = output_path
        
        # 4. Hit ratio visualization
        if 'hit_ratio' in analysis_results:
            hit_ratio = analysis_results['hit_ratio']
            miss_ratio = 1 - hit_ratio
            
            plt.figure(figsize=(8, 8))
            plt.pie([hit_ratio, miss_ratio], 
                   labels=['Hits', 'Misses'], 
                   autopct='%1.1f%%',
                   colors=['#5cb85c', '#d9534f'],
                   explode=(0.1, 0),
                   startangle=90)
            plt.title('Recommendation Hit Ratio', fontsize=14)
            
            output_path = output_dir / 'hit_ratio.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths['hit_ratio'] = output_path
        
        self.logger.info(f"Created {len(output_paths)} visualizations")
        return output_paths