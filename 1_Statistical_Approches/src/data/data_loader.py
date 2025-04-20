"""
Data loading module for reading datasets from different sources.
Handles loading train, test and products data with consistent interfaces.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional

# Import optimizer for data optimization
from .data_optimizer import DataOptimizer


class DataLoader:
    """
    Handles loading of train, test, and products datasets.
    Provides functionality to load from CSV, Parquet, or other sources.
    """
    
    def __init__(self, data_dir: Union[str, Path], enhanced_data_dir: Optional[Union[str, Path]] = None, 
                 log_level: int = logging.INFO):
        """
        Initialize DataLoader with paths to data directories.
        
        Args:
            data_dir: Path to the directory containing original data files
            enhanced_data_dir: Path to the directory containing enhanced/optimized data files
            log_level: Logging level (default: INFO)
        """
        # Convert to Path objects
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.enhanced_data_dir = None
        if enhanced_data_dir:
            self.enhanced_data_dir = Path(enhanced_data_dir) if isinstance(enhanced_data_dir, str) else enhanced_data_dir
            
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create data optimizer
        self.optimizer = DataOptimizer(log_level=log_level)
        
    def load_train_data(self, use_optimized: bool = True) -> pd.DataFrame:
        """
        Load train dataset from file.
        
        Args:
            use_optimized: Whether to use optimized version if available
            
        Returns:
            Train DataFrame
        """
        # Try to load optimized data first if requested
        if use_optimized and self.enhanced_data_dir and (self.enhanced_data_dir / "optimized_train.parquet").exists():
            self.logger.info("Loading optimized train data...")
            try:
                df = pd.read_parquet(self.enhanced_data_dir / "optimized_train.parquet")
                self.logger.info(f"Loaded optimized train data: {len(df)} rows, {len(df.columns)} columns")
                return df
            except Exception as e:
                self.logger.warning(f"Failed to load optimized train data: {e}")
        
        # Load original data
        self.logger.info("Loading original train data...")
        try:
            # Try parquet first
            if (self.data_dir / "train.parquet").exists():
                df = pd.read_parquet(self.data_dir / "train.parquet")
            # Then try CSV
            elif (self.data_dir / "train.csv").exists():
                df = pd.read_csv(self.data_dir / "train.csv")
            else:
                raise FileNotFoundError("No train data file found in data directory")
            
            self.logger.info(f"Loaded original train data: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading train data: {e}")
            raise
    
    def load_products_data(self, use_optimized: bool = True) -> pd.DataFrame:
        """
        Load products dataset from file.
        
        Args:
            use_optimized: Whether to use optimized version if available
            
        Returns:
            Products DataFrame
        """
        # Try to load optimized data first if requested
        if use_optimized and self.enhanced_data_dir and (self.enhanced_data_dir / "optimized_products.parquet").exists():
            self.logger.info("Loading optimized products data...")
            try:
                df = pd.read_parquet(self.enhanced_data_dir / "optimized_products.parquet")
                self.logger.info(f"Loaded optimized products data: {len(df)} rows, {len(df.columns)} columns")
                return df
            except Exception as e:
                self.logger.warning(f"Failed to load optimized products data: {e}")
        
        # Load original data
        self.logger.info("Loading original products data...")
        try:
            # Try parquet first
            if (self.data_dir / "products.parquet").exists():
                df = pd.read_parquet(self.data_dir / "products.parquet")
            # Then try CSV
            elif (self.data_dir / "products.csv").exists():
                df = pd.read_csv(self.data_dir / "products.csv")
            else:
                raise FileNotFoundError("No products data file found in data directory")
            
            self.logger.info(f"Loaded original products data: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading products data: {e}")
            raise
    
    def load_test_data(self, use_optimized: bool = True) -> pd.DataFrame:
        """
        Load test dataset from file.
        
        Args:
            use_optimized: Whether to use optimized version if available
            
        Returns:
            Test DataFrame
        """
        # Try to load optimized data first if requested
        if use_optimized and self.enhanced_data_dir and (self.enhanced_data_dir / "optimized_test.parquet").exists():
            self.logger.info("Loading optimized test data...")
            try:
                df = pd.read_parquet(self.enhanced_data_dir / "optimized_test.parquet")
                self.logger.info(f"Loaded optimized test data: {len(df)} rows, {len(df.columns)} columns")
                return df
            except Exception as e:
                self.logger.warning(f"Failed to load optimized test data: {e}")
        
        # Load original data
        self.logger.info("Loading original test data...")
        try:
            # Try parquet first
            if (self.data_dir / "test.parquet").exists():
                df = pd.read_parquet(self.data_dir / "test.parquet")
            # Then try CSV
            elif (self.data_dir / "test.csv").exists():
                df = pd.read_csv(self.data_dir / "test.csv")
            else:
                raise FileNotFoundError("No test data file found in data directory")
            
            self.logger.info(f"Loaded original test data: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            raise
    
    def load_all_data(self, use_optimized: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets (train, products, test).
        
        Args:
            use_optimized: Whether to use optimized versions if available
            
        Returns:
            Tuple of (train, products, test) DataFrames
        """
        self.logger.info("Loading all datasets...")
        train_df = self.load_train_data(use_optimized)
        products_df = self.load_products_data(use_optimized)
        test_df = self.load_test_data(use_optimized)
        
        return train_df, products_df, test_df
    
    def optimize_and_save_all_data(self, output_dir: Union[str, Path] = None) -> Dict[str, Path]:
        """
        Load, optimize, and save all datasets.
        
        Args:
            output_dir: Directory to save optimized data files
            
        Returns:
            Dictionary with paths to saved files
        """
        self.logger.info("Optimizing and saving all datasets...")
        
        # Set output directory
        if output_dir is None:
            if self.enhanced_data_dir:
                output_dir = self.enhanced_data_dir
            else:
                output_dir = self.data_dir / "enhanced_data"
        
        # Ensure output_dir is a Path object
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original data
        train_df = self.load_train_data(use_optimized=False)
        products_df = self.load_products_data(use_optimized=False)
        test_df = self.load_test_data(use_optimized=False)
        
        # Optimize data
        optimized_train = self.optimizer.optimize_train_data(train_df)
        optimized_products = self.optimizer.optimize_products_data(products_df)
        optimized_test = self.optimizer.optimize_test_data(test_df)
        
        # Save optimized data
        train_path = self.optimizer.save_optimized_data(optimized_train, "train", output_dir)
        products_path = self.optimizer.save_optimized_data(optimized_products, "products", output_dir)
        test_path = self.optimizer.save_optimized_data(optimized_test, "test", output_dir)
        
        # Return paths to saved files
        return {
            "train": train_path,
            "products": products_path,
            "test": test_path
        }
    
    def get_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic statistics about a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "null_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns}
        }
        
        return stats