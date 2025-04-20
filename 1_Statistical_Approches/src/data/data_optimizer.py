"""
Data optimizer module for reducing memory usage and optimizing datasets.
Implements memory reduction techniques and data cleaning operations.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional

class DataOptimizer:
    """
    Utility class for optimizing dataframes to reduce memory usage.
    Also handles data cleaning and preprocessing operations.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize DataOptimizer with logging configuration.
        
        Args:
            log_level: Logging level (default: INFO)
        """
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def reduce_memory_usage(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Reduce memory usage of a DataFrame by downcasting numeric columns.
        
        Args:
            df: DataFrame to optimize
            verbose: Whether to print memory usage information
            
        Returns:
            Optimized DataFrame
        """
        # Copy the dataframe to avoid modifying the original
        df_optimized = df.copy()
        
        # Calculate initial memory usage
        start_mem = df_optimized.memory_usage(deep=True).sum() / 1024**2
        
        # Iterate through each column
        for col in df_optimized.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df_optimized[col]):
                continue
                
            # Get min and max values
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            # Integer columns
            if pd.api.types.is_integer_dtype(df_optimized[col]):
                # Convert to the smallest integer type that can hold the data
                if col_min >= 0:  # Unsigned
                    if col_max < 2**8:
                        df_optimized[col] = df_optimized[col].astype(np.uint8)
                    elif col_max < 2**16:
                        df_optimized[col] = df_optimized[col].astype(np.uint16)
                    elif col_max < 2**32:
                        df_optimized[col] = df_optimized[col].astype(np.uint32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.uint64)
                else:  # Signed
                    if col_min > -2**7 and col_max < 2**7:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif col_min > -2**15 and col_max < 2**15:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif col_min > -2**31 and col_max < 2**31:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.int64)
            
            # Float columns
            elif pd.api.types.is_float_dtype(df_optimized[col]):
                # Convert to float32 if precision allows
                df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # Calculate memory savings
        end_mem = df_optimized.memory_usage(deep=True).sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        
        if verbose:
            self.logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)')
        
        return df_optimized
    
    def optimize_train_data(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize train dataset for memory efficiency.
        
        Args:
            train_df: Original train DataFrame
            
        Returns:
            Optimized train DataFrame
        """
        self.logger.info("Optimizing train data...")
        
        # Create a copy of the dataframe
        optimized_df = train_df.copy()
        
        # Apply memory reduction
        optimized_df = self.reduce_memory_usage(optimized_df)
        
        # Apply any train-specific optimizations
        if 'order_ts' in optimized_df.columns:
            # Convert timestamps to datetime
            optimized_df['order_ts'] = pd.to_datetime(optimized_df['order_ts'])
        
        # Sort by timestamp if available for better performance
        if 'order_ts' in optimized_df.columns:
            optimized_df = optimized_df.sort_values('order_ts').reset_index(drop=True)
        
        self.logger.info(f"Train data optimization complete: {len(optimized_df)} rows")
        return optimized_df
    
    def optimize_products_data(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize products dataset for memory efficiency.
        
        Args:
            products_df: Original products DataFrame
            
        Returns:
            Optimized products DataFrame
        """
        self.logger.info("Optimizing products data...")
        
        # Create a copy of the dataframe
        optimized_df = products_df.copy()
        
        # Apply memory reduction
        optimized_df = self.reduce_memory_usage(optimized_df)
        
        # Apply any products-specific optimizations
        # Convert categorical columns to category type
        categorical_cols = [
            'category_path', 'color', 'color_family', 'gender', 'brand', 'size',
            'material', 'product_name', 'product_type'
        ]
        
        for col in categorical_cols:
            if col in optimized_df.columns:
                optimized_df[col] = optimized_df[col].astype('category')
        
        self.logger.info(f"Products data optimization complete: {len(optimized_df)} rows")
        return optimized_df
    
    def optimize_test_data(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize test dataset for memory efficiency.
        
        Args:
            test_df: Original test DataFrame
            
        Returns:
            Optimized test DataFrame
        """
        self.logger.info("Optimizing test data...")
        
        # Create a copy of the dataframe
        optimized_df = test_df.copy()
        
        # Apply memory reduction
        optimized_df = self.reduce_memory_usage(optimized_df)
        
        # Apply any test-specific optimizations
        if 'timestamp' in optimized_df.columns:
            # Convert timestamps to datetime
            optimized_df['timestamp'] = pd.to_datetime(optimized_df['timestamp'])
        
        self.logger.info(f"Test data optimization complete: {len(optimized_df)} rows")
        return optimized_df
    
    def save_optimized_data(self, df: pd.DataFrame, name: str, 
                           output_dir: Union[str, Path]) -> Path:
        """
        Save optimized DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            name: Base name for the file
            output_dir: Directory to save the file
            
        Returns:
            Path to saved file
        """
        # Convert output_dir to Path if it's a string
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file path
        file_path = output_dir / f"optimized_{name}.parquet"
        
        # Save to parquet
        self.logger.info(f"Saving optimized {name} data to {file_path}")
        df.to_parquet(file_path, index=False)
        
        return file_path