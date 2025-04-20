"""
Data validation module for ensuring data integrity throughout the processing pipeline.
Implements comprehensive schema validation and data quality checks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional


class DataValidator:
    """
    Comprehensive data validation for train, test and products datasets.
    Ensures data integrity throughout the optimization process.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize DataValidator with schema definitions and logging configuration.
        
        Args:
            log_level: Logging level (default: INFO)
        """
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define schemas for each dataset
        self.train_schema = {
            'date': {'dtype': ['object', 'datetime64[ns]'], 'nulls': False, 'unique_min': 365},
            'transaction_id': {'dtype': ['int64', 'int32'], 'nulls': False, 'min': 0},
            'customer_id': {'dtype': ['int64', 'int32'], 'nulls': False, 'unique_exact': 100000},
            'product_id': {'dtype': ['int64', 'int32'], 'nulls': False},
            'has_loyality_card': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]},
            'store_id': {'dtype': ['int64', 'int16'], 'nulls': False, 'min': 0},
            'is_promo': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]},
            'quantity': {'dtype': ['float64', 'float32'], 'nulls': False, 'min': 0},
            'format': {'dtype': ['object', 'category'], 'nulls': False, 'unique_exact': 3},
            'order_channel': {'dtype': ['object', 'category'], 'nulls': False, 'unique_exact': 10}
        }
        
        self.products_schema = {
            'product_id': {'dtype': ['int64', 'int32'], 'nulls': False, 'min': 0},
            'department_key': {'dtype': ['object', 'category'], 'nulls': False},
            'class_key': {'dtype': ['object', 'category'], 'nulls': False},
            'subclass_key': {'dtype': ['object', 'category'], 'nulls': False},
            'brand_key': {'dtype': ['object', 'category'], 'nulls': True},  # 63 missing values
            'sector': {'dtype': ['object', 'category'], 'nulls': False},
            'bio': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]},
            'fresh': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]},
            'frozen': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]},
            'national_brand': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]},
            'carrefour_brand': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]},
            'first_price_brand': {'dtype': ['int64', 'int8'], 'nulls': False, 'values': [0, 1]}
        }
        
        self.test_schema = {
            'transaction_id': {'dtype': ['int64', 'int32'], 'nulls': False, 'min': 0},
            'customer_id': {'dtype': ['int64', 'int32'], 'nulls': False, 'min': 0},
            'product_id': {'dtype': ['int64', 'int32'], 'nulls': False, 'min': 0}
        }
        
        # Metrics storage
        self.before_metrics: Dict[str, Dict[str, Any]] = {}
        self.after_metrics: Dict[str, Dict[str, Any]] = {}
    
    def capture_metrics(self, df: pd.DataFrame, name: str, stage: str) -> dict:
        """
        Capture comprehensive metrics about the dataframe.
        
        Args:
            df: DataFrame to analyze
            name: Dataset name (train, products, test)
            stage: Processing stage ('before' or 'after')
            
        Returns:
            Dictionary with captured metrics
        """
        self.logger.info(f"Capturing {stage} metrics for {name} dataset...")
        
        metrics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns},
            'numeric_stats': {
                col: {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean() if df[col].dtype in ['int64', 'int32', 'float64', 'float32'] else None
                }
                for col in df.select_dtypes(include=['number']).columns
            }
        }
        
        if stage == 'before':
            self.before_metrics[name] = metrics
        else:
            self.after_metrics[name] = metrics
            
        return metrics

    def validate_schema(self, df: pd.DataFrame, schema: dict, name: str) -> None:
        """
        Validate dataframe against schema requirements.
        
        Args:
            df: DataFrame to validate
            schema: Schema definition
            name: Dataset name
            
        Raises:
            AssertionError: If validation fails
        """
        self.logger.info(f"Validating schema for {name} dataset...")
        
        for col, rules in schema.items():
            # Check column existence
            assert col in df.columns, f"{name}: Missing column {col}"
            
            # Check dtype
            assert df[col].dtype.name in rules['dtype'], (
                f"{name}: Column {col} has incorrect dtype {df[col].dtype.name}"
            )
            
            # Check for nulls
            if not rules.get('nulls', True):
                assert df[col].isnull().sum() == 0, f"{name}: Column {col} contains null values"
            
            # Check specific values if defined
            if 'values' in rules:
                assert df[col].isin(rules['values']).all(), (
                    f"{name}: Column {col} contains values outside allowed set {rules['values']}"
                )
            
            # Check minimum value if defined
            if 'min' in rules:
                assert df[col].min() >= rules['min'], (
                    f"{name}: Column {col} contains values below minimum {rules['min']}"
                )
            
            # Check exact unique count if defined
            if 'unique_exact' in rules:
                assert df[col].nunique() == rules['unique_exact'], (
                    f"{name}: Column {col} has {df[col].nunique()} unique values, "
                    f"expected {rules['unique_exact']}"
                )
            
            # Check minimum unique count if defined
            if 'unique_min' in rules:
                assert df[col].nunique() >= rules['unique_min'], (
                    f"{name}: Column {col} has fewer than {rules['unique_min']} unique values"
                )
                
        self.logger.info(f"Schema validation successful for {name} dataset")

    def validate_optimization(self, name: str) -> None:
        """
        Validate that optimization didn't corrupt the data.
        
        Args:
            name: Dataset name
            
        Raises:
            AssertionError: If validation fails
        """
        self.logger.info(f"Validating optimization for {name} dataset...")
        
        before = self.before_metrics[name]
        after = self.after_metrics[name]
        
        # Check row count preservation
        assert before['row_count'] == after['row_count'], (
            f"{name}: Row count changed during optimization"
        )
        
        # Define expected columns after optimization
        if name == 'products':
            expected_columns = {
                'product_id', 'department_key', 'class_key', 'subclass_key',
                'brand_key', 'sector', 'bio', 'fresh', 'frozen', 'national_brand',
                'carrefour_brand', 'first_price_brand'
            }
            # Check that we have exactly the columns we expect
            assert set(after['dtypes'].keys()) == expected_columns, (
                f"{name}: Unexpected columns after optimization. "
                f"Expected: {expected_columns}, "
                f"Got: {set(after['dtypes'].keys())}"
            )
        else:
            # For train dataset, all columns should be preserved
            assert set(before['dtypes'].keys()) == set(after['dtypes'].keys()), (
                f"{name}: Columns were modified during optimization"
            )
        
        # Check unique value preservation for remaining columns
        for col in after['unique_counts']:
            if col in before['unique_counts']:
                assert before['unique_counts'][col] == after['unique_counts'][col], (
                    f"{name}: Unique values changed for column {col}"
                )
        
        # Check numeric ranges preservation for remaining columns
        for col in after['numeric_stats']:
            if col in before['numeric_stats']:
                assert np.isclose(before['numeric_stats'][col]['min'], 
                                after['numeric_stats'][col]['min'], rtol=1e-5), (
                    f"{name}: Minimum value changed for column {col}"
                )
                assert np.isclose(before['numeric_stats'][col]['max'], 
                                after['numeric_stats'][col]['max'], rtol=1e-5), (
                    f"{name}: Maximum value changed for column {col}"
                )
                
        self.logger.info(f"Optimization validation successful for {name} dataset")

    def verify_memory_optimization(self, name: str, min_reduction: float = 0.3) -> None:
        """
        Verify that memory usage was actually reduced.
        
        Args:
            name: Dataset name
            min_reduction: Minimum expected memory reduction ratio (0.3 = 30%)
            
        Raises:
            AssertionError: If memory reduction is less than expected
        """
        self.logger.info(f"Verifying memory optimization for {name} dataset...")
        
        before = self.before_metrics[name]['memory_usage']
        after = self.after_metrics[name]['memory_usage']
        reduction = (before - after) / before
        
        assert reduction >= min_reduction, (
            f"{name}: Memory reduction of {reduction:.2%} is less than "
            f"expected minimum of {min_reduction:.2%}"
        )
        
        self.logger.info(f"Memory optimization verification successful: {reduction:.2%} reduction")

    def validate_dataset(self, df: pd.DataFrame, name: str, stage: str) -> None:
        """
        Complete validation of a dataset.
        
        Args:
            df: DataFrame to validate
            name: Dataset name ('train', 'products', or 'test')
            stage: Processing stage ('before' or 'after')
            
        Raises:
            ValueError: If invalid dataset name is provided
            AssertionError: If validation fails
        """
        self.logger.info(f"Validating {name} dataset ({stage})...")
        
        # Capture metrics
        metrics = self.capture_metrics(df, name, stage)
        
        # Validate against appropriate schema
        if name == 'train':
            self.validate_schema(df, self.train_schema, name)
        elif name == 'products':
            self.validate_schema(df, self.products_schema, name)
        elif name == 'test':
            self.validate_schema(df, self.test_schema, name)
        else:
            raise ValueError(f"Invalid dataset name: {name}")
        
        # Validate optimization results if in 'after' stage
        if stage == 'after':
            self.validate_optimization(name)
            self.verify_memory_optimization(name)
            
        self.logger.info(f"Dataset validation complete for {name} dataset ({stage})")
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Returns:
            Dictionary with validation summary
        """
        summary = {}
        
        for dataset in self.after_metrics:
            before = self.before_metrics[dataset]
            after = self.after_metrics[dataset]
            
            memory_before = before['memory_usage']
            memory_after = after['memory_usage']
            reduction = (memory_before - memory_after) / memory_before
            
            summary[dataset] = {
                'row_count': after['row_count'],
                'column_count': after['column_count'],
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_reduction': reduction,
                'column_dtypes': after['dtypes']
            }
            
        return summary