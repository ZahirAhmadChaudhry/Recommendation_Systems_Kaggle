# Statistical Approaches to Recommender Systems

This module implements several statistical approaches to product recommendations, including frequency-based, recency-based, popularity-based, and hybrid models. The code is organized in a modular structure for clarity, reusability, and maintainability.

## Overview

This implementation provides a comprehensive pipeline for processing e-commerce transaction data, generating features, building various recommendation models, and evaluating their performance. The approach focuses on statistical and rule-based models that do not require complex training procedures or neural networks.

## Key Features

- **Data Optimization**: Memory-efficient data processing for handling large datasets
- **Feature Engineering**: Generation of interaction, product, and temporal features
- **Multiple Recommendation Models**:
  - Frequency-based recommendations
  - Recency-based recommendations
  - Popularity-based recommendations
  - Hybrid recommendations (combining frequency and recency)
  - Customer-segmented recommendations
  - Weighted hybrid model (combining multiple approaches)
- **Comprehensive Evaluation**: Hit rate, precision metrics, and detailed recommendation analysis
- **Visualization Tools**: For model comparison and recommendation pattern analysis

## Project Structure

```
1_Statistical_Approaches/
├── main.py                # Main script for running the pipeline
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── data/              # Data handling modules
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading functionality
│   │   └── data_optimizer.py   # Memory optimization and preprocessing
│   ├── models/            # Recommendation models
│   │   ├── __init__.py
│   │   └── statistical_models.py   # Implementation of various statistical models
│   └── utils/             # Utility modules
│       ├── __init__.py
│       ├── data_validator.py      # Data validation utilities
│       ├── feature_engineering.py # Feature generation utilities
│       └── evaluation.py          # Evaluation metrics and visualization
├── Enhanced_data/         # Directory for optimized data
└── results/               # Directory for results and visualizations
```

## Implementation Details

### Data Processing

The data processing pipeline handles:
- Loading data from various sources (CSV, Parquet)
- Memory optimization to reduce resource usage
- Data validation to ensure integrity
- Preprocessing and feature engineering

### Feature Engineering

Features generated include:
- **Interaction Features**: Patterns in user-product interactions (frequency, recency, etc.)
- **Product Features**: Product characteristics and popularity metrics
- **Temporal Features**: Time-based patterns in user behavior

### Recommendation Models

Multiple recommendation strategies are implemented:

1. **Frequency-based**: Recommends products that a user has purchased most frequently
2. **Recency-based**: Prioritizes recently purchased products
3. **Popularity-based**: Recommends generally popular products across all users
4. **Hybrid Approach**: Combines frequency and recency with configurable weights
5. **Customer Segmentation**: Groups similar customers and generates segment-specific recommendations
6. **Weighted Hybrid**: Combines multiple models using weighted scoring

### Evaluation

The evaluation module provides:
- Hit rate at k metric
- Precision at k metric
- Model comparison tools
- Recommendation analysis utilities
- Visualization of results

## Usage

The pipeline can be run from the command line:

```bash
python main.py --data-dir <path_to_data> --models frequency recency hybrid --save-recommendations --save-plots
```

### Command Line Arguments

```
--data-dir: Directory containing the cleaned data files
--enhanced-data-dir: Directory containing enhanced/optimized data
--output-dir: Directory to save results and visualizations
--optimize-data: Optimize data before processing
--use-optimized: Use previously optimized data
--feature-engineering: Generate and save features
--models: Models to train and evaluate
--k-values: k values for evaluation metrics
--save-recommendations: Save recommendations to file
--save-plots: Save evaluation plots
--analyze-recommendations: Analyze recommendation patterns
--log-level: Logging level
--log-file: Path to log file
```