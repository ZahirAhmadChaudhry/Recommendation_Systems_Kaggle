# Neural Graph Collaborative Filtering (NGCF)

This folder contains the implementation of Neural Graph Collaborative Filtering (NGCF) for the Carrefour product recommendation Kaggle competition.

## Overview

Neural Graph Collaborative Filtering (NGCF) is a graph-based recommendation algorithm that exploits the user-item interaction graph structure by propagating embeddings on the graph. NGCF captures collaborative signals by modeling higher-order connectivity in the user-item graph, leading to more effective representations for users and items.

The key features of this implementation include:

- Graph-based modeling of user-item interactions
- Message passing and embedding propagation through the interaction graph
- Hard negative sampling for more effective training
- Mixed precision training for performance
- Evaluation using Hit Rate@K metric

## Project Structure

```
NGCF/
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── ngcf_model.py
│   └── utils/
│       ├── __init__.py
│       └── training.py
```

## Key Components

### Data Processing (`data_loader.py`)

The data processing module provides functions for:
- Loading and preprocessing the raw data
- Creating a bipartite graph from user-item interactions
- Initializing embeddings for users and items
- Preparing training and test data with efficient negative sampling

### Model Architecture (`ngcf_model.py`)

The NGCF model implementation includes:
- `EmbeddingPropagationLayer`: Single layer for message passing on the graph
- `GraphEmbeddingPropagation`: Multi-layer graph propagation module
- `NGCF`: Main model class that utilizes graph structure for recommendations
- Bayesian Personalized Ranking (BPR) loss for training

### Training and Evaluation (`training.py`)

The training module provides utilities for:
- Training the model with early stopping
- Evaluating the model using Hit Rate@K
- Generating recommendations for users
- Creating submission files for the competition

### Main Script (`main.py`)

The main script brings everything together with a command-line interface for:
- Training the model
- Evaluating on test data
- Generating recommendations

## Usage

### Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

### Training

To train the NGCF model:
```bash
python src/main.py --mode train --train_path path/to/train_data --products_path path/to/products_data.csv --test_path path/to/test_data.csv --save_model
```

### Evaluation

To evaluate a trained model and generate recommendations:
```bash
python src/main.py --mode evaluate --load_model path/to/model.pth --test_path path/to/test_data.csv
```

### Generating Recommendations

To interactively generate recommendations for specific users:
```bash
python src/main.py --mode recommend --load_model path/to/model.pth
```

## Key Parameters

- `--embedding_dim`: Dimension of user and item embeddings (default: 64)
- `--num_layers`: Number of graph convolutional layers (default: 3)
- `--dropout_ratio`: Dropout rate (default: 0.1)
- `--batch_size`: Batch size for training (default: 1024)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 20)
- `--k`: Number of items to recommend (default: 10)

## References

This implementation is based on the paper:
- Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). Neural graph collaborative filtering. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 165-174).

## License

This project is available under the same license as the overall repository.