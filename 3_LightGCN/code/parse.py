'''
Created on Mar 1, 2020
Modified for Carrefour Dataset Implementation
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN for Carrefour")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="batch size for training")
    parser.add_argument('--recdim', type=int, default=64,
                        help="embedding size")
    parser.add_argument('--layer', type=int, default=3,
                        help="number of GCN layers")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="weight decay for l2 regularization")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="dropout rate")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="fold number for sparse adjacency matrix")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="batch size for testing")
    parser.add_argument('--dataset', type=str, default='carrefour',
                        help="available datasets: [carrefour]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save model checkpoints")
    parser.add_argument('--topks', type=str, default="[10]",
                        help="top k for hit rate calculation")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard logging")
    parser.add_argument('--comment', type=str, default="carrefour_lgn")
    parser.add_argument('--load', type=int, default=0,
                        help="whether to load pretrained model")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument('--multicore', type=int, default=0,
                        help="whether to use multiprocessing in testing")
    parser.add_argument('--seed', type=int, default=2024,
                        help="random seed")
    parser.add_argument('--model', type=str, default='lgn',
                        help="model type, only lgn supported")
    return parser.parse_args()
