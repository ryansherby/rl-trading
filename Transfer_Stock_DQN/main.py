#!/usr/bin/env python3
"""
PyTorch implementation for training a stock trading bot
with Deep Q-Learning.

Usage:
    python train.py <train_stock> <val_stock> [--strategy STRATEGY]
                                         [--window-size WINDOW_SIZE]
                                         [--batch-size BATCH_SIZE]
                                         [--episode-count EP_COUNT]
                                         [--model-name MODEL_NAME]
                                         [--pretrained]
                                         [--debug]

Example:
    python train.py data/train.csv data/val.csv --window-size 10 --batch-size 32 --episode-count 100
"""

import argparse
from model import *
from utils import *

import numpy as np
import pandas as pd
from alphas101 import *



# --- Main Training Loop ---

def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False, debug=False):
    """
    Trains the stock trading bot using Deep Q-Learning.
    """
    # Use the window size as the state dimension. (Note: We use window_size+1 in get_state.)
    num_features = 84  # or however many you're using
    state_size = (window_size + 1) * num_features
    agent = Agent(state_size, strategy=strategy, pretrained=pretrained, model_name=model_name)

    # Gets the Adjusted Closing price
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    
    # Calculate an initial offset (could be used as a reference for profitability)
    initial_offset = val_data[1][-1] - val_data[0][-1]

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Stock Trading Bot with Deep Q-Learning using PyTorch")
    parser.add_argument("--train_stock", help="Path to the training stock CSV file")
    parser.add_argument("--val_stock", help="Path to the validation stock CSV file")
    parser.add_argument("--strategy", default="double-dqn", choices=["dqn", "t-dqn", "double-dqn"],
                        help="The learning strategy to use")
    parser.add_argument("--window-size", type=int, default=10,
                        help="The window size for state representation")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size for training")
    parser.add_argument("--episode-count", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--modelname", type=str, default="model_debug", help="Name of the model file")
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained model weights")
    # parser.add_argument("--debug", action="store_true", help="Enable debug level logging")
    args = parser.parse_args()

    # logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    main(args.train_stock, args.val_stock, args.window_size, args.batch_size,
         args.episode_count, strategy=args.strategy, pretrained=args.pretrained, model_name=args.modelname)
