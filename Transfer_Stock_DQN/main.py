#!/usr/bin/env python3
"""
PyTorch implementation for training a stock trading bot
with Deep Q-Learning.

Example:
    python train.py data/train.csv data/val.csv --window-size 10 --batch-size 32 --episode-count 100
"""

import argparse
from model import *
from utils import *
import glob


import numpy as np
import pandas as pd
from alphas101 import *
pd.options.mode.chained_assignment = None
# --- Main Training Loop ---

def main(train_dir, val_dir, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False, debug=False):
    """
    Trains the stock trading bot using Deep Q-Learning.
    """
    # Use the window size as the state dimension. (Note: We use window_size+1 in get_state.)
    num_features = 82  # alphas
    state_size = (window_size + 1) * num_features
    agent = Agent(state_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    print(train_dir + "*.csv")

    train_stock_files = glob.glob(train_dir + "*.csv")[:10]
    val_stock_files = glob.glob(val_dir + "*.csv")[:2]
    if  len(train_stock_files)==0 or  len(val_stock_files)==0:
        raise ValueError("No CSV files found in train or val directories")

    def safe_get(stock_file):
        try:
            return get_stock_data(stock_file)
        except Exception as e:
            print(f"Skipping {stock_file}: {e}")
            return None

    print(f"Training on {len(train_stock_files)} train stocks")
    print(f"Validating on {len(val_stock_files)} val stocks")

    for episode in range(1, ep_count + 1):
        np.random.shuffle(train_stock_files)

        
        train_total_profit = 0
        train_total_loss = 0
        train_profits = []

        #(episode, ep_count, total_profit, avg_loss_value)
        for train_data_file in train_stock_files:
            print(f"Train Run on {train_data_file}")
            train_data, raw_prices = safe_get(train_data_file)
            
            train_result = train_model(agent, episode, train_data, raw_prices, ep_count=ep_count,
                                    batch_size=batch_size, window_size=window_size)
            train_total_profit += train_result[2]
            train_profits.append(train_result[2])
            train_total_loss += train_result[3]
            show_train_result(episode, ep_count, train_result[2], train_result[3])
            

        train_avg_profit = train_total_profit / len(train_stock_files)
        train_avg_loss = train_total_loss / len(train_stock_files)
        
        val_total_profit = 0
        val_profits = []
        initial_offsets = []
        for val_data_file in val_stock_files:
            print(f"Train Run on {val_data_file}")
            val_data, raw_prices = safe_get(val_data_file)
            initial_offset = val_data[1][-1] - val_data[0][-1]
            val_profit, _ = evaluate_model(agent, val_data, raw_prices, window_size, debug)
            val_total_profit += val_profit
            val_profits.append(val_profit)
            show_val_result(episode, ep_count, train_result[2], initial_offset)

        val_avg_profit = val_total_profit / len(val_stock_files)
        

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Stock Trading Bot with Deep Q-Learning using PyTorch")
    parser.add_argument("--train_dir", help="Path to the training stock CSV file")
    parser.add_argument("--val_dir", help="Path to the validation stock CSV file")
    parser.add_argument("--strategy", default="double-dqn", choices=["dqn", "t-dqn", "double-dqn"],
                        help="The learning strategy to use")
    parser.add_argument("--window-size", type=int, default=30,
                        help="The window size for state representation")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for training")
    parser.add_argument("--episode-count", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--modelname", type=str, default="model_debug", help="Name of the model file")
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained model weights")
    # parser.add_argument("--debug", action="store_true", help="Enable debug level logging")
    args = parser.parse_args()
    # logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    main(args.train_dir, args.val_dir, args.window_size, args.batch_size,
         args.episode_count, strategy=args.strategy, pretrained=args.pretrained, model_name=args.modelname)
