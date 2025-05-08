import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Direction_DQN.alphas101 import *
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, BatchSampler
import torch

def get_state(data, t, window_size):
    """
    Constructs the state from a time-series window for a single stock.
    Returns shape: (1, window_size * feature_dim)
    """
    start = max(0, t - window_size + 1)
    window = data[start:t + 1]
    if len(window) < window_size:
        pad = [np.array(window[0])] * (window_size - len(window))
        window = pad + window
    window = np.array(window).astype(np.float32)
    
    return window.reshape(1, -1)  # Flatten to (1, window_size * feature_dim)





def get_data(stock_file):
    """
    Loads stock CSV, computes alphas + returns, and returns:
      - a list of fixed-length scaled feature vectors (all columns preserved),
      - a parallel list of raw close prices for P&L.
    """
    # Load & compute basic indicators
    df = pd.read_csv(stock_file, parse_dates=['Start','End'])
    
    # Reverse so earliest-first; i.e., t+1 is idx+1
    df = df.loc[::-1].reset_index(drop=True)
    
    df['Average'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Volume'] = df['Volume'].clip(lower=1)
    df['VWAP'] = (
        (df['Average'] * df['Volume']).cumsum() /
        df['Volume'].cumsum()
    )
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['Log Returns'] = np.log(1 + df['Returns'])

    # Compute alphas on the frame
    df_alphas = get_alpha(df)

    df_alphas.fillna(0, inplace=True)
    
    # Convert boolean alphas to integers
    alpha_cols = [c for c in df_alphas.columns if c.startswith('alpha')]

    for col in alpha_cols:
        # Check if column is timestamp-like (large integers or datetime)
        if df_alphas[col].dtype in ['datetime64[ns]', 'object'] or \
           (df_alphas[col].dtype in ['int64', 'float64'] and df_alphas[col].abs().max() > 1e9):
            print(f"Warning: Excluding timestamp-like column {col} in {stock_file}")
            alpha_cols.remove(col)

            
    for c in alpha_cols:
        if df_alphas[c].dtype == 'bool':
            df_alphas[c] = df_alphas[c].astype(int)

    # Build feature DataFrame: alphas + Returns
    df_alphas = df_alphas.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
    
    # Limiting maximum value
    df_alphas[alpha_cols] = df_alphas[alpha_cols].clip(-1e5, 1e5)
    
    
    return df_alphas




def show_train_result(ep, total_ep, train_profit, train_loss):
    """
    Displays training results.
    """

    print(f'Episode {ep}/{total_ep} - Train Position: {train_profit:.4f}  \
                                          Train Loss: {train_loss:.4f}')
    
def show_test_result(test_profit, action_counts):
    """
    Displays training results.
    """

    print(f'Test Rewards: {test_profit:.4f}')
    print(f'Action Counts: {action_counts}')
    
def show_val_result(ep, total_ep, val_profit, initial_offset):
    """
    Displays training results.
    """

    print(f'Episode {ep}/{total_ep} - Val Position: {val_profit:.4f}  \
                                          Initial Offset: {initial_offset:.4f}')
    
def show_eval_result(model_name, profit, initial_offset):
    """
    Displays evaluation results.
    """
    if profit == initial_offset or profit == 0.0:
        print('{}: USELESS\n'.format(model_name))
    else:
        print('{}: {}\n'.format(model_name, format_position(profit)))
        
        

def format_position(price):
    """Formats the profit/loss position as currency"""
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


def format_currency(price):
    """Formats the price value as currency"""
    return '${0:.2f}'.format(abs(price))