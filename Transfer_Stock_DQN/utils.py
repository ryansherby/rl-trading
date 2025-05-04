import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from alphas101 import *
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

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

def format_position(price):
    """Formats the profit/loss position as currency"""
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


def format_currency(price):
    """Formats the price value as currency"""
    return '${0:.2f}'.format(abs(price))


def get_stock_data(stock_file):
    """
    Loads stock CSV, computes alphas + returns, and returns:
      - a list of fixed-length scaled feature vectors (all columns preserved),
      - a parallel list of raw close prices for P&L.
    
    Scaling is (x - Q1) / IQR for non-constant columns, with constant columns
    centered (x - median) without scaling to avoid division by zero.
    """
    # 1) Load & compute basic indicators
    df = pd.read_csv(stock_file, parse_dates=['Start'])
    df['Average'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Volume'] = df['Volume'].clip(lower=1)
    df['VWAP'] = (
        (df['Average'] * df['Volume'])[::-1].cumsum() /
        df['Volume'][::-1].cumsum()
    )[::-1]
    df['Returns'] = df['Close'].pct_change().fillna(0)

    # 2) Reverse so earliest-first
    df = df.iloc[::-1].reset_index(drop=True)

    # 3) Save raw Close prices for profit math
    raw_prices = df['Close'].tolist()
    
    # 4) Compute alphas on the reversed frame
    df_alphas = get_alpha(df).iloc[::-1].reset_index(drop=True)

    df_alphas.fillna(0, inplace=True)

    
    
    # 5) Convert boolean alphas to integers
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

    # 6) Build feature DataFrame: alphas + Returns
    feature_cols = alpha_cols + ['Returns']
    df_feats = df_alphas[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 7) Compute IQR-based scaling for non-constant columns
    Q1 = df_feats.quantile(0.25)
    Q3 = df_feats.quantile(0.75)
    IQR = Q3 - Q1
    median = df_feats.median()

    # Initialize scaled DataFrame
    df_scaled = df_feats.copy()

    # Apply scaling: (x - Q1) / IQR for non-constant columns, (x - median) for constant
    for col in feature_cols:
        if IQR[col] > 0:  # Non-constant column
            df_scaled[col] = (df_feats[col] - Q1[col]) / IQR[col]
        else:  # Constant column
            df_scaled[col] = df_feats[col] - median[col]  # Center at zero

    # 8) Replace any remaining NaNs or infinities
    df_scaled = df_scaled.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(df_scaled.isna().sum().sum())
    # 9) Return parallel lists
    return df_scaled.values.tolist(), raw_prices

def show_train_result(ep, total_ep, train_profit, train_loss):
    """
    Displays training results.
    """

    print(f'Episode {ep}/{total_ep} - Train Position: {train_profit:.4f}  \
                                          Train Loss: {train_loss:.4f}')
    
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