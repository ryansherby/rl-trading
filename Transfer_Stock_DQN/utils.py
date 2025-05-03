import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from alphas101 import *

# def get_state(data, t, window_size):
#     """
#     Constructs the state representation from the data.
#     Uses a sliding window ending at index t (padded if necessary).

#     Args:
#         data (list or np.array): The raw price data.
#         t (int): Current time index.
#         window_size (int): The number of past data points to include.
        
#     Returns:
#         np.array: A numpy array of shape (1, window_size).
#     """
#     if t < window_size:
#         # Pad with the first element if there are not enough previous points.
#         block = [data[0]] * (window_size - t) + data[0:t]
#     else:
#         block = data[t - window_size:t]
#     return np.array([block])

def get_state(data, t, window_size):
    """
    Constructs the state from a time-series window.
    Each time step is a vector of features.
    Returns shape: (1, window_size, feature_dim)
    """
    start = max(0, t - window_size + 1)
    window = data[start:t + 1]

    if len(window) < window_size:
        pad = [window[0]] * (window_size - len(window))
        window = pad + window

    return np.expand_dims(np.array(window), axis=0).astype(np.float32)


def format_position(price):
    """Formats the profit/loss position as currency"""
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


def format_currency(price):
    """Formats the price value as currency"""
    return '${0:.2f}'.format(abs(price))


def get_stock_data(stock_file):
    """
    Reads stock data from CSV file, computes VWAP and Returns, applies alphas, 
    scales them, and returns a list of feature vectors.
    """
    # Load data
    df = pd.read_csv(stock_file, parse_dates=['Start'])

    # Clean volume and compute VWAP
    df['Average'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Volume'] = df['Volume'].apply(lambda x: 1 if x < 1 else x)
    df['VWAP'] = ((df['Average'] * df['Volume']).iloc[::-1].cumsum() / df['Volume'].iloc[::-1].cumsum()).iloc[::-1]

    # Calculate returns (reverse pct_change for RL style training)
    df['Returns'] = df['Close'].pct_change().fillna(0)

    # Reverse the DataFrame so it's oldest → newest
    df = df[::-1].reset_index(drop=True)

    # Compute alphas on reversed data and flip it back
    df_with_alphas = get_alpha(df)[::-1].reset_index(drop=True)

    # Clean NaNs
    df_with_alphas.fillna(0, inplace=True)

    # Select only alpha columns + returns
    alpha_columns = [col for col in df_with_alphas.columns if col.startswith('alpha')]

    # Convert booleans to integers
    df_with_alphas[alpha_columns] = df_with_alphas[alpha_columns].apply(
        lambda col: col.astype(int) if col.dtype == bool else col)

    # Normalize alphas
    scaler = StandardScaler()
    df_with_alphas[alpha_columns] = scaler.fit_transform(df_with_alphas[alpha_columns])

    df_with_alphas = df_with_alphas[alpha_columns + ['Close']].fillna(0)
    
    # Add 'Returns' column to the list of features
    features = df_with_alphas.values.tolist()

    return features

def show_train_result(result, val_position, initial_offset):
    """
    Displays training results.
    """
    if val_position == initial_offset or val_position == 0.0:
        print('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        print('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3]))


def show_eval_result(model_name, profit, initial_offset):
    """
    Displays evaluation results.
    """
    if profit == initial_offset or profit == 0.0:
        print('{}: USELESS\n'.format(model_name))
    else:
        print('{}: {}\n'.format(model_name, format_position(profit)))

