import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from alphas101 import *
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import torch

def get_state(data, t, window_size, raw_prices, agent):
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
    return np.expand_dims(window, axis=0) 
# return window.reshape(1, -1)  # Flatten to (1, window_size * feature_dim)

# """
# def get_state(data, t, window_size, raw_prices, agent):
#     """
#     Constructs the state from a time-series window for a single stock.
#     Returns shape: (1, window_size * feature_dim)
#     """
#     start = max(0, t - window_size + 1)
#     window = data[start:t + 1]
#     if len(window) < window_size:
#         pad = [np.array(window[0])] * (window_size - len(window))
#         window = pad + window
#     window = np.array(window).astype(np.float32)
#     # 2. Add market regime indicators
#     if t >= 20:  # Need some history
#         # Calculate recent returns for regime detection
#         recent_prices = raw_prices[max(0, t-20):t+1]
        
#         # Trend indicator (positive: uptrend, negative: downtrend)
#         short_ma = np.mean(recent_prices[-5:])
#         long_ma = np.mean(recent_prices)
#         trend = (short_ma / long_ma) - 1.0
        
#         # Volatility indicator
#         returns = [(recent_prices[i+1] - recent_prices[i])/recent_prices[i] 
#                   for i in range(len(recent_prices)-1)]
#         volatility = np.std(returns) if len(returns) > 0 else 0.01
        
#         # Mean reversion indicator (z-score of current price)
#         mean_price = np.mean(recent_prices)
#         std_price = np.std(recent_prices) if len(recent_prices) > 1 else 0.01
#         z_score = (raw_prices[t] - mean_price) / std_price if std_price > 0 else 0
        
#         regime_features = np.array([trend, volatility, z_score])
#     else:
#         # Default regime features if not enough history
#         regime_features = np.array([0.0, 0.01, 0.0])
    
#     # 3. Add position information
#     # Current inventory size and average cost basis
#     inv_size = len(agent.inventory)
#     avg_cost = np.mean(agent.inventory) if inv_size > 0 else 0
#     unrealized_pl = (raw_prices[t] - avg_cost) * inv_size if inv_size > 0 else 0
    
#     # Normalized position features
#     max_position = 5  # Assuming max reasonable position size
#     norm_size = inv_size / max_position  # Normalized inventory size
#     norm_pl = unrealized_pl / (avg_cost * max_position) if avg_cost > 0 else 0  # Normalized P&L
    
#     position_features = np.array([norm_size, norm_pl])
    
#     # 5. Combine all features
#     # Reshape window to be 2D
#     window_flat = np.expand_dims(window, axis=0) 
    
#     # Combine with additional features
#     additional_features = np.concatenate([
#         regime_features, 
#         position_features,
#     ])

#     static_feats = additional_features[np.newaxis, np.newaxis, :] 
#     # repeat across all 61 timesteps → (1, 5, 61)
    
#     static_feats = np.tile(static_feats, (1, window_size, 1))

#     enhanced_input = np.concatenate([window_flat, static_feats], axis=2)
    
#     return enhanced_input
# """
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
            # print(f"Warning: Excluding timestamp-like column {col} in {stock_file}")
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
    # print(df_scaled.isna().sum().sum())
    # 9) Return parallel lists
    return df_scaled.values.tolist(), raw_prices

def show_train_result(ep, total_ep, train_profit, train_loss, time):
    """
    Displays training results.
    """

    print(f'Episode {ep}/{total_ep} - Train Position: {train_profit:.4f}  \
                                          Train Loss: {train_loss:.4f}  TimeTaken {time:.4f}')
    
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


# Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# class PrioritizedReplayBuffer:
#     """
#     Prioritized Experience Replay Buffer
#     Stores experiences and samples based on TD error priority
#     """
#     def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
#         self.capacity = capacity
#         self.memory = []
#         self.priorities = np.zeros((capacity,), dtype=np.float32)
#         self.position = 0
#         self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
#         self.beta = beta  # Importance sampling correction (starts low, anneals to 1)
#         self.beta_increment = beta_increment  # How much to increase beta each sampling
#         self.max_priority = 1.0
        
#     def push(self, state, action, reward, next_state, done):
#         """Add a new experience to memory with max priority"""
#         experience = Experience(state, action, reward, next_state, done)
        
#         if len(self.memory) < self.capacity:
#             self.memory.append(experience)
#         else:
#             self.memory[self.position] = experience
            
#         # New experiences get max priority to ensure exploration
#         self.priorities[self.position] = self.max_priority
#         self.position = (self.position + 1) % self.capacity
        
#     def sample(self, batch_size):
#         """Sample a batch of experiences based on priority"""
#         if len(self.memory) < batch_size:
#             return [], [], []  # Not enough samples
            
#         # Get sampling probabilities from priorities
#         priorities = self.priorities[:len(self.memory)]
#         probabilities = priorities ** self.alpha
#         probabilities /= probabilities.sum()
        
#         # Sample indices based on probabilities
#         indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
#         # Get experiences for these indices
#         experiences = [self.memory[i] for i in indices]
        
#         # Calculate importance sampling weights
#         weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
#         weights /= weights.max()  # Normalize to stabilize updates
#         weights = torch.tensor(weights, dtype=torch.float32)
        
#         # Increase beta for next sampling
#         self.beta = min(1.0, self.beta + self.beta_increment)
        
#         return experiences, indices, weights
        
#     def update_priorities(self, indices, td_errors):
#         """Update priorities based on new TD errors"""
#         for i, td_error in zip(indices, td_errors):
#             # Add a small constant to avoid zero priority
#             priority = abs(td_error) + 1e-5
#             self.priorities[i] = priority
#             self.max_priority = max(self.max_priority, priority)
            
#     def __len__(self):
#         return len(self.memory)
