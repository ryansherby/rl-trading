import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

# Uncomment the following if you haven't installed stable-baselines3 already.
# !pip install stable-baselines3

from stable_baselines3 import DQN


def compute_sma(data, window=5):
    """Compute simple moving average."""
    return data.rolling(window=window).mean()

def compute_rsi(data, window=14):
    """Compute simplified Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use rolling mean for gains and losses
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-6)  # add small number to avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------------
# Custom Trading Environment
# -----------------------------
class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_cash=10000, transaction_cost_pct=0.001):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        # Ensure our indicators are computed
        if "SMA" not in self.df.columns:
            self.df['SMA'] = compute_sma(self.df['Close'], window=5)
        if "RSI" not in self.df.columns:
            self.df['RSI'] = compute_rsi(self.df['Close'], window=14)
        
        # Remove initial rows with NaN values (from rolling calculations)
        self.df.dropna(inplace=True)
        self.df = self.df.reset_index(drop=True)
        
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: [Current Price, SMA, RSI, Cash, Stock Owned]
        # Prices and indicators are normalized here for simplicity.
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         100,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Initialize state variables
        self.reset()
        
    def _get_obs(self):
        # Get features at current step
        current_data = self.df.iloc[self.current_step]
        obs = np.array([
            current_data['Close'],
            current_data['SMA'],
            current_data['RSI'],
            self.cash,
            self.stock_owned
        ], dtype=np.float32)
        return obs
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.stock_owned = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        # Record net worth history for rendering purposes
        self.net_worths = [self.initial_cash]
        return self._get_obs()
    
    def step(self, action):
        done = False
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Initialize reward to zero
        reward = 0
        
        # Print debug info if needed:
        # print(f"Step {self.current_step}, Action: {action}, Price: {current_price}")
        
        # Execute trade
        if action == 1:  # Buy
            # Buy only if not already in a position; for simplicity, we go all-in.
            if self.stock_owned == 0:
                # Determine number of shares to buy with all available cash
                shares = self.cash // current_price
                if shares > 0:
                    cost = shares * current_price * (1 + self.transaction_cost_pct)
                    self.cash -= cost
                    self.stock_owned += shares
                    self.total_shares_bought += shares
        elif action == 2:  # Sell
            if self.stock_owned > 0:
                # Sell all shares
                proceeds = self.stock_owned * current_price * (1 - self.transaction_cost_pct)
                self.cash += proceeds
                self.total_shares_sold += self.stock_owned
                profit = proceeds - (self.stock_owned * current_price)
                self.total_profit += profit
                self.stock_owned = 0
        
        # Compute net worth
        net_worth = self.cash + self.stock_owned * current_price
        self.net_worths.append(net_worth)
        
        # Reward is defined as the change in net worth
        if len(self.net_worths) >= 2:
            reward = self.net_worths[-1] - self.net_worths[-2]
        
        # Move to the next time step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}
    
    def render(self, mode='human', close=False):
        plt.figure(figsize=(10, 5))
        plt.plot(self.net_worths)
        plt.xlabel("Timestep")
        plt.ylabel("Net Worth")
        plt.title("Portfolio Net Worth Over Time")
        plt.show()

# -----------------------------
# Main Functionality: Training & Inference
# -----------------------------
if __name__ == "__main__":
    # Try to load data from CSV file, or create synthetic data if not available
    DATA_FILE = "stock_data.csv"
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Ensure the CSV has a 'Close' column. You may adjust column names as needed.
    else:
        # Generate synthetic stock price data (a random walk) for demonstration
        np.random.seed(42)
        dates = pd.date_range(datetime.datetime.today() - datetime.timedelta(1000), periods=1000)
        price = np.cumsum(np.random.randn(1000)) + 100  # random walk around 100
        df = pd.DataFrame({'Date': dates, 'Close': price})
    
    # For safety, sort by date if available
    if "Date" in df.columns:
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # Create the environment
    env = StockTradingEnv(df)
    
    # Create a DQN agent using stable-baselines3
    model = DQN("MlpPolicy", env, verbose=1)
    
    # Train the agent (adjust timesteps as needed)
    model.learn(total_timesteps=10000)
    
    # After training, test the agent on the environment
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Use the model to predict an action given the current observation
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    print("Total Reward after test episode:", total_reward)
    env.render()
    
    # -----------------------------
    # Inference example:
    # At inference time, you might have a function that, given a stock name, loads the corresponding data,
    # creates an environment, and then uses the trained model to output a recommendation.
    # -----------------------------
    def infer_action(stock_ticker, model, data_path="stock_data.csv"):
        """
        Inference function:
          Given a stock ticker, load its data, initialize the environment,
          and return the recommended action (buy=1, sell=2, hold=0).
          This example assumes the CSV file exists and contains data for the stock.
        """
        # For demonstration, we assume the CSV data is for the given stock. In real application,
        # you might query a database or an API.
        if os.path.exists(data_path):
            df_stock = pd.read_csv(data_path)
            if "Date" in df_stock.columns:
                df_stock.sort_values("Date", inplace=True)
                df_stock.reset_index(drop=True, inplace=True)
            env_stock = StockTradingEnv(df_stock)
            obs = env_stock.reset()
            action, _ = model.predict(obs, deterministic=True)
            action_meaning = {0: "Hold", 1: "Buy", 2: "Sell"}
            print(f"Action for stock {stock_ticker}: {action_meaning.get(action, 'Unknown')}")
            return action
        else:
            print("No data found for stock:", stock_ticker)
            return None
    
    # Example inference: for a stock called "FAKE"
    infer_action("FAKE", model)
