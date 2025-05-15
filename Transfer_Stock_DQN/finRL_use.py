import os
import pandas as pd
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from collections import OrderedDict
import os
import pandas as pd
import sys
print(sys.path)
import finrl
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import sys
from finrl.config import INDICATORS
import torch
from stable_baselines3 import SAC
import numpy as np
import time
# sys.path.append('/insomnia001/depts/free/users/ik2592/finrl')

from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import os
import pandas as pd
import matplotlib.pyplot as plt

tickers = [
    "LVMUY","ACGBY",
    "LLY","TSM","CRM","NVO","NFLX","TM","TCEHY",
    "ORCL","WMT","SAP","TSLA","NSRGY","GOOGL","META","BAC","CSCO", "WFC", "AMZN","KO","RHHBY","COST",
    "LVMUY","ACGBY",
    "LLY","TSM","CRM","NVO","NFLX","TM","TCEHY",
    "ORCL","WMT","SAP","TSLA","NSRGY","GOOGL","META","BAC","CSCO", "WFC", "AMZN","KO","RHHBY","COST"
    ,"NVO","NFLX","TM","TCEHY","LLY","TSM","CRM","NVO","NFLX","TM","TCEHY",
    "ORCL","WMT","SAP","TSLA","NSRGY","GOOGL","META","BAC","CSCO", "WFC", "AMZN","KO","RHHBY","COST"
    ,"NVO","NFLX","TM","TCEHY"
]

def flatten_dict(nested, parent_key: str = "", sep: str = "."):
    """
    Turn e.g. 
      { "a": {0: TensorA, 1: TensorB}, "b": [TensorC, {"x": TensorD}] }
    into
      { "a.0": TensorA, "a.1": TensorB,
        "b.0": TensorC, "b.1.x": TensorD }
    """
    flat = {}
    if isinstance(nested, dict):
        for k, v in nested.items():
            key = parent_key + sep + str(k) if parent_key else str(k)
            if isinstance(v, (dict, list, tuple)):
                flat.update(flatten_dict(v, key, sep=sep))
            else:
                flat[key] = v
    elif isinstance(nested, (list, tuple)):
        for idx, v in enumerate(nested):
            key = parent_key + sep + str(idx)
            if isinstance(v, (dict, list, tuple)):
                flat.update(flatten_dict(v, key, sep=sep))
            else:
                flat[key] = v
    else:
        # we shouldn't hit this at top level, but just in case:
        flat[parent_key] = nested
    return flat
# … (your other imports and globals) …
META_LR = 0.02   # how fast we move the meta-weights toward each task
INNER_TIMESTEPS = 40000    # how long we fine-tune on each ticker

all_returns = {}
metrics = []
def update_params(target_params, adapted_params, lr=META_LR):
        """Recursively update parameters with Reptile algorithm"""
        if isinstance(target_params, dict):
            for k in target_params.keys():
                if k in adapted_params:
                    target_params[k] = update_params(target_params[k], adapted_params[k], lr)
            return target_params
        elif isinstance(target_params, list):
            for i in range(len(target_params)):
                if i < len(adapted_params):
                    target_params[i] = update_params(target_params[i], adapted_params[i], lr)
            return target_params
        else:
            # For individual parameters (tensors or scalars)
            if isinstance(target_params, torch.Tensor):
                # For tensors, use PyTorch methods
                return target_params + lr * (adapted_params - target_params)
            elif isinstance(target_params, (int, float, bool, np.number)):
                # For scalar values
                return target_params + lr * (adapted_params - target_params)
            else:
                # For other types, just return the adapted value
                return adapted_params
# Hyper-parameters for Reptile
           
ENT= 0.7
LR = 1e-3
DATA_DIR    = "/insomnia001/home/ik2592/slurm_outs/RL2/data/train"
OUTPUT_ROOT = "/insomnia001/depts/free/users/ik2592/outputs"

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=False,
    user_defined_feature=False
)

def print_weights(model, label=""):
    print(f"\n--- {label} Weights ---")
    for name, param in model.policy.named_parameters():
        if param.requires_grad:
            print(f"{name}:")
            print(param.data)

# 1) Instantiate once on the first ticker
agent   = None
model_ppo = None

# set up logger
os.makedirs(OUTPUT_ROOT, exist_ok=True)
logger = configure(OUTPUT_ROOT, ["stdout", "csv", "tensorboard"])

for idx, tic in enumerate(tickers):
    print(f"\n--- Task {idx+1}/{len(tickers)}: {tic} ---")

    # a) load & preprocess exactly as before …
    df = pd.read_csv(f"{DATA_DIR}/{tic}_market_data.csv").rename(columns={
        "Start":     "date",
        "Adj Close": "adjcp",
        "Close":     "close",
        "High":      "high",
        "Low":       "low",
        "Volume":    "volume",
        "Open":      "open",
    })
    df["tic"] = tic
    proc = fe.preprocess_data(df)
    train = data_split(proc, "2009-01-05", "2022-12-20")
    
    stock_dim   = len(train.tic.unique())      # will be 1
    state_space = 1 + 2*stock_dim + len(INDICATORS)*stock_dim
    env_kwargs = {
        "hmax":               100,
        "initial_amount":     1_000_000,
        "buy_cost_pct":       [0.002]*stock_dim,
        "sell_cost_pct":      [0.002]*stock_dim,
        "num_stock_shares":   [0]*stock_dim,
        "stock_dim":          stock_dim,
        "action_space":       stock_dim,
        "state_space":        state_space,
        "tech_indicator_list": INDICATORS,
        "reward_scaling":      1e-3,
    }
    # b) build env_train (same env_kwargs as before) …
    env_train, _ = StockTradingEnv(df=train, **env_kwargs).get_sb_env()
    print(f"Training Data Len {len(train)}")
    if idx == 0:
        # initialize meta-model on first task
        agent    = DRLAgent(env=env_train)
        
        policy_kwargs = dict(
            net_arch=[256, 512, 256],
            activation_fn=torch.nn.ReLU,
            optimizer_class=torch.optim.Adam,
        )
        model_ppo =  SAC(
            "MlpPolicy",
            env_train,
            verbose=1,
            # batch_size= 32,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-3,
            buffer_size=100_000,
            tau=0.005,
            learning_starts=1000,
            ent_coef=0.8,
            target_entropy= 0.01,
            gradient_steps=1,
            train_freq=1,
            seed=32  # You can set a seed for reproducibility
        )
        model_ppo.set_logger(logger)
    else:
        # For subsequent tasks:
        # 1. Store current meta-learned parameters
        ENT = ENT*0.94
        LR = LR*0.94
        current_params = model_ppo.get_parameters()
        policy_kwargs = dict(
            net_arch=[256, 512, 256],
            activation_fn=torch.nn.ReLU,
            optimizer_class=torch.optim.AdamW,
        )
        # 2. Create completely new model instance (with fresh empty buffer)
        model_ppo = SAC(
            "MlpPolicy",
            env_train,
            verbose=1,
            # batch_size= 32,
            policy_kwargs=policy_kwargs,
            learning_rate=LR,
            buffer_size=100_000,
            tau=0.005,
            learning_starts=1000,
            ent_coef=ENT,
            target_entropy= 0.01,
            gradient_steps=1,
            train_freq=1,
            seed=32   # You can set a seed for reproducibility
        )
        model_ppo.set_logger(logger)
        model_ppo.set_env(env_train)
        
        # 3. Load meta-learned parameters into fresh model
        model_ppo.set_parameters(current_params)
    
    for param in model_ppo.policy.parameters():
        if param.requires_grad:
            param.register_hook(lambda grad: torch.clamp(grad, -2.5, 2.5))

    # === Reptile inner loop + meta-update ===
    # 1) snapshot current “meta” weights
    meta_params_nested = model_ppo.get_parameters()
    flat_meta = flatten_dict(meta_params_nested)

    # print_weights(model_ppo, f"Before training on {tic}")
    # 2) Fine-tune on this ticker
    model_ppo.learn(
        total_timesteps=INNER_TIMESTEPS,
        reset_num_timesteps=False,
    )
    
    # 3) Get adapted parameters
    adapted_params_nested = model_ppo.get_parameters()
    flat_adapt = flatten_dict(adapted_params_nested)

    # Apply Reptile update
    meta_params_nested = update_params(meta_params_nested, adapted_params_nested)

    # 5) Restructure flat parameters back to nested format and write back to the model
    # We need to maintain the original nested structure when setting parameters
    model_ppo.set_parameters(meta_params_nested)

    # ==========================================

    # … then your evaluation & logging on this ticker, exactly as before …
    
    trade = data_split(proc, "2022-12-21", "2023-12-20")
    print(f"Trading Data Len {len(train)}")
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(
        model=model_ppo,
        environment=e_trade_gym
    )
    print(list(df_actions["actions"]))
    all_returns[tic] = df_daily_return.copy()
    metrics.append({
        "idx": idx,
        "ticker": tic,
        "end_total_asset": df_daily_return["account_value"].iloc[-1],
        "start_total_asset": df_daily_return["account_value"].iloc[0],
        "return": (df_daily_return["account_value"].iloc[-1] / df_daily_return["account_value"].iloc[0]) - 1,
        # "action_mean": df_actions["actions"].mean()
    })
    print(f"Evaluation metrics for {tic}:")
    print(f"  - End total asset: {metrics[-1]['end_total_asset']:.2f}")
    print(f"  - Return: {metrics[-1]['return']*100:.2f}%")
    # print(f"  - Action mean/std: {metrics[-1]['action_mean']:.4f}")
    
    # optional: checkpoint every 5 tasks
    if (idx + 1) % 5 == 0:
        ckpt = os.path.join(OUTPUT_ROOT, f"SAC_reptile_meta_after_{tic}.zip")
        model_ppo.save(ckpt)
        print(f"  ↳ checkpoint saved to {ckpt}")



ckpt = os.path.join(OUTPUT_ROOT, f"BF_SAC_{time.time()}_final_model.zip")
model_ppo.save(ckpt)

print("\n✅ Completed Reptile meta-training over all tickers.")
print(ckpt)
print(metrics)

"""
# === Fine-tune on Bitcoin ("BTC") ===
# assume your BTC file is named BTC_market_data.csv in DATA_DIR
df_btc = pd.read_csv(f"{DATA_DIR}/BTC_market_data.csv").rename(...)
df_btc["tic"] = "BTC"
proc_btc = fe.preprocess_data(df_btc)
train_btc = data_split(proc_btc, "2009-01-05", "2022-12-20")
env_btc, _ = StockTradingEnv(df=train_btc, **env_kwargs).get_sb_env()

# plug the meta-model into BTC env
model_ppo.set_env(env_btc)
# (we use fewer inner steps, since meta-init should be strong)
model_ppo.learn(total_timesteps=20_000, reset_num_timesteps=False)

print("🔧 Fine-tuned on BTC; ready for evaluation.")
"""
