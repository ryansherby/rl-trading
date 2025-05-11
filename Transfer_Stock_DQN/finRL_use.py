import os
import pandas as pd
import sys
print(sys.path)
import finrl
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import sys
from finrl.config import INDICATORS
from stable_baselines3 import SAC
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
tickers = {
    "btc"
    }

DATA_DIR    = "./Transfer_Stock_DQN/data"
OUTPUT_ROOT = "./Transfer_Stock_DQN/outputs"

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True,
    user_defined_feature=False
)
common_env_kwargs = {
    "hmax": 10,
    "initial_amount": 1_000_000,
    "buy_cost_pct": None,  # filled per-ticker below
    "sell_cost_pct": None, # "
    "num_stock_shares": None, # "
    "reward_scaling": 1e-4
}
# 3. Prepare a place to save checkpoints
os.makedirs(OUTPUT_ROOT, exist_ok=True)
model_ppo = None
agent     = None
policy_kwargs = dict(
    dueling=True,
    net_arch=[512, 256],
)
all_returns = {}
metrics = [] 

for idx, tic in enumerate(tickers):
    print(f"\n--- Epoch {idx+1}/{len(tickers)}: Training on {tic.upper()} ---")

    # a) load & rename
    df = pd.read_csv(f"{DATA_DIR}/{tic}.csv").rename(columns={
        "Start":     "date",
        "Adj Close": "adjcp",
        "Close":     "close",
        "High":      "high",
        "Low":       "low",
        "Volume":    "volume",
        "Open":      "open",
    })
    df["tic"] = tic

    # b) featurize
    proc = fe.preprocess_data(df)

    # c) split
    train = data_split(proc, "2013-12-31", "2024-01-01")
    # (you can skip making trade set here if you’re not evaluating yet)

    # d) env kwargs
    stock_dim   = len(train.tic.unique())      # will be 1
    state_space = 1 + 2*stock_dim + len(INDICATORS)*stock_dim
    env_kwargs = {
        "hmax":               10,
        "initial_amount":     1_000_000,
        "buy_cost_pct":       [0.001]*stock_dim,
        "sell_cost_pct":      [0]*stock_dim,
        "num_stock_shares":   [0]*stock_dim,
        "stock_dim":          stock_dim,
        "action_space":       stock_dim,
        "state_space":        state_space,
        "tech_indicator_list": INDICATORS,
        "reward_scaling":      1e-4,
    }

    

    # e) make SB3 env
    env_train, _ = StockTradingEnv(df=train, **env_kwargs).get_sb_env()

    if idx == 0:
        # first ticker → instantiate agent + model
        agent   = DRLAgent(env=env_train)
        # model_ppo = agent.get_model("ppo")
        # model_ppo  = SAC(
        #     "MlpPolicy",
        #     env_train,
        #     verbose=0,
        #     gamma = 0.95,
        #     device="cuda",
        #     tensorboard_log=OUTPUT_ROOT,
        #     policy_kwargs={
        #         "net_arch": [512, 256],
        #         # optional: choose activation, etc.
        #     },
        # )
        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": .01,
            "device": "mps",
        }
        model_ppo = agent.get_model("sac",model_kwargs = SAC_PARAMS)

        # set up logging once
        logger = configure(OUTPUT_ROOT, ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(logger)

    else:
        # reuse the same model, just point it at the new env
        model_ppo.set_env(env_train)

    # f) train (will continue from previous weights)
    # model_ppo = agent.train_model(
    #     model=model_ppo,
    #     tb_log_name=f"{tic}_ppo",
    #     total_timesteps=50000,   # adjust per‐ticker budget
    # )
    # model_ppo.learn(
    # total_timesteps=50000,
    # reset_num_timesteps=False,)
    trained_sac = agent.train_model(model=model_ppo, 
                             tb_log_name='sac',
                             total_timesteps=80000)

    trade = data_split(proc, "2024-01-01", "2025-01-01")
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    df_daily_return, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac,
        environment=e_trade_gym
    )

    # store the test curve
    all_returns[tic] = df_daily_return.copy()
    metrics.append({
        "ticker": tic,
        "end_total_asset": df_daily_return["account_value"].iloc[-1]
    })

    # g) checkpoint after each ticker
    if idx%5 == 0:
        ckpt = os.path.join(OUTPUT_ROOT, f"dqn_after_{tic}.zip")
        model_ppo.save(ckpt)
        print(f"  ↳ checkpoint saved to {ckpt}")

print("\n✅ All tickers trained sequentially on one shared PPO model.")
print(metrics)


# Path to save
out_path = os.path.join(OUTPUT_ROOT, "all_tickers_equity_curves.png")

# Grab the “date” series from one of the tickers and parse as datetime
example_dates = pd.to_datetime(next(iter(all_returns.values()))["date"])
first_date = example_dates.iloc[0]
last_date  = example_dates.iloc[-1]

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
for ticker, df_ret in all_returns.items():
    # ensure we plot datetime on x-axis
    dates = pd.to_datetime(df_ret["date"])
    ax.plot(dates, df_ret["account_value"], label=ticker)

# Only set ticks at first & last date
ax.set_xticks([first_date, last_date])
ax.set_xticklabels(
    [first_date.strftime("%Y-%m-%d"), last_date.strftime("%Y-%m-%d")],
    rotation=45,
    ha="right"
)

ax.set_xlabel("Date")
ax.set_ylabel("Account Value")
ax.set_title("Test Equity Curves by Ticker")
ax.legend(ncol=2, fontsize="small")
plt.tight_layout()
plt.savefig(out_path)
print(f"↳ Saved combined equity-curve plot to {out_path}")

# 2) (Optional) bar‐plot of final asset
df_metrics = pd.DataFrame(metrics)
plt.figure(figsize=(10, 6))
plt.bar(df_metrics["ticker"], df_metrics["end_total_asset"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("End Total Asset")
plt.title("Final Portfolio Value by Ticker")
plt.tight_layout()
out_path2 = os.path.join(OUTPUT_ROOT, "final_asset_by_ticker.png")
plt.savefig(out_path2)
print(f"↳ Saved bar‐plot of end‐period assets to {out_path2}")