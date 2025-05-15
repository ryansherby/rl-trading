# FinReptile: A Data-Efficient Meta-Learning Framework for Cross-Domain Algorithmic Trading

## Dataset

Run the $createdata.py$ file to download and format publicly available stock data available from the $yfinance$ library. Select a training set and test set from the data based on a cutoff date for both stock data and crypto data.

## Training Environment

Download the $StockTradingEnv$ from $finRL$. Ensure the reward function is consistent with the description in the paper.

## Agent

Define the parameters for each DRL algorithm and initialize an agent instance provided by $stable_baselines3$ library.

## Training Runtime Loop (Pre-Training)

For each agent and collected stock ticker, run a training loop by processing each ticker sequentially on the given agent. Save the weights of these agents.

## Training Runtime Loop (Post-Training)

For each stored agent and collected crypto ticker, run a training loop by processing each ticker sequentially on the given agent. We specify particularly that this is a training instance as the weights of these agents will be modified during the runtime.

Initialize new agents under the same RL algorithms defined previously. Run a training loop for these agents on the same crypto tickers.

Save the policies for both the pre-trained agents and the randomly initialized agents.

## Testing Loop

Run the pre-trained agents' policies and the randomly initialized agents' policies on the crypto test set.

## Evaluation

Compare the net worth increase using the crypto test set and time to policy convergence between the pre-trained agents and the randomly initialized agents for each algorithm.