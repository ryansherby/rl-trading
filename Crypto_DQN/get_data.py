import yfinance as _yf
import pandas as pd



def yfinance_daily_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV candles via yFinance.

    `ticker` should follow Yahoo’s format, e.g. 'BTC-USD', 'ETH-USD'.
    """
    if _yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    raw = _yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False)
    raw.columns = raw.columns.droplevel("Ticker")
    raw = raw.reset_index()
    raw.columns.name = None
    

    return raw

tickers = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "USDT-USD",  # Tether
    "BNB-USD",   # BNB (Binance Coin)
    "SOL-USD",   # Solana
    "XRP-USD",   # XRP (Ripple)
    "USDC-USD",  # USD Coin
    "DOGE-USD",  # Dogecoin
    "ADA-USD",   # Cardano
    "AVAX-USD",  # Avalanche
    "TRX-USD",   # Tron
    "DOT-USD",   # Polkadot
    "SHIB-USD"   # Shiba Inu
]

if __name__ == "__main__":
    for tic in tickers:
        print(f"Downloading {tic}...")
        df = yfinance_daily_ohlcv(tic,end="2025-01-01")
        df.to_csv(f"./Crypto_DQN/data/{tic}.csv",index=False)
        print(f"Saved {tic} data to data/{tic}.csv")