import yfinance as yf
import pandas as pd
from datetime import datetime

def get_stock_data_with_market_cap(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Download historical OHLCV data
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    # Get number of shares outstanding
    shares_outstanding = stock.info.get('sharesOutstanding', None)
    if shares_outstanding is None:
        raise ValueError("Could not retrieve shares outstanding (market cap can't be calculated).")

    # Clean up DataFrame
    hist.reset_index(inplace=True)
    hist['Start'] = hist['Date']
    hist['End'] = hist['Date']
    hist['Market Cap'] = hist['Close'] * shares_outstanding

    # Final selection of columns
    result = hist[['Start', 'End', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']].copy()
    return result

def save_to_csv(df: pd.DataFrame, ticker: str, file_name: str = None, is_train:bool = True):
    if file_name is None and is_train:
        file_name = f"./data/train/{ticker}_market_data.csv"
    else:
        file_name = f"./data/val/{ticker}_market_data.csv"
    df.to_csv(file_name, index=False)
    print(f"✅ Data saved to {file_name}")

# Example usage
if __name__ == "__main__":
    ticker = "AXP"
    start = "2009-01-01"
    end = "2023-12-31"
    is_train = True
    tickers = [
    "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "2222.SR", "META", "BRK.B", "AVGO", "TSLA",
    "TSM", "WMT", "LLY", "JPM", "V", "TCEHY", "MA", "NFLX", "XOM", "COST",
    "ORCL", "JNJ", "PG", "HD", "UNH", "SAP", "ABBV", "NVO", "BAC",
    "KO", "PLTR", "BABA", "MDIKY", "TMUS", "HESAY", "LVMUY", "NSRGY", "PM", "ASML",
    "CRM", "RHHBY", "SSNLF", "ACGBY", "TM", "CVX", "CSCO", "WFC", "LRLCY", "ABT"
    ]
    import random
    random.shuffle(tickers)
    
    for ticker in tickers[:((len(tickers)*3)//4)]:
        try:
            df = get_stock_data_with_market_cap(ticker, start, end)
            print(f"Done {ticker}")
        except:
            print(f"Not done {ticker}")
        
        # Save to CSV
        save_to_csv(df, ticker, is_train=True)

    for ticker in tickers[((len(tickers)*3)//4):]:
        try:
            df = get_stock_data_with_market_cap(ticker, start, end)
            print(f"Done {ticker}")
        except:
            print(f"Not done {ticker}")
        
        # Save to CSV
        save_to_csv(df, ticker, is_train=False)