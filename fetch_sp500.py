"""
Daily closing prices of S&P 500 stocks, February 2013 to February 2018.
Source: the Kaggle dataset "S&P 500 stock data" (camnugent), mirrored by plotly on GitHub. It holds the
companies in the index in February 2018 (505 tickers), so it carries survivorship bias, as does the paper's
own selection of stocks that traded through the whole window.

Writes
  data/all_stocks_5yr.csv     the raw file
  data/sp500_close.csv        closing prices, dates x tickers
  data/sp500_returns.csv      simple daily returns in the paper's window, 22 April 2013 to 19 March 2014,
                              for the stocks with a complete price history in the window
"""
import os
import urllib.request
import pandas as pd
from common import DATA

URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
RAW = os.path.join(DATA, "all_stocks_5yr.csv")
START, END = "2013-04-22", "2014-03-19"        # the paper: T = 230 trading days

os.makedirs(DATA, exist_ok=True)
if not os.path.exists(RAW):
    urllib.request.urlretrieve(URL, RAW)
d = pd.read_csv(RAW, parse_dates=["date"])
close = d.pivot(index="date", columns="Name", values="close").sort_index()
close.to_csv(os.path.join(DATA, "sp500_close.csv"))

window = close.loc[:END]
window = window.iloc[window.index.get_indexer([pd.Timestamp(START)])[0] - 1:]   # one day before the first return
window = window.dropna(axis=1)
r = window.pct_change().iloc[1:]
r.to_csv(os.path.join(DATA, "sp500_returns.csv"))
print(f"{close.shape[1]} tickers in the file, {r.shape[1]} with a complete history in the window")
print(f"{len(r)} daily returns, {r.index[0].date()} to {r.index[-1].date()} (the paper: T = 230, 417 stocks)")
print(f"largest absolute daily return: {r.abs().max().max():.3f} ({r.abs().max().idxmax()})")
