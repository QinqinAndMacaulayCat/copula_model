import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

START_DATE = "2010-01-01"
END_DATE   = "2025-11-03"

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    tbl = pd.read_html(StringIO(html))[0]
    syms = (tbl["Symbol"].astype(str)
                    .str.replace(".", "-", regex=False)
                    .str.upper())
    return sorted(syms.unique())

tickers = get_sp500_tickers()

raw = yf.download(
    tickers=" ".join(tickers),
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True,
    progress=False
)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw.xs("Close", axis=1, level=-1)
else:
    col = "Close" if "Close" in raw.columns else raw.columns[0]
    prices = raw[[col]].rename(columns={col: tickers[0]})

prices = prices.sort_index().dropna(how="all")
prices = prices.loc[:, prices.notna().sum() > 0]
prices.to_csv("sp500_prices.csv", float_format="%.6f")

logret = np.log(prices / prices.shift(1)).dropna(how="all")
logret.to_csv("sp500_log_returns.csv", float_format="%.8f")

print(f"{prices.index.min().date()} ~ {prices.index.max().date()}，共 {prices.shape[1]} 只股票")
