"""Download and load the data.

    python -m gmv.data      S&P 500 prices 2013-2018 (Kaggle "S&P 500 stock data", 30 MB) and the
                            Kenneth French daily portfolios and risk-free rate (16 MB)
"""
import datetime
import io
import os
import urllib.request
import zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

SP500_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
PAPER_WINDOW = ("2013-04-22", "2014-03-19")                        # T = 230, as in the paper
FRENCH_URL = "mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH = {"ff100": "100_Portfolios_10x10_Daily_CSV.zip",
          "ind49": "49_Industry_Portfolios_daily_CSV.zip",
          "factors": "F-F_Research_Data_Factors_daily_CSV.zip"}
FRENCH_START = {"ff100": "1963-07-01", "ind49": "1971-03-12"}       # first day without gaps, see README


def results_dir(study):
    path = os.path.join(RESULTS, f"study{study}")
    os.makedirs(path, exist_ok=True)
    return path


def sp500_prices():
    return pd.read_csv(os.path.join(DATA, "sp500_close.csv"), index_col=0, parse_dates=True)


def sp500_returns():
    """Daily returns 2013-2018 of the stocks with a complete history."""
    return sp500_prices().dropna(axis=1).pct_change().iloc[1:]


def sp500_paper_window():
    return pd.read_csv(os.path.join(DATA, "sp500_returns.csv"), index_col=0, parse_dates=True)


def french(name):
    return pd.read_csv(os.path.join(DATA, f"{name}_daily.csv"), index_col=0, parse_dates=True)


def risk_free(dates):
    rf = pd.read_csv(os.path.join(DATA, "ff_rf_daily.csv"), index_col=0, parse_dates=True)["RF"]
    return rf.reindex(dates).to_numpy()


def fetch_sp500():
    raw = os.path.join(DATA, "all_stocks_5yr.csv")
    if not os.path.exists(raw):
        urllib.request.urlretrieve(SP500_URL, raw)
    d = pd.read_csv(raw, parse_dates=["date"])
    close = d.pivot(index="date", columns="Name", values="close").sort_index()
    close.to_csv(os.path.join(DATA, "sp500_close.csv"))
    start, end = PAPER_WINDOW
    w = close.loc[:end]
    w = w.iloc[w.index.get_indexer([pd.Timestamp(start)])[0] - 1:].dropna(axis=1)
    r = w.pct_change().iloc[1:]
    r.to_csv(os.path.join(DATA, "sp500_returns.csv"))
    print(f"S&P 500: {close.shape[1]} tickers, {r.shape[1]} complete in the paper's window, {len(r)} days")


def _download_french(name):
    path = os.path.join(DATA, "french", name)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for scheme in ("http://", "https://"):                         # the library served plain HTTP in 2026
        try:
            req = urllib.request.Request(scheme + FRENCH_URL + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            if body[:2] == b"PK":
                with open(path, "wb") as f:
                    f.write(body)
                return path
        except OSError:
            pass
    raise RuntimeError(f"could not download {name}")


def _first_table(path):
    """The first table of a French CSV (value-weighted returns, or the factors), in decimals, NaN for missing."""
    z = zipfile.ZipFile(path)
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head = next(k for k, l in enumerate(lines) if l.startswith(","))
    end = next(k for k in range(head + 1, len(lines)) if not lines[k].strip()[:8].isdigit())
    df = pd.read_csv(io.StringIO("\n".join(lines[head:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.columns = df.columns.str.strip()
    return df.mask(df <= -99.99) / 100, lines[head - 1].strip(), lines[0].strip()


def fetch_french():
    info = [f"Downloaded {datetime.date.today()} from http://{FRENCH_URL}"]
    for key in ("ff100", "ind49"):
        df, title, created = _first_table(_download_french(FRENCH[key]))
        assert "Value Weighted" in title, title
        sub = df[FRENCH_START[key]:]
        gaps = list(sub.columns[sub.isna().any()])
        sub = sub.drop(columns=gaps)
        sub.to_csv(os.path.join(DATA, f"{key}_daily.csv"), float_format="%.6g")
        big = sub.abs().stack().nlargest(3)
        info += [f"{FRENCH[key]}: {created} Section: {title}.",
                 f"  full file {df.index[0].date()} to {df.index[-1].date()}, {df.shape[1]} series;"
                 f" kept {sub.index[0].date()} to {sub.index[-1].date()} ({len(sub)} days), p = {sub.shape[1]}",
                 f"  dropped for missing values after {FRENCH_START[key]}: {', '.join(gaps) if gaps else 'none'}",
                 "  largest absolute daily returns: " + "; ".join(f"{c} {d.date()} {v:.1%}" for (d, c), v in big.items())]
    f, _, created = _first_table(_download_french(FRENCH["factors"]))
    f[["RF"]].to_csv(os.path.join(DATA, "ff_rf_daily.csv"), float_format="%.6g")
    info.append(f"{FRENCH['factors']}: {created} Risk-free rate to {f.index[-1].date()}.")
    with open(os.path.join(DATA, "french_info.txt"), "w") as fh:
        fh.write("\n".join(info) + "\n")
    print("\n".join(info))


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    fetch_sp500()
    fetch_french()
