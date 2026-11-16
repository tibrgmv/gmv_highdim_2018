"""
Daily data from the Kenneth French data library, for Study 6.

  100 portfolios formed on size and book-to-market (10 x 10), value-weighted   ->  data/ff100_daily.csv
  49 industry portfolios, value-weighted                                     ->  data/ind49_daily.csv
  the daily risk-free rate (Fama-French research factors)                     ->  data/ff_rf_daily.csv
  a short record of what was downloaded and which series were kept          ->  data/french_info.txt

Returns are converted from percent to decimals; -99.99 and -999 mean missing. The portfolios are built from
CRSP, which keeps delisted firms, so the data have no survivorship bias.

Missing values (decided by looking at where they are, before any backtest):
  100 portfolios: we start on 1 July 1963, when Compustat coverage begins, and drop the five portfolios that
    still have gaps after that date (ME7 BM10, ME9 BM10, ME10 BM8, ME10 BM9, BIG HiBM: large growth or value
    corners that are empty in some years, BIG HiBM until the end of the sample). This leaves p = 95.
  49 industries: every series is complete from 12 March 1971 (Health starts in July 1969, Software has one
    missing day, 11 March 1971), so we start there and keep all 49.

The raw zips stay in data/french/ (git-ignored). The library serves these files over plain HTTP; the HTTPS
address returned 404 when we downloaded them.
"""
import io
import os
import sys
import zipfile
import datetime
import urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "french")
BASE = "mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"ff100": "100_Portfolios_10x10_Daily_CSV.zip",
         "ind49": "49_Industry_Portfolios_daily_CSV.zip",
         "factors": "F-F_Research_Data_Factors_daily_CSV.zip"}
START = {"ff100": "1963-07-01", "ind49": "1971-03-12"}


def download(name):
    path = os.path.join(RAW, name)
    if os.path.exists(path):
        return path
    os.makedirs(RAW, exist_ok=True)
    for scheme in ("http://", "https://"):
        try:
            req = urllib.request.Request(scheme + BASE + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            if body[:2] == b"PK":
                with open(path, "wb") as f:
                    f.write(body)
                return path
        except Exception as e:                           # try the other scheme
            print(scheme, name, e, file=sys.stderr)
    raise RuntimeError(f"could not download {name}")


def first_section(path):
    """The first table of a French CSV file (value-weighted returns, or the factors), in decimals, NaN for missing."""
    z = zipfile.ZipFile(path)
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head = next(k for k, l in enumerate(lines) if l.startswith(","))            # the header row of the first table
    end = next(k for k in range(head + 1, len(lines)) if not lines[k].strip()[:8].isdigit())
    df = pd.read_csv(io.StringIO("\n".join(lines[head:end])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.columns = df.columns.str.strip()
    title = lines[head - 1].strip()
    return df.mask(df <= -99.99) / 100, title, lines[0].strip()


def main():
    info = [f"Downloaded {datetime.date.today()} from http://{BASE}"]
    for key in ("ff100", "ind49"):
        df, title, created = first_section(download(FILES[key]))
        assert "Value Weighted" in title, title
        sub = df[START[key]:]
        gaps = list(sub.columns[sub.isna().any()])
        sub = sub.drop(columns=gaps)
        assert not sub.isna().any().any()
        sub.to_csv(os.path.join(DATA, f"{key}_daily.csv"), float_format="%.6g")
        big = sub.abs().stack().nlargest(3)
        info += [f"{FILES[key]}: {created} Section: {title}.",
                 f"  full file {df.index[0].date()} to {df.index[-1].date()}, {df.shape[1]} series;"
                 f" kept {sub.index[0].date()} to {sub.index[-1].date()} ({len(sub)} days), p = {sub.shape[1]}",
                 f"  dropped for missing values after {START[key]}: {', '.join(gaps) if gaps else 'none'}",
                 "  largest absolute daily returns: " + "; ".join(f"{c} {d.date()} {v:.1%}" for (d, c), v in big.items())]
    f, title, created = first_section(download(FILES["factors"]))
    f[["RF"]].to_csv(os.path.join(DATA, "ff_rf_daily.csv"), float_format="%.6g")
    info.append(f"{FILES['factors']}: {created} Risk-free rate to {f.index[-1].date()}.")
    with open(os.path.join(DATA, "french_info.txt"), "w") as fh:
        fh.write("\n".join(info) + "\n")
    print("\n".join(info))


if __name__ == "__main__":
    main()
