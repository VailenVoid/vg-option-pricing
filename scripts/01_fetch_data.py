"""Download and cache a market snapshot.

    python scripts/01_fetch_data.py [--ticker ^XSP] [--history-ticker ^GSPC]

Writes data/<ticker>_chain_<timestamp>.csv, data/<ticker>_history.csv and
data/snapshot.json.  Everything downstream reads the cached snapshot, so the
paper's numbers stay reproducible after the live feed has moved on.

The default is `^XSP` (Mini-SPX): its options are **European-style** and
cash-settled, which is the contract the paper's Theorem 4.1 actually prices.
SPY is more liquid but its options are American and can be exercised early, so
model and data would be describing different instruments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vg.data import (
    DEFAULT_DATA_DIR,
    MarketSnapshot,
    download_chain,
    download_history,
    risk_free_from_irx,
    save_snapshot,
    trailing_dividend_yield,
)

# Options on a price index are European; options on an ETF are American.
EXERCISE_STYLE = {
    "^XSP": "European (cash-settled index option)",
    "^SPX": "European (cash-settled index option)",
    "SPY": "American (physically settled ETF option)",
    "QQQ": "American (physically settled ETF option)",
}

# A price index pays no dividends of its own, so its trailing yield reads 0 --
# but its options price off a forward that *is* reduced by the constituents'
# dividends.  The tracking ETF is the standard proxy for that fallback.
DIVIDEND_PROXY = {"^XSP": "SPY", "^SPX": "SPY"}

# Longer, equivalent return history for the same underlying process.
HISTORY_PROXY = {"^XSP": "^GSPC", "^SPX": "^GSPC"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="^XSP", help="option chain ticker")
    ap.add_argument("--history-ticker", default=None,
                    help="ticker for the return history (defaults to a sensible proxy)")
    ap.add_argument("--dividend-proxy", default=None,
                    help="ticker to read the fallback dividend yield from")
    ap.add_argument("--history-period", default="10y")
    ap.add_argument("--max-expiries", type=int, default=12)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    hist_ticker = args.history_ticker or HISTORY_PROXY.get(args.ticker, args.ticker)
    div_ticker = args.dividend_proxy or DIVIDEND_PROXY.get(args.ticker, args.ticker)
    style = EXERCISE_STYLE.get(args.ticker, "unknown -- check before relying on it")

    print(f"Option chain: {args.ticker}   exercise style: {style}")
    if "American" in style:
        print("  WARNING: the model prices European options.  Early exercise makes")
        print("  these the wrong instrument; consider --ticker ^XSP or ^SPX.")

    chain, spot = download_chain(args.ticker, max_expiries=args.max_expiries)
    print(f"  spot = {spot:.2f}, {len(chain):,} contracts, "
          f"{chain['expiry'].nunique()} expiries")

    print(f"\nReturn history: {hist_ticker} ({args.history_period})")
    if hist_ticker != args.ticker:
        print(f"  ({args.ticker} has only a short history; {hist_ticker} is the same")
        print("   underlying process, so the return law estimated from it is the same.)")
    history = download_history(hist_ticker, args.history_period)
    print(f"  {len(history):,} daily observations, "
          f"{history['Date'].min().date()} to {history['Date'].max().date()}")

    r = risk_free_from_irx()
    q = trailing_dividend_yield(div_ticker)
    print(f"\nFallback rates: r = {r:.4%} (^IRX), q = {q:.4%} (trailing dividends "
          f"of {div_ticker})")
    print("  -- used only where put-call parity cannot be fitted.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = args.ticker.lstrip("^")
    meta = MarketSnapshot(
        ticker=args.ticker,
        asof=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        spot=spot,
        risk_free_irx=r,
        dividend_yield=q,
        chain_file=f"{slug}_chain_{stamp}.csv",
        history_file=f"{hist_ticker.lstrip('^')}_history.csv",
        n_raw_quotes=len(chain),
        history_ticker=hist_ticker,
        exercise_style=style,
    )
    save_snapshot(chain, history, meta, args.data_dir)

    print(f"\nSaved to {args.data_dir}")
    print(f"  {meta.chain_file}")
    print(f"  {meta.history_file}")
    print("  snapshot.json")

    two_sided = ((pd.to_numeric(chain["bid"], errors="coerce").fillna(0) > 0)
                 & (pd.to_numeric(chain["ask"], errors="coerce").fillna(0) > 0)).mean()
    print(f"\nTwo-sided quotes: {two_sided:.1%} of contracts.")
    if two_sided < 0.2:
        print("  US options market is closed; last traded prices will be used")
        print("  instead of mids.  Re-run during 09:30-16:00 US/Eastern for")
        print("  live bid/ask if the calibration is meant to be a live one.")


if __name__ == "__main__":
    main()
