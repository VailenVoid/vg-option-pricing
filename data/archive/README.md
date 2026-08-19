# Archived snapshots

Superseded market data, kept because option chains cannot be re-downloaded after
the fact -- `yfinance` serves only the live chain, so a snapshot that is deleted
is gone.

* `SPY_chain_20260817T115546Z.csv`, `SPY_history.csv` — the first snapshot the
  project was built on. SPY options are **American**, and the paper's Theorem 4.1
  prices a European call, so the project moved to `^XSP` (European,
  cash-settled). These files are no longer read by anything; they are here so the
  earlier numbers stay reproducible.

Nothing in `data/archive/` is loaded by the pipeline. The active snapshot is
whatever `data/snapshot.json` points at.
