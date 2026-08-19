"""Market data for the VG study.

Source: Yahoo Finance through `yfinance` -- free, no API key, and it gives both
the full listed option chain and the underlying's price history, which is what
the paper needs.  Every download is written to `data/` as CSV with the snapshot
timestamp in the filename, so the results of the paper stay reproducible even
though the live feed moves.

Which contract to use.  The paper prices a *European* call, so the data has to
be European too.  Listed options on the SPY ETF are American-style and can be
exercised early, which makes them the wrong instrument however convenient they
are.  The default here is therefore `^XSP` (Mini-SPX): options on the S&P 500
index itself, European-style and cash-settled, so there is no early exercise to
account for and the model and the data describe the same contract.  `^SPX` is
the same instrument at ten times the size.

A consequence of using an index rather than an ETF: `^XSP` has price history
only from 2021, while the return law we want to estimate is that of the S&P 500,
for which `^GSPC` goes back decades.  The two are the same process up to a
factor of ten -- measured over their common sample the daily log returns
correlate at 0.99933 and the annualised volatilities agree to 0.03 percentage
points -- so the chain is taken from `^XSP` and the return history from
`^GSPC`.  `download_history` and `download_chain` therefore accept different
tickers.

Two more things deserve comment.

Risk-free rate and dividends.  The underlying pays dividends, so pricing off the raw spot
with a Treasury rate is wrong by ~1% of forward, which is much larger than the
effects we are trying to measure.  Rather than plug in a dividend yield we
recover the discount factor D and the forward F for each expiry from put-call
parity,

    C(K) - P(K) = D * (F - K),

by regressing C - P on K over the liquid strikes: the slope is -D and the
intercept is D*F.  This is what the market itself implies and it absorbs rate,
dividend and borrow in one number.  Where there are too few usable put/call
pairs we fall back on the 13-week T-bill (^IRX) and the trailing dividend yield,
and the fallback is recorded in the output.

Quote prices.  The mid of a two-sided quote is used whenever one exists.
Outside US trading hours Yahoo reports bid = ask = 0, and then the last traded
price is used instead; which one was used is kept in a `price_source` column so
that the downstream filters (and the reader) can tell the difference.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "MarketSnapshot",
    "download_chain",
    "download_history",
    "risk_free_from_irx",
    "trailing_dividend_yield",
    "implied_forward_curve",
    "build_quotes",
    "save_snapshot",
    "load_snapshot",
]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class MarketSnapshot:
    """Everything needed to reproduce a calibration, in one object."""

    ticker: str
    asof: str
    spot: float
    risk_free_irx: float
    dividend_yield: float
    chain_file: str
    history_file: str
    n_raw_quotes: int
    history_ticker: str = ""
    exercise_style: str = ""

    def __post_init__(self) -> None:
        if not self.history_ticker:
            self.history_ticker = self.ticker

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "MarketSnapshot":
        return MarketSnapshot(**json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _last_close(ticker: str, period: str = "10d") -> float:
    """Most recent finite close.

    The `.iloc[-1]` shortcut is not safe here: before the US open Yahoo already
    lists a bar for the current day with a NaN close, and taking it silently
    poisons the spot and everything downstream of it.
    """
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"no price history returned for {ticker}")
    closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if closes.empty:
        raise RuntimeError(f"no finite close in the last {period} for {ticker}")
    return float(closes.iloc[-1])


def download_chain(
    ticker: str = "^XSP",
    *,
    max_expiries: int | None = 14,
    min_days: int = 5,
    max_days: int = 550,
) -> tuple[pd.DataFrame, float]:
    """Download the listed option chain and the spot price.

    Returns a long DataFrame with one row per contract (calls and puts stacked,
    distinguished by `cp`) and the spot.  Expiries are filtered to
    [min_days, max_days] and, if `max_expiries` is given, thinned to a spread of
    maturities rather than just the first few -- the term structure is one of
    the things we want to look at.

    Defaults to `^XSP`, whose options are European-style; see the module
    docstring on why that matters.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = _last_close(ticker)

    asof = pd.Timestamp.now(tz="UTC").normalize()
    expiries = []
    for e in tk.options:
        days = (pd.Timestamp(e, tz="UTC") - asof).days
        if min_days <= days <= max_days:
            expiries.append((e, days))
    if not expiries:
        raise RuntimeError(f"no expiries for {ticker} within [{min_days}, {max_days}] days")

    if max_expiries is not None and len(expiries) > max_expiries:
        # Spread the selection roughly uniformly in log-maturity.
        days = np.array([d for _, d in expiries], dtype=float)
        targets = np.exp(np.linspace(np.log(days.min()), np.log(days.max()), max_expiries))
        keep_idx = sorted({int(np.abs(days - t).argmin()) for t in targets})
        expiries = [expiries[i] for i in keep_idx]

    frames = []
    for exp, days in expiries:
        try:
            chain = tk.option_chain(exp)
        except Exception as exc:  # network hiccup on a single expiry
            warnings.warn(f"skipping expiry {exp}: {exc}")
            continue
        for cp, df in (("C", chain.calls), ("P", chain.puts)):
            if df is None or df.empty:
                continue
            d = df.copy()
            d["cp"] = cp
            d["expiry"] = exp
            d["days_to_expiry"] = days
            frames.append(d)

    if not frames:
        raise RuntimeError(f"no option data downloaded for {ticker}")

    out = pd.concat(frames, ignore_index=True)
    out["spot"] = spot
    out["snapshot_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out, spot


def download_history(ticker: str = "^GSPC", period: str = "10y") -> pd.DataFrame:
    """Daily history for the physical-parameter estimation.

    `auto_adjust=True` so that for an ETF the dividend drops are not mistaken
    for jumps; on a price index it is a no-op.  Rows with a non-finite close are
    dropped -- Yahoo publishes the current day's bar before the market opens.
    """
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"no history for {ticker}")
    df = df.reset_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[np.isfinite(df["Close"])].reset_index(drop=True)
    if len(df) < 2:
        raise RuntimeError(f"history for {ticker} has fewer than 2 usable closes")
    df["log_return"] = np.log(df["Close"]).diff()
    return df


def risk_free_from_irx(default: float = 0.04) -> float:
    """Continuously-compounded short rate from the 13-week T-bill index (^IRX).

    ^IRX is quoted as an annualised percentage; we convert to a continuous rate
    with log(1 + y).  Only a fallback -- the parity-implied discount factors are
    preferred wherever they are available.
    """
    import yfinance as yf

    try:
        h = yf.Ticker("^IRX").history(period="5d")
        if h.empty:
            return default
        return float(np.log1p(float(h["Close"].iloc[-1]) / 100.0))
    except Exception as exc:
        warnings.warn(f"could not read ^IRX ({exc}); using {default}")
        return default


def trailing_dividend_yield(ticker: str = "SPY", spot: float | None = None) -> float:
    """Trailing 12-month dividend yield, as a continuous rate.

    Fallback only -- the parity-implied forward already contains the yield and is
    preferred wherever it can be fitted.

    Note for index tickers: a price index such as `^XSP` or `^SPX` pays no
    dividends of its own and this returns 0, which is *not* the number its
    options price off.  Their forward is the index less the dividend yield of
    its constituents, so pass the tracking ETF (`SPY`) as `ticker` to get a
    usable proxy; `01_fetch_data.py` does this automatically.
    """
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        divs = tk.dividends
        if divs is None or len(divs) == 0:
            return 0.0
        cutoff = divs.index.max() - pd.Timedelta(days=365)
        paid = float(divs[divs.index > cutoff].sum())
        if spot is None:
            spot = _last_close(ticker)
        return float(np.log1p(paid / spot))
    except Exception as exc:
        warnings.warn(f"could not read dividends ({exc}); using 0.0")
        return 0.0


# --------------------------------------------------------------------------- #
# Quote construction
# --------------------------------------------------------------------------- #


def _mid_price(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(price, spread, source) with mid where a two-sided quote exists."""
    bid = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
    ask = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
    last = pd.to_numeric(df["lastPrice"], errors="coerce")

    two_sided = (bid > 0) & (ask > 0) & (ask >= bid)
    price = np.where(two_sided, 0.5 * (bid + ask), last)
    spread = np.where(two_sided, ask - bid, np.nan)
    source = np.where(two_sided, "mid", "last")
    return pd.Series(price, index=df.index), pd.Series(spread, index=df.index), pd.Series(
        source, index=df.index
    )


def implied_forward_curve(
    chain: pd.DataFrame,
    spot: float,
    *,
    fallback_r: float,
    fallback_q: float,
    min_pairs: int = 5,
    band: tuple[float, float] = (0.95, 1.05),
    q_limits: tuple[float, float] = (-0.01, 0.06),
    r_limits: tuple[float, float] = (-0.005, 0.08),
    fwd_tol: float = 0.004,
) -> pd.DataFrame:
    """Per-expiry forward F and discount factor D from put-call parity.

    Two estimators, because quote quality varies enormously with whether the US
    market happens to be open.

    *Regression* (used only when most pairs are genuine two-sided quotes) fits
    `C - P = D F - D K` for both unknowns; the slope gives D and the intercept
    D F.

    *Robust* (the default) fixes `D = exp(-r T)` from the Treasury rate and
    takes `F = median_K [ K + (C - P)/D ]` over near-the-money strikes.  This
    matters more than it looks.  Outside trading hours Yahoo reports
    `bid = ask = 0` and the only prices available are last trades, which for
    calls and puts happened at different moments -- sometimes days apart.  The
    two-parameter regression then reads that timing noise as curvature and
    returns nonsense: on the snapshot this project was built against it produced
    discount factors above 1 (negative interest rates) and dividend yields near
    -4% on half the expiries.  Estimating one parameter instead of two, from a
    median instead of a mean, and only near the money, removes that failure.

    What the result is checked against.  Not the quoted spot: the option quotes
    and the underlying's last close need not come from the same moment, and on
    the snapshot this was built against they did not -- the index history's last
    bar was Friday's close while the chain carried Monday's quotes, leaving the
    "spot" about 1% above what every expiry's parity implied.  Testing the
    parity forwards against that stale number rejected all eight short expiries,
    whose forwards were in fact excellent (the two independent estimators agreed
    to two thousandths of a dollar).

    So the curve is checked against **itself**.  Across expiries the forwards
    must satisfy log F(T) = log S_eff + carry * T; fitting that line gives an
    effective spot implied by the options alone, and each expiry is then tested
    against the fitted curve.  Expiries that still fail fall back to the fitted
    curve rather than to `(fallback_r, fallback_q)`, and the method used is
    recorded in `fwd_method` so nothing silently enters the calibration on a bad
    forward.

    Returns the per-expiry table; `effective_spot` is a constant column on it.
    """
    # ---- pass 1: raw candidates per expiry, no screening yet ----------------
    raw = []
    for (exp, T), grp in chain.groupby(["expiry", "T"], sort=True):
        calls = grp[grp.cp == "C"].set_index("strike")
        puts = grp[grp.cp == "P"].set_index("strike")
        common = calls.index.intersection(puts.index)

        k = c = p = np.empty(0)
        two_sided = np.empty(0, dtype=bool)
        if len(common) >= min_pairs:
            k_all = np.asarray(common, dtype=float)
            sel = (k_all > band[0] * spot) & (k_all < band[1] * spot)
            k = k_all[sel]
            c = calls.loc[common, "price"].to_numpy()[sel]
            p = puts.loc[common, "price"].to_numpy()[sel]
            two_sided = (
                calls.loc[common, "price_source"].to_numpy()[sel] == "mid"
            ) & (puts.loc[common, "price_source"].to_numpy()[sel] == "mid")
            ok = np.isfinite(c) & np.isfinite(p) & np.isfinite(k)
            k, c, p, two_sided = k[ok], c[ok], p[ok], two_sided[ok]
        n_pairs = int(k.size)

        # Best-conditioned estimator first.  The two-parameter regression only
        # has leverage on the slope when the strike range is wide relative to
        # the noise; at a two-week expiry the discount factor is 0.9994 and the
        # fit reads quote noise as a slope.  The one-parameter estimator (D from
        # the rate curve, F from a median) stays well behaved there.
        candidates = []
        if n_pairs >= min_pairs:
            if two_sided.mean() >= 0.6:
                slope, intercept = np.polyfit(k, c - p, 1)
                if slope < 0:
                    d = -slope
                    candidates.append(("parity (regression)", d, intercept / d))
            d0 = float(np.exp(-fallback_r * T))
            candidates.append(
                ("parity (robust forward)", d0, float(np.median(k + (c - p) / d0)))
            )
        raw.append({"expiry": exp, "T": T, "n_pairs": n_pairs, "candidates": candidates})

    # ---- fit log F(T) = log S_eff + carry * T across expiries ---------------
    usable = [
        (row["T"], row["candidates"][0][2])
        for row in raw
        if row["candidates"] and row["candidates"][0][2] > 0
    ]
    if len(usable) >= 2:
        t_arr = np.array([t for t, _ in usable])
        log_f = np.log(np.array([f for _, f in usable]))
        carry, log_s_eff = np.polyfit(t_arr, log_f, 1)
        effective_spot = float(np.exp(log_s_eff))
    else:  # pragma: no cover - a chain this thin cannot be calibrated anyway
        carry = fallback_r - fallback_q
        effective_spot = float(spot)

    def diagnose(d_fit, f_fit, T):
        """(r, q, sane) against the fitted forward curve.

        The test is on the **forward**, not on the implied dividend yield.  `q`
        is derived by dividing by `T`, so at a six-day expiry a one-tenth of a
        percent error in the forward -- economically nothing, well inside the
        bid-ask spread -- becomes a six percent error in `q` and fails any
        plausible yield band.  What has to be right for pricing is the forward.
        """
        if not (np.isfinite(d_fit) and np.isfinite(f_fit)) or d_fit <= 0 or f_fit <= 0:
            return np.nan, np.nan, False
        r = float(-np.log(min(d_fit, 1.0)) / T)
        q = float(r - np.log(f_fit / effective_spot) / T)
        deviation = abs(np.log(f_fit / effective_spot) - carry * T)
        sane = (
            0.80 < d_fit <= 1.0 + 1e-9
            and deviation <= max(fwd_tol, 0.02 * T)
            # The discount factor is the regression's weak leg: at a six-day
            # expiry it is 0.9994 and the slope that produces it is fitted
            # through quote noise, which on this snapshot returned implied short
            # rates from 1% to 11% across neighbouring expiries.  An 11% six-day
            # rate is not a rate, so a regression that produces one is rejected
            # and the one-parameter estimator -- which takes D from the rate
            # curve and only fits the forward -- is used instead.
            and r_limits[0] <= r <= r_limits[1]
        )
        return r, q, sane

    # ---- pass 2: screen each expiry against the fitted curve ----------------
    rows = []
    for row in raw:
        exp, T, n_pairs = row["expiry"], row["T"], row["n_pairs"]
        method, d_fit, f_fit, r, q = "curve fit", np.nan, np.nan, np.nan, np.nan
        for name, d_cand, f_cand in row["candidates"]:
            r_c, q_c, sane = diagnose(d_cand, f_cand, T)
            if sane:
                method, d_fit, f_fit, r, q = name, d_cand, f_cand, r_c, q_c
                break

        if method == "curve fit":
            f_fit = float(effective_spot * np.exp(carry * T))
            d_fit = float(np.exp(-fallback_r * T))
            r = fallback_r
            q = float(r - carry)

        rows.append(
            {
                "expiry": exp,
                "T": T,
                "forward": float(f_fit),
                "discount": float(d_fit),
                "r": float(r),
                "q": float(q),
                "effective_spot": effective_spot,
                "n_parity_pairs": n_pairs,
                "fwd_method": method,
            }
        )
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def _standardised_moneyness(quotes: pd.DataFrame) -> pd.Series:
    """log(K/F) measured in units of the expiry's own at-the-money move.

    A fixed +-20% strike band means something entirely different at two weeks
    than at eighteen months: for the two-week expiry it reaches far past any
    strike that trades, where the only available "price" is a stale print.
    Dividing by the at-the-money volatility times sqrt(T) puts every expiry on
    the same footing, which is the scale a smile is actually shaped on.
    """
    out = pd.Series(np.nan, index=quotes.index, dtype=float)
    for _, grp in quotes.groupby("expiry", sort=False):
        T = float(grp["T"].iloc[0])
        atm_row = grp["log_moneyness"].abs().idxmin()
        atm_iv = float(grp.loc[atm_row, "iv"])
        if not np.isfinite(atm_iv) or atm_iv <= 0 or T <= 0:
            continue
        out.loc[grp.index] = grp["log_moneyness"] / (atm_iv * np.sqrt(T))
    return out


def _drop_iv_outliers(quotes: pd.DataFrame, n_mad: float = 4.0, min_quotes: int = 10):
    """Remove quotes whose implied vol is far off their own expiry's smile.

    A real smile is smooth in log-moneyness, so a quadratic in x = log(K/F) is
    an adequate local description and anything far from it is a bad print rather
    than a feature.  Necessary here because outside trading hours the only
    available prices are last trades, and in the wings those can be days stale --
    a single such print implies a wildly wrong volatility and, being squared in
    the objective, would otherwise drag the whole calibration toward itself.

    The cut-off is on a median-absolute-deviation scale, not a standard
    deviation, so that the outliers do not inflate the very threshold meant to
    catch them.
    """
    keep = []
    for _, grp in quotes.groupby("expiry", sort=False):
        if len(grp) < min_quotes:
            keep.append(grp)
            continue
        x = grp["log_moneyness"].to_numpy(dtype=float)
        y = grp["iv"].to_numpy(dtype=float)
        resid = y - np.polyval(np.polyfit(x, y, 2), x)
        centre = np.median(resid)
        scale = 1.4826 * np.median(np.abs(resid - centre))
        if scale <= 0:
            keep.append(grp)
            continue
        keep.append(grp[np.abs(resid - centre) <= n_mad * scale])
    return pd.concat(keep).sort_values(["T", "strike"]).reset_index(drop=True)


def build_quotes(
    chain: pd.DataFrame,
    spot: float,
    *,
    fallback_r: float,
    fallback_q: float,
    cp: str = "C",
    min_price: float = 0.10,
    moneyness: tuple[float, float] = (0.80, 1.20),
    max_rel_spread: float = 0.35,
    require_activity: bool = True,
    max_quote_age_days: float = 5.0,
    max_std_moneyness: float = 3.0,
    iv_outlier_mads: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a raw chain into a clean set of quotes for calibration.

    Returns (quotes, filter_report).  The report counts how many contracts each
    filter removed, which belongs in the paper: a calibration is only as
    credible as the screen that produced its inputs.
    """
    df = chain.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["T"] = pd.to_numeric(df["days_to_expiry"], errors="coerce") / 365.0
    price, spread, source = _mid_price(df)
    df["price"] = price
    df["spread"] = spread
    df["price_source"] = source

    last_trade = pd.to_datetime(df["lastTradeDate"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["quote_age_days"] = (now - last_trade).dt.total_seconds() / 86400.0

    steps: list[tuple[str, pd.Series]] = []

    def keep(name: str, mask: pd.Series) -> None:
        steps.append((name, mask))

    keep("valid strike/maturity", df["strike"].gt(0) & df["T"].gt(0))
    keep("finite positive price", np.isfinite(df["price"]) & df["price"].gt(min_price))
    keep(
        f"moneyness in [{moneyness[0]}, {moneyness[1]}]",
        df["strike"].between(moneyness[0] * spot, moneyness[1] * spot),
    )
    rel_spread = df["spread"] / df["price"]
    keep(
        f"relative spread <= {max_rel_spread}",
        rel_spread.isna() | rel_spread.le(max_rel_spread),
    )
    if require_activity:
        vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
        oi = pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0.0)
        two_sided = df["price_source"].eq("mid")
        keep("traded or quoted (volume/OI/two-sided)", (vol > 0) | (oi > 0) | two_sided)
    # Staleness only matters for prices that *are* a past trade.  A resting
    # two-sided quote is the current price of the option no matter when the
    # contract last changed hands, and on a less-traded name like an index
    # option most live quotes sit on contracts that have not traded in days --
    # screening those out would throw away the best data in the set.
    stale = (
        df["price_source"].eq("last")
        & df["quote_age_days"].notna()
        & df["quote_age_days"].gt(max_quote_age_days)
    )
    keep(f"last-trade price fresher than {max_quote_age_days}d", ~stale)

    report = []
    mask = pd.Series(True, index=df.index)
    n_before = int(mask.sum())
    report.append({"filter": "raw contracts", "kept": n_before, "dropped": 0})
    for name, m in steps:
        mask = mask & m.fillna(False)
        n_after = int(mask.sum())
        report.append({"filter": name, "kept": n_after, "dropped": n_before - n_after})
        n_before = n_after

    clean = df[mask].copy()

    # Parity needs both wings; do it before restricting to a single option type.
    fwd = implied_forward_curve(
        clean, spot, fallback_r=fallback_r, fallback_q=fallback_q
    )
    clean = clean.merge(fwd, on=["expiry", "T"], how="left")

    if cp is not None:
        clean = clean[clean.cp == cp].copy()
        report.append(
            {"filter": f"option type == {cp}", "kept": len(clean), "dropped": n_before - len(clean)}
        )
        n_before = len(clean)

    # Everything downstream prices off the forward curve, so `spot` becomes the
    # effective spot the options themselves imply.  The quoted last close is
    # kept alongside it: the gap between the two is a data-freshness
    # diagnostic, not something to average away.
    clean["quoted_spot"] = clean["spot"]
    clean["spot"] = clean["effective_spot"]

    clean["log_moneyness"] = np.log(clean["strike"] / clean["forward"])

    # No-arbitrage band for a European call written on the forward.
    lower = np.maximum(clean["discount"] * (clean["forward"] - clean["strike"]), 0.0)
    upper = clean["discount"] * clean["forward"]
    arb_ok = clean["price"].gt(lower + 1e-8) & clean["price"].lt(upper - 1e-8)
    clean = clean[arb_ok].copy()
    report.append(
        {"filter": "inside no-arbitrage band", "kept": len(clean), "dropped": n_before - len(clean)}
    )
    n_before = len(clean)

    from vg.pricing import implied_vol

    clean["iv"] = [
        implied_vol(
            float(row.price),
            float(row.spot),
            float(row.strike),
            float(row.r),
            float(row.T),
            float(row.q),
        )
        for row in clean.itertuples()
    ]
    iv_ok = clean["iv"].between(0.02, 2.0)
    clean = clean[iv_ok].copy()
    report.append(
        {"filter": "implied vol in [2%, 200%]", "kept": len(clean), "dropped": n_before - len(clean)}
    )
    n_before = len(clean)

    clean["std_moneyness"] = _standardised_moneyness(clean)
    clean = clean[clean["std_moneyness"].abs() <= max_std_moneyness].copy()
    report.append(
        {
            "filter": f"|log(K/F)| <= {max_std_moneyness} ATM sigma*sqrt(T)",
            "kept": len(clean),
            "dropped": n_before - len(clean),
        }
    )
    n_before = len(clean)

    clean = _drop_iv_outliers(clean, n_mad=iv_outlier_mads)
    report.append(
        {
            "filter": f"IV within {iv_outlier_mads} MAD of the expiry's smile",
            "kept": len(clean),
            "dropped": n_before - len(clean),
        }
    )

    cols = [
        "expiry", "T", "days_to_expiry", "cp", "strike", "price", "spread",
        "price_source", "quote_age_days", "volume", "openInterest",
        "spot", "quoted_spot", "forward", "discount", "r", "q",
        "fwd_method", "n_parity_pairs",
        "log_moneyness", "std_moneyness", "iv", "contractSymbol", "snapshot_utc",
    ]
    cols = [c for c in cols if c in clean.columns]
    quotes = clean[cols].sort_values(["T", "strike"]).reset_index(drop=True)
    return quotes, pd.DataFrame(report)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_snapshot(
    chain: pd.DataFrame,
    history: pd.DataFrame,
    meta: MarketSnapshot,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    chain.to_csv(data_dir / meta.chain_file, index=False)
    history.to_csv(data_dir / meta.history_file, index=False)
    meta.to_json(data_dir / "snapshot.json")


def load_snapshot(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, MarketSnapshot]:
    meta = MarketSnapshot.from_json(data_dir / "snapshot.json")
    chain = pd.read_csv(data_dir / meta.chain_file)
    history = pd.read_csv(data_dir / meta.history_file)
    return chain, history, meta

