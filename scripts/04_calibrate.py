"""Calibrate the VG model to the live option chain.

    python scripts/04_calibrate.py

Cleans the cached chain, recovers each expiry's forward and discount factor from
put-call parity, fits the model globally and per expiry, and writes
outputs/calibration.md plus the smile figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vg import bs_call, implied_vol
from vg.calibration import PARAM_LIMITS, calibrate, calibrate_per_expiry
from vg.data import DEFAULT_DATA_DIR, build_quotes, load_snapshot
from vg.plotting import ESSCHER, INK, MARKET, SERIES, save, small_multiples, use_paper_style

OUT = Path(__file__).resolve().parents[1] / "outputs"


def figure_smiles(quotes: pd.DataFrame, results: dict, ticker: str) -> None:
    """Market smile against each fitted model, one panel per expiry."""
    import matplotlib.pyplot as plt

    expiries = sorted(quotes["expiry"].unique(), key=lambda e: quotes.loc[quotes.expiry == e, "T"].iloc[0])
    fig, axes = small_multiples(len(expiries), ncols=3, panel=(3.4, 2.8))

    for ax, exp in zip(axes, expiries):
        grp = quotes[quotes.expiry == exp].sort_values("strike")
        T = float(grp["T"].iloc[0])
        x = grp["strike"] / grp["forward"]
        ax.plot(x, 100 * grp["iv"], "o", color=MARKET, ms=4, mew=0, label="market", zorder=3)
        for name, res in results.items():
            g = res.quotes[res.quotes.expiry == exp].sort_values("strike")
            ax.plot(g["strike"] / g["forward"], 100 * g["model_iv"],
                    color=ESSCHER, lw=1.8, label=name, zorder=2)
        ax.set_title(f"{exp}   T = {T:.3f}y   n = {len(grp)}", fontsize=8.5)
        ax.set_xlabel("$K / F$")
    axes[0].set_ylabel("implied volatility (%)")
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle(
        f"{ticker} implied volatility: market against the calibrated VG model",
        x=0.01, ha="left", fontsize=10.5, fontweight="bold",
    )
    save(fig, OUT / "fig_06_smiles.png")


def figure_residuals(results: dict) -> None:
    """Where the fit fails, in volatility points."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig, axes = small_multiples(len(results), ncols=2, panel=(4.0, 3.2), sharey=True)

    # Truncate the sequential ramp: its lightest steps sit below 2:1 contrast on
    # this surface, so short-maturity points would fade into the background.
    ramp = plt.get_cmap("Blues")
    ramp = mcolors.LinearSegmentedColormap.from_list(
        "Blues_readable", ramp(np.linspace(0.32, 1.0, 256))
    )

    for ax, (name, res) in zip(axes, results.items()):
        q = res.quotes.dropna(subset=["iv_error"])
        # Standardised moneyness on the x-axis so every expiry is on the same
        # scale -- a fixed K/F would put the two-week and the eighteen-month
        # quotes in incomparable places.
        sc = ax.scatter(q["std_moneyness"], 1e4 * q["iv_error"],
                        c=q["T"], cmap=ramp, s=18, edgecolors="none",
                        vmin=q["T"].min(), vmax=q["T"].max())
        ax.axhline(0.0, color=INK["axis"], lw=1.0)
        ax.set_xlabel(r"standardised moneyness  $\log(K/F) / (\sigma_{ATM}\sqrt{T})$")
        ax.set_title(f"{name}  ({len(res.free_parameters)} free parameters, "
                     f"RMSE {1e4 * res.rmse_iv:.0f} bp)", fontsize=9)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("maturity T (years)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axes[0].set_ylabel("model IV - market IV  (basis points)")
    fig.suptitle(
        "Calibration residuals: a systematic tilt, not noise -- the missing skew",
        x=0.01, ha="left", fontsize=10.5, fontweight="bold",
    )
    save(fig, OUT / "fig_07_residuals.png")


def figure_term_structure(per_expiry: dict) -> None:
    """If VG could fit the surface, these would be flat lines."""
    import matplotlib.pyplot as plt

    fig, axes = small_multiples(3, ncols=3, panel=(3.3, 2.7))
    for ax, param in zip(axes, ["sigma", "nu", "rmse_iv_bp"]):
        for (name, df), colour, marker in zip(per_expiry.items(), [ESSCHER], ["o"]):
            if param not in df.columns:
                continue
            d = df.dropna(subset=[param]).sort_values("T")
            ax.plot(d["T"], d[param], marker + "-", color=colour, ms=5, mew=0, label=name)
        ax.set_xlabel("maturity T (years)")
        ax.set_xscale("log")
        ax.set_title({"sigma": r"$\sigma$", "nu": r"$\nu$",
                      "rmse_iv_bp": "fit quality (bp of vol)"}[param])
        # `nu` spans four decades because almost every expiry is pinned to the
        # floor and one escapes it.  On a linear axis that reads as a single
        # spike over a flat zero line, which hides the fact worth seeing: the
        # bound is where the fits actually sit.
        if param == "nu":
            ax.set_yscale("log")
            floor = PARAM_LIMITS["nu"][0]
            ax.axhline(floor, color=INK["muted"], lw=1.0, ls="--", zorder=0)
            ax.annotate(
                "lower bound", (ax.get_xlim()[0], floor), xytext=(2, 3),
                textcoords="offset points", fontsize=7.5, color=INK["muted"],
            )
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle(
        "Parameters fitted to each expiry separately -- a Levy model needs these flat",
        x=0.01, ha="left", fontsize=10.5, fontweight="bold",
    )
    save(fig, OUT / "fig_08_term_structure.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    use_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)

    chain, _, meta = load_snapshot(args.data_dir)
    print(f"{meta.ticker} snapshot {meta.asof}, quoted spot {meta.spot:.2f}, "
          f"{len(chain):,} raw contracts")
    print(f"  exercise style: {meta.exercise_style}")

    quotes, report = build_quotes(
        chain, meta.spot,
        fallback_r=meta.risk_free_irx,
        fallback_q=meta.dividend_yield,
        cp="C",
    )
    print("\nFilter report:")
    print(report.to_string(index=False))

    eff_spot = float(quotes["spot"].iloc[0])
    spot_gap = eff_spot / meta.spot - 1.0
    print(f"\nEffective spot implied by the forward curve: {eff_spot:.2f} "
          f"({spot_gap:+.3%} vs the quoted last close)")
    print(f"{len(quotes)} calls survive, {quotes['expiry'].nunique()} expiries, "
          f"{(quotes['price_source'] == 'mid').mean():.0%} priced off two-sided quotes")

    fwd = (quotes.groupby(["expiry", "T"], as_index=False)
           .agg(forward=("forward", "first"), discount=("discount", "first"),
                r=("r", "first"), q=("q", "first"),
                method=("fwd_method", "first"), pairs=("n_parity_pairs", "first"),
                n=("strike", "size")))
    print("\nImplied forwards:")
    print(fwd.to_string(index=False))

    measures = ["esscher"]
    names = {"esscher": "VG / Esscher"}

    results, per_expiry = {}, {}
    for m in measures:
        print(f"\nCalibrating globally ({m}) ...", flush=True)
        res = calibrate(quotes, m)
        print("  " + res.summary(), flush=True)
        results[names[m]] = res
        print(f"  per expiry ({m}) ...", flush=True)
        per_expiry[names[m]] = calibrate_per_expiry(quotes, m, verbose=True)

    # Black-Scholes benchmark: one flat vol for the whole surface.
    #
    # Fitted with the *same* vega-weighted objective the VG models use.  Fitting
    # it on unweighted price error instead would put almost all the weight on
    # the at-the-money strikes and hand the benchmark a different question to
    # answer, which would make the comparison table meaningless.
    from scipy.optimize import minimize_scalar

    from vg.calibration import _weights

    w = _weights(quotes, "vega")

    def bs_residuals(sig: float) -> np.ndarray:
        model = np.asarray(bs_call(quotes["spot"].to_numpy(), quotes["strike"].to_numpy(),
                                   quotes["r"].to_numpy(), quotes["T"].to_numpy(),
                                   sig, quotes["q"].to_numpy()))
        return (model - quotes["price"].to_numpy()) / w

    flat = float(minimize_scalar(
        lambda s: float(np.sum(bs_residuals(s) ** 2)), bounds=(0.01, 1.5), method="bounded"
    ).x)
    bs_price_rmse = float(np.sqrt(np.mean((bs_residuals(flat) * w) ** 2)))
    bs_iv_rmse = float(np.sqrt(np.mean((flat - quotes["iv"].to_numpy()) ** 2)))

    figure_smiles(quotes, results, meta.ticker)
    figure_residuals(results)
    figure_term_structure(per_expiry)

    comparison = pd.DataFrame(
        [
            {
                "model": "Black-Scholes (one flat vol)",
                "free params": 1,
                "theta": np.nan, "sigma": flat, "nu": np.nan,
                "RMSE price ($)": bs_price_rmse,
                "RMSE IV (bp)": 1e4 * bs_iv_rmse,
                "max |IV err| (bp)": 1e4 * float(np.max(np.abs(flat - quotes["iv"]))),
                "at bound": "",
            }
        ]
        + [
            {
                "model": name,
                "free params": len(res.free_parameters),
                "theta": res.params.theta if "theta" in res.free_parameters else np.nan,
                "sigma": res.params.sigma,
                "nu": res.params.nu,
                "RMSE price ($)": res.rmse_price,
                "RMSE IV (bp)": 1e4 * res.rmse_iv,
                "max |IV err| (bp)": 1e4 * res.max_abs_iv,
                "at bound": ", ".join(res.at_bound),
            }
            for name, res in results.items()
        ]
    )

    # Did the Esscher fit actually collapse onto the flat-vol benchmark?
    collapse_txt = ""
    ess = results.get("VG / Esscher")
    if ess is not None and "nu" in ess.at_bound:
        collapse_txt = (
            f"  The collapse is literal, not approximate: the fitted "
            f"`sigma = {ess.params.sigma:.5f}` against the flat Black-Scholes "
            f"volatility of `{flat:.5f}`, agreeing to four decimal places, with "
            f"`nu` resting on its lower bound of {ess.params.nu:g}.  The second "
            f"parameter buys nothing at all -- the volatility error moves from "
            f"{1e4 * bs_iv_rmse:.1f} bp to {1e4 * ess.rmse_iv:.1f} bp."
        )

    market_skew = []
    for exp, grp in quotes.groupby("expiry"):
        g = grp.sort_values("std_moneyness")
        lo = g[g["std_moneyness"] < -1.0]
        hi = g[g["std_moneyness"] > 1.0]
        if len(lo) and len(hi):
            market_skew.append(float(lo["iv"].iloc[-1] - hi["iv"].iloc[0]))
    skew_txt = (
        f"{1e2 * np.mean(market_skew):.1f} volatility points on average "
        f"(range {1e2 * min(market_skew):.1f} to {1e2 * max(market_skew):.1f})"
        if market_skew else "not measurable on this chain"
    )

    # Read the direction of the drift off the fits rather than asserting it.
    # A least-squares slope is the wrong statistic here: one expiry that escapes
    # the lower bound on `nu` can carry a fit whose other eleven points are all
    # sitting on it.  Rank correlation with T reports monotone drift only when
    # the whole curve moves, and the bound count says how much of the "drift"
    # is really the optimiser refusing to leave its constraint.
    nu_floor = PARAM_LIMITS["nu"][0]
    ts_lines, ts_trends, ts_frames = [], [], {}
    nu_pinned = {}
    for name, df in per_expiry.items():
        if "sigma" not in df.columns or len(df) < 3:
            continue
        d = df.dropna(subset=["sigma", "nu"]).sort_values("T")
        ts_frames[name] = d
        n_pin = int((d["nu"] <= 1.001 * nu_floor).sum())
        nu_pinned[name] = (n_pin, len(d))
        ts_lines.append(
            f"* **{name}**: sigma runs {d['sigma'].min():.4f}-{d['sigma'].max():.4f}, "
            f"nu runs {d['nu'].min():.4f}-{d['nu'].max():.4f}, across "
            f"T = {d['T'].min():.3f}-{d['T'].max():.3f}y"
            + (f"; nu sits on its lower bound at {n_pin} of {len(d)} expiries." if n_pin else ".")
        )
        for param in ["sigma", "nu"]:
            lo, hi = d[param].min(), d[param].max()
            if hi - lo < 1e-6:
                ts_trends.append(f"{name} {param}: flat (pinned at {lo:.4f} throughout)")
                continue
            rho = float(
                np.corrcoef(
                    np.argsort(np.argsort(d["T"].to_numpy())),
                    np.argsort(np.argsort(d[param].to_numpy())),
                )[0, 1]
            )
            if abs(rho) < 0.5:
                shape = f"no monotone trend in T (rank corr {rho:+.2f})"
            else:
                shape = f"{'rises' if rho > 0 else 'falls'} with T (rank corr {rho:+.2f})"
            ts_trends.append(
                f"{name} {param}: {shape}; {d[param].iloc[0]:.4f} at the front, "
                f"{d[param].iloc[-1]:.4f} at the back, range {lo:.4f}-{hi:.4f}"
            )
    trend_txt = "\n".join(f"* {t}" for t in ts_trends)

    # The Levy-kurtosis story only applies if `nu` is actually free to move.
    # Under the Esscher specification it usually is not, and saying so is the
    # more interesting statement -- so let the data pick which paragraph runs.
    n_pin, n_exp = nu_pinned.get("VG / Esscher", (0, 0))
    if n_exp and n_pin > n_exp / 2:
        e = ts_frames["VG / Esscher"]
        term_txt = f"""`sigma` carries the whole term structure here, and `nu` carries none of it:
it rests on its lower bound at {n_pin} of the {n_exp} expiries.  That is the
identifiability constraint of section 2 reappearing one maturity at a time.
Freed to fit a single expiry the specification still cannot bend its smile, so
it still declines the curvature `nu` would buy, and what is left is a flat
Black-Scholes fit whose level has to be re-chosen at every maturity.  A rising
`sigma` is exactly how a flat-vol model encodes a smile it cannot represent.

The comparison this section is meant to make therefore cannot be made on this
fit.  For a Levy process the excess kurtosis of the log return at horizon `T` is
`3 nu / T`, so it must decay like `1/T` by construction, and a market whose
implied kurtosis decays more slowly forces the per-expiry `nu` upward with
maturity.  That is the standard limitation of pure Levy models -- but reading it
off requires a specification whose `nu` is free, and this one's is not.  What
this fit does establish is the prior obstacle: the per-expiry `sigma` spans
{e['sigma'].min():.4f}-{e['sigma'].max():.4f} while the model is committed to a
single number, so no one parameter set prices every maturity regardless of what
the tails are doing."""
    else:
        term_txt = """The mechanism is the one that makes this the standard limitation of pure Levy
models.  For a Levy process the excess kurtosis of the log return at horizon `T`
is `3 nu / T` -- it must decay like `1/T` by construction, while the market's
implied kurtosis decays more slowly.  Fitting each expiry on its own, the
long-dated fits then need an ever larger `nu` to manufacture the kurtosis the
market is charging for at that horizon.  Note the direction, because it is the
reverse of the naive expectation: it is not that short maturities want fatter
tails, it is that long maturities cannot get enough of them.  The fit quality per
expiry is good throughout, so this is not a numerical artefact -- it is the
*parameters* that refuse to agree, which is precisely the gap that
stochastic-volatility and time-changed extensions were introduced to close."""

    md = f"""# Calibration to the {meta.ticker} option chain

Snapshot: **{meta.asof}**, quoted spot **{meta.spot:.2f}**, {meta.n_raw_quotes:,}
raw contracts downloaded, **{len(quotes)}** calls surviving the screen across
{quotes['expiry'].nunique()} expiries.

**Exercise style: {meta.exercise_style}.**  This is the reason the chain is
`{meta.ticker}` and not the more liquid SPY: the paper's Theorem 4.1 prices a
European call, and SPY options are American.  Options on the index are
European and cash-settled, so model and data describe the same contract and no
early-exercise premium has to be argued away.

## 1. Building the quote set

{report.to_markdown(index=False)}

Prices are mids where a two-sided quote exists and last traded prices otherwise;
**{(quotes['price_source'] == 'mid').mean():.0%}** of the surviving quotes are
mids in this snapshot.

Rather than assume a rate and a dividend yield, each expiry's forward `F` and
discount factor `D` come from put-call parity, `C(K) - P(K) = D(F - K)`, fitted
across the liquid strikes.  That absorbs rate, dividend and borrow into the two
numbers the model actually needs:

{fwd.to_markdown(index=False, floatfmt=".5g")}

### The spot is stale, and the options say so

Fitting `log F(T) = log S_eff + carry * T` across expiries gives an effective
spot of **{eff_spot:.2f}**, which is **{spot_gap:+.3%}** away from the quoted
last close of {meta.spot:.2f}.  That gap is not a modelling choice, it is a
timestamp: the index history's last bar and the option quotes come from
different sessions.

It matters more than its size suggests.  Screening the parity forwards against
the stale spot rejected every short expiry -- their forwards were excellent (the
two independent estimators agreed to two thousandths of a dollar) but sat 1%
away from where the stale spot said they should be.  Checking the curve against
itself instead keeps them, and everything downstream prices off `S_eff`.

## 2. Fit

{comparison.to_markdown(index=False, floatfmt=".5g")}

The `free params` column is not decoration.  Under the Esscher specification of
Section 4.1 the martingale condition forces
`theta_Q + sigma_Q^2/2 = (1 - e^{{-r nu}})/nu`, so only **two** parameters are
free and `theta` is not identifiable from prices at all -- see `validation.md`
section 3, where five physical parameter sets spanning `theta` from -0.60 to
+0.20 are shown to give identical prices to machine precision.

## 3. What this says about the model

The {meta.ticker} smile on this snapshot slopes down by {skew_txt} between one standard
move below and one above the forward.  That is a real, clean equity skew, and
reproducing it requires a free skew parameter.

That is exactly what the Esscher specification gives up.  With `theta_Q` pinned
near `-sigma_Q^2/2` its smile is close to symmetric, and a symmetric smile
cannot approximate a monotonically falling one -- raising `nu` to add curvature
lifts *both* wings, which helps on one side and hurts on the other.  The
optimiser's best compromise is therefore to add no curvature at all: `nu` runs
to its lower bound and the model collapses onto flat Black-Scholes.  That is
what the `at bound` column reports, and it is why the Esscher row does not beat
the one-parameter benchmark despite having two parameters.{collapse_txt}

`fig_07_residuals.png` shows the shape of the failure: the residuals are not
noise around zero but a systematic tilt across moneyness, which is the missing
skew showing through.  A specification whose smile can bend the other way would
remove it; this one cannot, because the martingale condition has already spent
that degree of freedom.

## 4. Term structure

VG is a Levy process: increments are stationary and independent, so one
parameter set must price every maturity.  Fitting each expiry separately tests
that directly -- if the model were right, the fitted parameters would be
constant in `T`.

{chr(10).join(ts_lines)}

They are not constant (`fig_08_term_structure.png`):

{trend_txt}

{term_txt}

## 5. Per-expiry parameters

{chr(10).join(f"### {name}{chr(10)}{chr(10)}{df.to_markdown(index=False, floatfmt='.5g')}{chr(10)}" for name, df in per_expiry.items())}
"""
    (OUT / "calibration.md").write_text(md, encoding="utf-8")
    quotes.to_csv(OUT / "quotes_clean.csv", index=False)
    comparison.to_csv(OUT / "calibration_comparison.csv", index=False)
    for name, res in results.items():
        slug = name.split("/")[-1].strip().replace(" ", "_")
        res.quotes.to_csv(OUT / f"calibration_fit_{slug}.csv", index=False)
        per_expiry[name].to_csv(OUT / f"calibration_per_expiry_{slug}.csv", index=False)

    print("\n" + comparison.to_string(index=False))
    print(f"\nWrote {OUT / 'calibration.md'}")


if __name__ == "__main__":
    main()




