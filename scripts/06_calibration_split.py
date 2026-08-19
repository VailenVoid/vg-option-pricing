"""Calibration scored out of sample: fit on part of the surface, test on the rest.

    python scripts/06_calibration_split.py

Why this script exists
----------------------
A calibrated model reprices the quotes it was fitted to almost by construction,
so an in-sample calibration error measures *flexibility*, not accuracy.  Quoting
one as if it were a prediction error overstates the model, and comparing it with
a model estimated from historical returns overstates it twice over: only one of
the two has seen the answer.

So every model here gets identical treatment -- same training quotes, same
vega-weighted objective, same held-out test quotes -- and is scored on data it
was not fitted to.  Black-Scholes with a single volatility is included on
exactly those terms, fitted rather than estimated, so that the ranking is a
statement about the models and not about their information sets.

Three splits, because each one probes a different limitation:

  * **interleaved strikes** -- every other strike within each expiry.  The easy
    case: the model only has to interpolate between neighbouring quotes.  A
    model that cannot pass this is broken.
  * **at-the-money to wings** -- fit within one standard move of the forward,
    test outside it.  This probes the *shape* of the smile, which is where the
    Esscher specification is known to be constrained.
  * **short maturities to long** -- fit the near expiries, test the far ones.
    This probes the term structure, where a Levy model is forced to let excess
    kurtosis decay like 1/T whatever the market does.

Writes outputs/calibration_split.md and fig_10.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vg import bs_call, implied_vol
from vg.calibration import calibrate, calibrate_flat_vol, price_quotes
from vg.data import DEFAULT_DATA_DIR, build_quotes, load_snapshot
from vg.plotting import INK, SERIES, save, small_multiples, use_paper_style

OUT = Path(__file__).resolve().parents[1] / "outputs"


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def split_interleaved(quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every other strike within each expiry."""
    train_idx = []
    for _, grp in quotes.groupby("expiry", sort=False):
        ordered = grp.sort_values("strike").index
        train_idx.extend(ordered[::2])
    train = quotes.loc[quotes.index.isin(train_idx)]
    return train, quotes.loc[~quotes.index.isin(train_idx)]


def split_atm_to_wings(quotes: pd.DataFrame, cut: float = 1.0):
    """Fit within `cut` standard moves of the forward, test outside."""
    inside = quotes["std_moneyness"].abs() <= cut
    return quotes[inside], quotes[~inside]


def split_short_to_long(quotes: pd.DataFrame):
    """Fit the near half of the expiries, test the far half."""
    expiries = quotes.groupby("expiry")["T"].first().sort_values()
    near = set(expiries.index[: len(expiries) // 2])
    is_near = quotes["expiry"].isin(near)
    return quotes[is_near], quotes[~is_near]


SPLITS = {
    "interleaved strikes": split_interleaved,
    "ATM -> wings": split_atm_to_wings,
    "short -> long maturity": split_short_to_long,
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def model_prices(quotes: pd.DataFrame, kind: str, fit) -> np.ndarray:
    if kind == "black_scholes":
        return np.asarray(
            bs_call(quotes["spot"].to_numpy(), quotes["strike"].to_numpy(),
                    quotes["r"].to_numpy(), quotes["T"].to_numpy(),
                    fit, quotes["q"].to_numpy())
        )
    return price_quotes(quotes, fit, kind)


def score(quotes: pd.DataFrame, predicted: np.ndarray) -> dict:
    err = predicted - quotes["price"].to_numpy()
    ivs = np.array([
        implied_vol(float(v), float(row.spot), float(row.strike),
                    float(row.r), float(row["T"]), float(row.q))
        for v, (_, row) in zip(predicted, quotes.iterrows())
    ])
    iv_err = ivs - quotes["iv"].to_numpy()
    iv_err = iv_err[np.isfinite(iv_err)]
    return {
        "n": len(quotes),
        "MSE": float(np.mean(err**2)),
        "RMSE ($)": float(np.sqrt(np.mean(err**2))),
        "MAE ($)": float(np.mean(np.abs(err))),
        "RMSE (vol pts)": float(100 * np.sqrt(np.mean(iv_err**2))) if iv_err.size else np.nan,
    }


def figure(rows: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    splits = list(SPLITS)
    fig, axes = small_multiples(len(splits), ncols=3, panel=(3.6, 3.1), sharey=True)
    models = ["black_scholes", "esscher"]
    labels = {"black_scholes": "Black-Scholes\n(1 param)",
              "esscher": "VG / Esscher\n(2 params)"}
    width = 0.36
    x = np.arange(len(models))

    for ax, split in zip(axes, splits):
        d = rows[rows["split"] == split].set_index("model")
        ins = [d.loc[m, "train RMSE (vol pts)"] for m in models]
        oos = [d.loc[m, "test RMSE (vol pts)"] for m in models]
        ax.bar(x - width / 2, ins, width, color=INK["grid"], label="in-sample (fitted)")
        ax.bar(x + width / 2, oos, width, color=SERIES[0], label="out-of-sample (held out)")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in models], fontsize=7.5)
        ax.set_title(split, fontsize=9)
        for xi, v in zip(x + width / 2, oos):
            ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom",
                        fontsize=7.5, color=INK["secondary"])
    axes[0].set_ylabel("RMSE (volatility points)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(
        "Every model fitted the same way, then scored on quotes it never saw",
        x=0.01, ha="left", fontsize=10.5, fontweight="bold",
    )
    save(fig, OUT / "fig_10_calibration_split.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    use_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)

    chain, _, meta = load_snapshot(args.data_dir)
    quotes, _ = build_quotes(chain, meta.spot,
                             fallback_r=meta.risk_free_irx,
                             fallback_q=meta.dividend_yield, cp="C")
    print(f"{len(quotes)} European calls on {meta.ticker}, "
          f"{quotes['expiry'].nunique()} expiries\n")

    rows, fits = [], {}
    for split_name, splitter in SPLITS.items():
        train, test = splitter(quotes)
        print(f"{split_name}: {len(train)} train / {len(test)} test", flush=True)

        for kind in ["black_scholes", "esscher"]:
            if kind == "black_scholes":
                fit = calibrate_flat_vol(train)
                described = f"sigma={fit:.4f}"
                n_params = 1
            else:
                res = calibrate(train, kind, starts=None)
                fit = res.params
                described = str(fit)
                n_params = len(res.free_parameters)

            tr = score(train, model_prices(train, kind, fit))
            te = score(test, model_prices(test, kind, fit))
            print(f"    {kind:<14} {described:<44} "
                  f"train {tr['RMSE (vol pts)']:6.2f} -> test {te['RMSE (vol pts)']:6.2f} vol pts",
                  flush=True)

            rows.append({
                "split": split_name,
                "model": kind,
                "free params": n_params,
                "fitted": described,
                "n train": tr["n"],
                "n test": te["n"],
                "train RMSE ($)": tr["RMSE ($)"],
                "test RMSE ($)": te["RMSE ($)"],
                "test MSE": te["MSE"],
                "test MAE ($)": te["MAE ($)"],
                "train RMSE (vol pts)": tr["RMSE (vol pts)"],
                "test RMSE (vol pts)": te["RMSE (vol pts)"],
            })
            fits[(split_name, kind)] = fit
        print()

    table = pd.DataFrame(rows)
    figure(table)

    degradation = table.assign(
        ratio=table["test RMSE (vol pts)"] / table["train RMSE (vol pts)"]
    )

    md = f"""# Calibration scored out of sample

{len(quotes)} European calls on `{meta.ticker}`, {quotes['expiry'].nunique()}
expiries.  Snapshot {meta.asof}.

## The point of this table

A calibrated model reprices its own training quotes nearly by construction, so
an in-sample calibration error measures how *flexible* a model is, not how
*accurate*.  Every model below is therefore fitted on one part of the surface
and scored on a part it never saw -- and Black-Scholes is **fitted** to option
prices on identical terms rather than estimated from historical returns, so the
comparison is between models rather than between information sets.

That distinction is not pedantry.  Fit a model by minimising squared error
against option prices, report squared error against those same prices, and
compare it with a model estimated from returns, and the fitted model wins
whatever it is -- the result is a property of the setup, not of the model.

## Results

{table.to_markdown(index=False, floatfmt='.4g')}

## Reading it

The gap between the `train` and `test` columns is the honest cost of
calibration:

{degradation[['split', 'model', 'train RMSE (vol pts)', 'test RMSE (vol pts)', 'ratio']].to_markdown(index=False, floatfmt='.4g')}

* **Interleaved strikes** is the easy split -- neighbouring strikes carry almost
  the same information, so any model that fits in-sample also fits out. A model
  failing here would be broken.
* **ATM to wings** asks the model to extrapolate the smile's *shape* outward
  from the money.  This is where a specification that cannot bend pays for it.
* **Short to long maturity** asks it to extrapolate the *term structure*, which
  a Levy model cannot do freely: its excess kurtosis must decay like `1/T`.

The three splits together say more than any single number, and none of them is
the number a calibration paper usually quotes.

## The full picture

Three comparisons, all now on the same footing:

| what is compared | question answered | where |
|---|---|---|
| both estimated from returns | which **predicts** better? | `validation_out_of_sample.md` |
| both calibrated, scored in sample | which is more **flexible**? | `calibration.md` |
| both calibrated, scored out of sample | which **generalises**? | this file |

Only the first and third are tests of a model.  The second is a test of a
parametrisation, and it is the one most often reported as if it were the first.
"""
    (OUT / "calibration_split.md").write_text(md, encoding="utf-8")
    table.to_csv(OUT / "calibration_split.csv", index=False)

    print(table[["split", "model", "free params", "train RMSE (vol pts)",
                 "test RMSE (vol pts)"]].to_string(index=False))
    print(f"\nWrote {OUT / 'calibration_split.md'}")


if __name__ == "__main__":
    main()
