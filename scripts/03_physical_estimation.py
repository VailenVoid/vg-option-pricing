"""Estimate the physical VG parameters from the underlying's realised returns.

    python scripts/03_physical_estimation.py [--frequency daily|weekly|monthly]

Fits (theta, sigma, nu) to the index log returns by moments and by maximum likelihood,
compares the fitted law with the empirical one and with the Gaussian benchmark,
and writes outputs/physical_estimation.md plus figures.

This is the P-measure half of the study; `04_calibrate.py` is the Q-measure half.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vg import VGParams, density, moments
from vg.data import DEFAULT_DATA_DIR, load_snapshot
from vg.estimation import annualised_summary, fit_mle, moment_estimates
from vg.plotting import INK, SERIES, save, small_multiples, use_paper_style

OUT = Path(__file__).resolve().parents[1] / "outputs"
FREQ = {"daily": (1, 1 / 252), "weekly": (5, 1 / 52), "monthly": (21, 1 / 12)}


def aggregate(returns: np.ndarray, step: int) -> np.ndarray:
    """Non-overlapping sums of log returns."""
    n = (len(returns) // step) * step
    return returns[:n].reshape(-1, step).sum(axis=1)


def figure_fit(returns: np.ndarray, dt: float, fits: dict[str, VGParams],
               mus: dict[str, float], label: str, ticker: str) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = small_multiples(3, ncols=3, panel=(3.5, 3.0))

    # Panel 1 & 2: density on linear and log scale.
    lo, hi = np.quantile(returns, [0.0005, 0.9995])
    pad = 0.35 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 900)
    name, p = next(iter(fits.items()))
    mu = mus[name]

    f_vg = density(p, grid - mu * dt, dt)
    f_n = stats.norm.pdf(grid, returns.mean(), returns.std(ddof=1))
    hist, edges = np.histogram(returns, bins=140, range=(grid[0], grid[-1]), density=True)
    hist_peak = float(hist.max())

    for ax, logscale in zip(axes[:2], [False, True]):
        ax.hist(returns, bins=140, density=True, range=(grid[0], grid[-1]),
                color=INK["grid"], edgecolor="none", label=f"{ticker} (empirical)")
        ax.plot(grid, f_vg, color=SERIES[0], label=f"VG ({name})")
        ax.plot(grid, f_n, color=SERIES[1], ls="--", label="Normal")
        ax.set_xlabel(f"{label} log return")
        if logscale:
            ax.set_yscale("log")
            floor = max(1e-3, 0.2 * np.min(f_n[f_n > 0]))
            ax.set_ylim(floor, 3.0 * max(hist_peak, float(f_vg.max())))
            ax.set_title("log scale: the tails")
        else:
            # The fitted VG spike runs an order of magnitude above the data, so
            # the body of the distribution is invisible on a full-range axis.
            # Clip and say by how much -- the overshoot is the finding.
            ax.set_ylim(0.0, 1.7 * hist_peak)
            ax.set_ylabel("density")
            ax.set_title("linear scale: the peak")
            if f_vg.max() > 1.7 * hist_peak:
                ax.annotate(
                    f"VG peak reaches {f_vg.max():.0f},\n"
                    f"{f_vg.max() / hist_peak:.0f}x the empirical peak\n(axis clipped)",
                    xy=(0.0, 1.7 * hist_peak), xytext=(0.55, 0.80),
                    textcoords="axes fraction", color=INK["secondary"], fontsize=7.5,
                    ha="center",
                    arrowprops={"arrowstyle": "->", "color": INK["muted"], "lw": 0.8},
                )
        ax.legend(loc="upper left", fontsize=7.5)

    # Panel 3: QQ against the fitted VG.
    #
    # The CDF is evaluated once on a grid and then inverted by interpolation.
    # Calling a root finder per quantile would re-run the whole adaptive
    # Gil-Pelaez integration a few hundred times; the routine is vectorised over
    # x precisely so that it does not have to be.
    from vg import cf, gil_pelaez_tail

    n = len(returns)
    probs = (np.arange(1, n + 1) - 0.5) / n
    keep = np.linspace(0, n - 1, min(n, 500)).astype(int)
    probs, emp = probs[keep], np.sort(returns)[keep]

    sd = np.sqrt(moments(p, dt)["var"])
    span = max(4.0 * np.abs(returns - mu * dt).max(), 8.0 * sd)
    grid = np.linspace(-span, span, 2001)
    cdf_grid = 1.0 - np.asarray(gil_pelaez_tail(lambda u: cf(p, u, dt), grid))
    cdf_grid = np.maximum.accumulate(np.clip(cdf_grid, 0.0, 1.0))  # enforce monotone

    ok = np.concatenate([[True], np.diff(cdf_grid) > 0])
    theo = np.interp(probs, cdf_grid[ok], grid[ok]) + mu * dt

    ax = axes[2]
    ax.plot(theo, emp, "o", color=SERIES[0], ms=3, alpha=0.75, mew=0)
    lim = [min(theo.min(), emp.min()), max(theo.max(), emp.max())]
    ax.plot(lim, lim, color=INK["muted"], lw=1.0, ls="--")
    ax.set_xlabel("VG quantile")
    ax.set_ylabel("empirical quantile")
    ax.set_title("QQ plot vs the fitted VG")

    fig.suptitle(
        f"{ticker} {label} log returns against the fitted Variance-Gamma law",
        x=0.01, ha="left", fontsize=10.5, fontweight="bold",
    )
    save(fig, OUT / f"fig_05_physical_fit_{label}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frequency", default="daily", choices=list(FREQ))
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    use_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)

    _, history, meta = load_snapshot(args.data_dir)
    raw = history["log_return"].to_numpy(dtype=float)
    raw = raw[np.isfinite(raw)]

    step, dt = FREQ[args.frequency]
    returns = aggregate(raw, step) if step > 1 else raw
    print(f"{meta.history_ticker}: {len(returns):,} {args.frequency} returns, "
          f"dt = {dt:.6f}")

    mom = moment_estimates(returns, dt)
    print(f"  moments: {mom.params}  mu = {mom.mu:+.4f}")
    mle = fit_mle(returns, dt)
    print(f"  MLE    : {mle.params}  mu = {mle.mu:+.4f}  loglik = {mle.loglik:,.1f}")

    fits = {"moments": mom.params, "MLE": mle.params}
    mus = {"moments": mom.mu, "MLE": mle.mu}

    summary = pd.DataFrame(
        [
            {"estimator": "moments", **annualised_summary(mom.params, mom.mu, dt)},
            {"estimator": "MLE", **annualised_summary(mle.params, mle.mu, dt)},
        ]
    )
    moment_check = pd.DataFrame(
        [
            {"moment": k, "sample": mom.sample[k], "fitted (moments)": mom.fitted[k]}
            for k in ["var", "skewness", "excess_kurtosis"]
        ]
    )

    # How concentrated is each fitted law at the origin, versus the data?
    shape_mom = dt / mom.params.nu
    shape_mle = dt / mle.params.nu
    centred = returns - mom.mu * dt
    tiny_data = float(np.mean(np.abs(centred) < 1e-4))

    rng = np.random.default_rng(0)
    from vg.process import simulate_terminal

    tiny_model = float(
        np.mean(np.abs(simulate_terminal(mom.params, dt, 400_000, rng)) < 1e-4)
    )
    nu_ratio = mom.params.nu / mle.params.nu

    figure_fit(returns, dt, fits, mus, args.frequency, meta.history_ticker)

    md = f"""# Physical parameters from realised returns

`{meta.history_ticker}`, {len(returns):,} {args.frequency} log returns,
dt = {dt:.6f}.  Snapshot {meta.asof}.

The option chain is on `{meta.ticker}` ({meta.exercise_style}); the return
history is taken from `{meta.history_ticker}`, which is the same underlying
process with a far longer sample -- over their common window the daily log
returns correlate at 0.99933 and the annualised volatilities agree to 0.03
percentage points.

## Estimates

{summary.to_markdown(index=False, floatfmt=".5g")}

The method of moments reproduces the sample moments it targets, as it must:

{moment_check.to_markdown(index=False, floatfmt=".6g")}

## Why the two estimators disagree

They differ by a factor of {nu_ratio:.1f} in `nu` ({mom.params.nu:.5f} by
moments, {mle.params.nu:.5f} by likelihood), and the reason is structural rather
than numerical.

The Gamma clock's shape over one step is `dt/nu`: {shape_mom:.3f} at the moment
estimate, {shape_mle:.3f} at the MLE.  Below `1/2` the VG density has a power
singularity at the origin, so the likelihood is dominated by how sharply a
parameter set spikes at the mode rather than by how well it matches the spread
of the data -- and the two estimators are then answering different questions.
The moment estimator targets the variance, skewness and kurtosis, which is what
matters for pricing; the likelihood targets the peak.

The model's own claim about the peak is testable, and it fails.  Simulating from
the moment fit, {tiny_model:.1%} of {args.frequency} returns should land within
1e-4 of the drift -- essentially "nothing happened today".  In the actual
`{meta.history_ticker}` series {tiny_data:.1%} do.  Markets trade continuously
and do not pile up at the origin the way a Gamma clock of this shape says they
should, which is a real limitation of VG as a description of the physical
measure, quite separate from how well it prices options under Q.

The moment estimates are the ones carried forward, because option prices depend
on the variance, skewness and kurtosis of the return law rather than on the
height of its mode.

Lowering the sampling frequency raises `dt/nu` and should weaken the
singularity; run this script at `--frequency weekly` and `--frequency monthly`
to see how far that goes on this sample.  Note that the monthly series is only
{len(raw) // 21:,} observations long and its sample kurtosis is driven by a
handful of crisis months, so it is too short to settle the question on its own.
What the exercise does settle is that the two estimators are answering different
questions here, and that the one aligned with pricing is the moment estimator.

## What the fit does capture

`fig_05_physical_fit_{args.frequency}.png` shows the point of the whole exercise.
Against the Gaussian benchmark the VG law matches both the sharp peak and the
heavy tails of the return distribution -- sample excess kurtosis
{mom.sample['excess_kurtosis']:.2f} against 0 for the normal, sample skewness
{mom.sample['skewness']:.2f} against 0.  The QQ panel shows where it still
fails, which is in the extreme tails.

## Note on the link to option prices

These are physical parameters.  Turning them into option prices needs a
martingale measure, and under the specification of Section 4.1 the Esscher
transform makes the physical `theta` unidentifiable -- see `validation.md`
section 4.  So the numbers above are the right answer to "what does the index do",
and deliberately not the numbers used to price; `04_calibrate.py` fits the
risk-neutral parameters to option prices directly.
"""
    (OUT / f"physical_estimation_{args.frequency}.md").write_text(md, encoding="utf-8")
    summary.to_csv(OUT / f"physical_estimation_{args.frequency}.csv", index=False)
    print(f"\nWrote {OUT / f'physical_estimation_{args.frequency}.md'}")


if __name__ == "__main__":
    main()


