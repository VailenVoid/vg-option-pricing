"""Risk-neutral calibration of the VG model to a market option chain.

The model has three parameters and the chain has hundreds of quotes across
several maturities, so this is a heavily over-determined least-squares problem.
Two design decisions are worth stating explicitly.

Vega weighting.  The residuals are price errors divided by the Black-Scholes
vega of the quote.  To first order that converts a price error into an implied
volatility error, which is the scale on which options are actually compared: a
5-cent error on a deep out-of-the-money option is a large mispricing, the same
5 cents on a deep in-the-money one is noise.  Unweighted price fitting would
put essentially all of the weight on the at-the-money strikes.

One parameter set for all maturities.  VG is a Levy model: increments are
stationary and independent, so a single (theta, sigma, nu) has to reprice every
expiry at once.  We therefore fit globally, and separately fit each expiry on
its own.  The gap between the two is the model's term-structure error, and it is
a real and well known limitation of pure Levy models rather than a numerical
artefact -- see the results table.

How many parameters are actually free: **two**, not three.  Under the
specification of Section 4.1, S_t = S_0 exp(X_t), the martingale condition
forces theta_Q + sigma_Q^2/2 = (1 - e^{-r nu})/nu, so option prices depend on
the physical parameters only through (sigma_Q, nu) -- whole families of
(theta, sigma) price identically to machine precision.  Fitting three would be
optimising along a flat direction and reporting a meaningless value for theta,
so theta is fixed at 0 and (sigma, nu) are fitted, which reaches exactly the
same set of prices.  See g.esscher.identifiability_constraint.

The consequence for the empirical section is that this specification can fit the
*level* and the *curvature* of an index smile but not its *slope*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vg.esscher import esscher_domain
from vg.pricing import (
    bs_call,
    bs_vega,
    implied_vol,
    vg_call_esscher_fourier,
)
from vg.process import VGParams

__all__ = [
    "CalibrationResult",
    "PARAM_LIMITS",
    "price_quotes",
    "calibrate",
    "calibrate_flat_vol",
    "calibrate_per_expiry",
]

# Cheaper quadrature during the optimisation; the final report is re-priced
# with the default (much tighter) settings.
# Quadrature settings for the optimiser's inner loop.  Deliberately loose: the
# short expiries of an index chain sit in the heavy-tailed regime where the
# Fourier integral cannot converge and would otherwise burn the full node budget
# on every one of a few thousand objective evaluations.  At these settings the
# worst-case pricing error is around 1e-4 dollars -- some three orders of
# magnitude below the bid-ask spread of the quotes being fitted, so it cannot
# move the optimum.  `_report` re-prices the final parameters at full accuracy.
FAST_QUAD = {"tol": 1e-8, "n_gl": 10, "max_nodes": 30_000}

MAX_NFEV = 150  # 2-3 parameters; least_squares converges well inside this

# Free parameters per measure -- see the module docstring on identifiability.
FREE_PARAMS = {"esscher": ("sigma", "nu")}

PARAM_LIMITS = {"theta": (-2.0, 1.0), "sigma": (0.005, 2.0), "nu": (1e-3, 5.0)}


def _pack(params: VGParams, measure: str) -> np.ndarray:
    return np.array([getattr(params, n) for n in FREE_PARAMS[measure]], dtype=float)


def _unpack(z, measure: str) -> VGParams:
    values = dict(zip(FREE_PARAMS[measure], np.asarray(z, dtype=float)))
    return VGParams(
        theta=float(values.get("theta", 0.0)),
        sigma=float(values["sigma"]),
        nu=float(values["nu"]),
    )


def _bounds(measure: str):
    names = FREE_PARAMS[measure]
    return (
        tuple(PARAM_LIMITS[n][0] for n in names),
        tuple(PARAM_LIMITS[n][1] for n in names),
    )


@dataclass
class CalibrationResult:
    params: VGParams
    measure: str
    rmse_price: float
    rmse_iv: float
    mae_iv: float
    max_abs_iv: float
    n_quotes: int
    success: bool
    message: str
    n_starts: int = 1
    quotes: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def free_parameters(self) -> tuple[str, ...]:
        return FREE_PARAMS[self.measure]

    @property
    def at_bound(self) -> tuple[str, ...]:
        """Free parameters that ended up pinned against their bounds.

        Worth surfacing rather than burying.  A parameter at a bound usually
        means the model is being pushed somewhere it cannot go -- for VG under
        the Esscher measure, nu driven to its floor is the model collapsing to
        Black-Scholes because it cannot bend the way the data asks.
        """
        out = []
        for name in self.free_parameters:
            value = getattr(self.params, name)
            lo, hi = PARAM_LIMITS[name]
            width = hi - lo
            if value - lo < 1e-3 * width or hi - value < 1e-3 * width:
                out.append(name)
        return tuple(out)

    def risk_neutral_params(self, rate: float) -> VGParams:
        """The VG parameters of the log-price under the fitted measure."""
        if self.measure == "esscher":
            from vg.esscher import risk_neutral_params as rn

            return rn(self.params, rate)[0]
        return self.params

    def summary(self) -> str:
        theta = (
            f"theta={self.params.theta:+.4f}"
            if "theta" in self.free_parameters
            else "theta=  fixed"
        )
        pinned = f"  [at bound: {', '.join(self.at_bound)}]" if self.at_bound else ""
        return (
            f"{self.measure:>8} | {theta} "
            f"sigma={self.params.sigma:.4f} nu={self.params.nu:.4f} | "
            f"RMSE {self.rmse_price:.4f} USD, {1e4 * self.rmse_iv:6.1f} bp vol | "
            f"n={self.n_quotes}, {len(self.free_parameters)} free{pinned}"
        )


def _price_one_expiry(
    params: VGParams,
    spot: float,
    strikes: np.ndarray,
    r: float,
    q: float,
    T: float,
    measure: str,
    quad: dict,
) -> np.ndarray:
    if measure != "esscher":
        raise ValueError(f"unknown measure {measure!r}; only 'esscher' is supported")
    return np.asarray(vg_call_esscher_fourier(spot, strikes, r, T, params, q, **quad))


def price_quotes(
    quotes: pd.DataFrame,
    params: VGParams,
    measure: str = "esscher",
    quad: dict | None = None,
) -> np.ndarray:
    """Model prices aligned with the rows of `quotes`.

    Quotes are grouped by (T, r, q) so the characteristic function and its
    quadrature grid are built once per expiry rather than once per strike.
    """
    quad = {} if quad is None else quad
    out = np.empty(len(quotes), dtype=float)
    for _, idx in quotes.groupby(["T", "r", "q"], sort=False).indices.items():
        idx = np.asarray(idx)
        row = quotes.iloc[idx[0]]
        out[idx] = _price_one_expiry(
            params,
            float(row.spot),
            quotes["strike"].to_numpy()[idx],
            float(row.r),
            float(row.q),
            float(row["T"]),
            measure,
            quad,
        )
    return out


def _feasible(params: VGParams, measure: str) -> bool:
    """Both h* and h*+1 have to lie inside the MGF domain for the tilt to exist."""
    try:
        esscher_domain(params)
    except ValueError:
        return False
    return True


def _residuals(
    z: np.ndarray, quotes: pd.DataFrame, measure: str, weights: np.ndarray, quad: dict
) -> np.ndarray:
    n = len(quotes)
    try:
        params = _unpack(z, measure)
    except ValueError:
        return np.full(n, 1e3)
    if not _feasible(params, measure):
        return np.full(n, 1e3)
    try:
        model = price_quotes(quotes, params, measure, quad)
    except (ValueError, RuntimeError, FloatingPointError):
        return np.full(n, 1e3)
    if not np.isfinite(model).all():
        return np.full(n, 1e3)
    return (model - quotes["price"].to_numpy()) / weights


def _weights(quotes: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "vega":
        v = np.asarray(
            bs_vega(
                quotes["spot"].to_numpy(),
                quotes["strike"].to_numpy(),
                quotes["r"].to_numpy(),
                quotes["T"].to_numpy(),
                quotes["iv"].to_numpy(),
                quotes["q"].to_numpy(),
            ),
            dtype=float,
        )
        # Floor at 1% of the median vega: a near-zero vega would otherwise turn
        # a negligible quote into an infinitely important residual.
        return np.maximum(v, 0.01 * np.median(v))
    if scheme == "relative":
        return np.maximum(quotes["price"].to_numpy(), 0.05)
    if scheme == "price":
        return np.ones(len(quotes))
    raise ValueError(f"unknown weight scheme {scheme!r}")


def _default_starts(measure: str) -> list[VGParams]:
    """A small multi-start grid.

    The objective is not convex in nu -- a low-vol/high-nu fit and a
    high-vol/low-nu fit can both be locally optimal -- so a single start is not
    enough.  These cover the range of (skew, tail) regimes seen on index chains.
    Under the Esscher measure theta is ignored (it is not a free parameter), so
    the grid collapses to the distinct (sigma, nu) pairs.
    """
    grid = [
        VGParams(-0.10, 0.15, 0.20),
        VGParams(-0.30, 0.12, 0.50),
        VGParams(-0.05, 0.20, 0.05),
        VGParams(-0.50, 0.25, 1.00),
        VGParams(-0.15, 0.10, 0.10),
    ]
    if measure == "esscher":
        seen, out = set(), []
        for p in grid:
            key = (round(p.sigma, 6), round(p.nu, 6))
            if key not in seen:
                seen.add(key)
                out.append(VGParams(0.0, p.sigma, p.nu))
        return out
    return grid


def calibrate(
    quotes: pd.DataFrame,
    measure: str = "esscher",
    *,
    weight: str = "vega",
    starts: list[VGParams] | None = None,
    quad: dict | None = None,
    verbose: bool = False,
) -> CalibrationResult:
    """Fit the VG parameters to `quotes` under the chosen martingale measure.

    The free parameters are (sigma, nu); theta is not identifiable under this
    specification and is held at 0.  See the module docstring.

    `quotes` must have the columns produced by `vg.data.build_quotes`:
    spot, strike, T, r, q, price, iv.
    """
    if measure not in FREE_PARAMS:
        raise ValueError(f"unknown measure {measure!r}; only 'esscher' is supported")
    if len(quotes) < 5:
        raise ValueError(f"need at least 5 quotes to calibrate, got {len(quotes)}")

    quad = FAST_QUAD if quad is None else quad
    w = _weights(quotes, weight)
    starts = starts or _default_starts(measure)

    best = None
    failures: list[str] = []
    for start in starts:
        try:
            sol = _least_squares(_pack(start, measure), quotes, measure, w, quad)
        except Exception as exc:
            failures.append(f"{start}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  start {start} failed: {exc}")
            continue
        if verbose:
            print(f"  start {start} -> cost {sol.cost:.6g}, x = {sol.x}")
        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:
        # Report why rather than just that -- a bare "all starts failed" hides
        # whether the model was infeasible, the data malformed, or the machine
        # simply out of memory.
        detail = "\n  ".join(failures) or "(no starts were attempted)"
        raise RuntimeError(
            f"all {len(starts)} calibration starts failed for measure "
            f"{measure!r} on {len(quotes)} quotes:\n  {detail}"
        )

    return _report(
        _unpack(best.x, measure), quotes, measure, best.success, str(best.message), len(starts)
    )


def _least_squares(x0, quotes, measure, w, quad):
    from scipy import optimize

    return optimize.least_squares(
        _residuals,
        x0=np.asarray(x0, dtype=float),
        bounds=_bounds(measure),
        args=(quotes, measure, w, quad),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=MAX_NFEV,
    )


def _report(
    params: VGParams,
    quotes: pd.DataFrame,
    measure: str,
    success: bool,
    message: str,
    n_starts: int,
) -> CalibrationResult:
    """Re-price at full quadrature accuracy and build the per-quote diagnostics."""
    out = quotes.copy()
    out["model_price"] = price_quotes(out, params, measure)
    out["price_error"] = out["model_price"] - out["price"]
    out["model_iv"] = [
        implied_vol(
            float(row.model_price), float(row.spot), float(row.strike),
            float(row.r), float(row["T"]), float(row.q),
        )
        for _, row in out.iterrows()
    ]
    out["iv_error"] = out["model_iv"] - out["iv"]

    iv_err = out["iv_error"].to_numpy(dtype=float)
    iv_err = iv_err[np.isfinite(iv_err)]
    return CalibrationResult(
        params=params,
        measure=measure,
        rmse_price=float(np.sqrt(np.mean(out["price_error"].to_numpy() ** 2))),
        rmse_iv=float(np.sqrt(np.mean(iv_err**2))) if iv_err.size else float("nan"),
        mae_iv=float(np.mean(np.abs(iv_err))) if iv_err.size else float("nan"),
        max_abs_iv=float(np.max(np.abs(iv_err))) if iv_err.size else float("nan"),
        n_quotes=len(out),
        success=success,
        message=message,
        n_starts=n_starts,
        quotes=out,
    )


def calibrate_flat_vol(quotes: pd.DataFrame, weight: str = "vega") -> float:
    """Black-Scholes with one volatility for the whole surface.

    Exists so the benchmark gets **exactly** the same treatment as the VG
    models: same quote set, same vega-weighted objective, same notion of a fit.
    Estimating Black-Scholes from historical returns and then comparing it with
    a VG model fitted to option prices is not a comparison of models -- one has
    seen the answer and the other has not, and the fitted model wins by
    construction no matter whether it is any good.  If the two are to be ranked
    on option prices, both must be fitted to option prices.
    """
    from scipy.optimize import minimize_scalar

    w = _weights(quotes, weight)
    spot = quotes["spot"].to_numpy()
    strike = quotes["strike"].to_numpy()
    r = quotes["r"].to_numpy()
    t = quotes["T"].to_numpy()
    q = quotes["q"].to_numpy()
    market = quotes["price"].to_numpy()

    def cost(sigma: float) -> float:
        model = np.asarray(bs_call(spot, strike, r, t, float(sigma), q))
        return float(np.sum(((model - market) / w) ** 2))

    return float(minimize_scalar(cost, bounds=(0.01, 1.5), method="bounded").x)


def calibrate_per_expiry(
    quotes: pd.DataFrame,
    measure: str = "esscher",
    *,
    min_quotes: int = 6,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Fit each expiry independently.

    Compared with the single global fit this is what quantifies the model's
    term-structure error: if VG could reproduce the whole surface, the
    per-expiry parameters would be constant across T.
    """
    rows = []
    for (exp, T), grp in quotes.groupby(["expiry", "T"], sort=True):
        if len(grp) < min_quotes:
            continue
        if verbose:
            print(f"    {exp} (T={T:.4f}, n={len(grp)}) ...", flush=True)
        try:
            # A single expiry is a much easier surface than the whole chain, so
            # a reduced start set is enough and keeps the sweep affordable.
            res = calibrate(grp, measure, starts=_default_starts(measure)[:3], **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            rows.append({"expiry": exp, "T": T, "n": len(grp), "error": str(exc)})
            continue
        rows.append(
            {
                "expiry": exp,
                "T": T,
                "n": res.n_quotes,
                "theta": res.params.theta,
                "sigma": res.params.sigma,
                "nu": res.params.nu,
                "rmse_price": res.rmse_price,
                "rmse_iv_bp": 1e4 * res.rmse_iv,
            }
        )
    return pd.DataFrame(rows)

