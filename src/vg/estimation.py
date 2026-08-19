"""Estimating the *physical* VG parameters from a return series.

This is the P-measure counterpart of `vg.calibration` (which works under Q).
Having both is the point: the Esscher transform is precisely the bridge between
them, so we can estimate (theta, sigma, nu) from the index's realised returns, map
them through h* and compare the resulting option prices with what the market
actually charges.

Two estimators are provided:

  * method of moments -- match variance, skewness and excess kurtosis, which
    have the closed forms in `vg.process.cumulants`.  Fast, no local optima,
    and it makes the role of each parameter transparent: sigma sets the scale,
    nu the tail thickness, theta the asymmetry.

  * maximum likelihood -- maximise the Bessel-K density of `vg.process`.

Both allow a deterministic drift mu, so the model for a log-return over dt is
mu*dt + X(dt).

Which one to trust at daily frequency
-------------------------------------
Prefer the moment estimator on daily data, and read the MLE as a diagnostic.
The reason is structural, not numerical.  The shape of the Gamma clock over one
step is a = dt/nu, and for daily returns with the nu that index data implies
(dt = 1/252, nu ~ 0.2) this is around 0.02.  At such a small shape the VG
density has a power singularity |x|^{2a-1} at the origin, so the model asserts
that most days see essentially no activity and a handful carry all the movement:
a simulated sample has over 60% of its returns within 1e-8 of the drift.  The
log-likelihood is then dominated by how sharply a parameter set spikes at the
mode, and it can be raised by inflating nu well past the value that generated
the data.  Verified on synthetic samples in the test suite: at a ~ 0.02 the MLE
drifts to roughly 6x the true nu even though the density itself is exact (it
integrates to 1 and reproduces every moment to ten digits), while the moment
estimator recovers it.  The pathology disappears once a is O(1) -- weekly or
monthly returns, or a small nu -- where the MLE is the better estimator.

Real returns do not pile up at the origin the way the model says they should,
which is itself worth reporting: it is a concrete limitation of VG as a
description of the physical measure, quite separate from how well it prices
options under Q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from vg.process import VGParams, cumulants, log_density

__all__ = [
    "MomentFit",
    "MLEFit",
    "moment_estimates",
    "fit_mle",
    "historical_volatility",
    "annualised_summary",
]


def historical_volatility(returns, dt: float) -> float:
    """Annualised sample volatility -- the Black-Scholes parameter.

    The one-parameter counterpart of `moment_estimates`: Black-Scholes has a
    single parameter and estimating it from a return series is just the sample
    standard deviation, annualised.  Kept here so that the two models are
    estimated side by side from exactly the same data.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError(f"need at least 2 returns, got {r.size}")
    return float(np.std(r, ddof=1) / np.sqrt(dt))


@dataclass
class MomentFit:
    params: VGParams
    mu: float
    dt: float
    sample: dict[str, float]
    fitted: dict[str, float]
    success: bool


@dataclass
class MLEFit:
    params: VGParams
    mu: float
    dt: float
    loglik: float
    n_obs: int
    success: bool
    message: str

    @property
    def aic(self) -> float:
        return 2 * 4 - 2 * self.loglik  # mu, theta, sigma, nu


def _sample_moments(r: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(r)),
        "var": float(np.var(r, ddof=1)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
    }


def _model_moments(p: VGParams, dt: float) -> dict[str, float]:
    k = cumulants(p, dt)
    return {
        "var": k["k2"],
        "skewness": k["k3"] / k["k2"] ** 1.5,
        "excess_kurtosis": k["k4"] / k["k2"] ** 2,
    }


def moment_estimates(returns, dt: float) -> MomentFit:
    """Match variance, skewness and excess kurtosis of the log returns.

    Starting values come from the small-theta approximations

        sigma^2 ~ var/dt,   nu ~ excess_kurtosis * dt / 3,
        theta   ~ skewness * sigma * sqrt(dt) / (3 nu),

    after which the three exact moment equations are solved numerically.
    sigma and nu are optimised in logs so they stay positive.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 50:
        raise ValueError(f"need at least 50 returns, got {r.size}")

    s = _sample_moments(r)
    sigma0 = np.sqrt(max(s["var"], 1e-12) / dt)
    nu0 = float(np.clip(max(s["excess_kurtosis"], 1e-3) * dt / 3.0, 1e-4, 5.0))
    theta0 = s["skewness"] * sigma0 * np.sqrt(dt) / (3.0 * nu0)
    theta0 = float(np.clip(theta0, -5.0, 5.0))

    target = np.array([s["var"], s["skewness"], s["excess_kurtosis"]])
    # Scale each residual by the magnitude of its target so no single moment
    # (variance is ~1e-4, kurtosis is O(1)) dominates the fit.
    scale = np.maximum(np.abs(target), [1e-8, 1e-2, 1e-2])

    def residuals(z: np.ndarray) -> np.ndarray:
        theta, log_sigma, log_nu = z
        try:
            p = VGParams(theta, float(np.exp(log_sigma)), float(np.exp(log_nu)))
            m = _model_moments(p, dt)
        except ValueError:
            return np.full(3, 1e6)
        got = np.array([m["var"], m["skewness"], m["excess_kurtosis"]])
        return (got - target) / scale

    sol = optimize.least_squares(
        residuals,
        x0=np.array([theta0, np.log(sigma0), np.log(nu0)]),
        method="lm",
        xtol=1e-14,
        ftol=1e-14,
        max_nfev=20_000,
    )
    theta, log_sigma, log_nu = sol.x
    p = VGParams(float(theta), float(np.exp(log_sigma)), float(np.exp(log_nu)))
    mu = (s["mean"] - p.theta * dt) / dt

    return MomentFit(
        params=p,
        mu=float(mu),
        dt=dt,
        sample=s,
        fitted=_model_moments(p, dt),
        success=bool(sol.success),
    )


def fit_mle(
    returns,
    dt: float,
    start: VGParams | None = None,
    mu0: float | None = None,
    max_eval: int = 4000,
) -> MLEFit:
    """Maximum likelihood using the closed-form VG density.

    Optimises (mu, theta, log sigma, log nu) with Nelder-Mead, which is slower
    than a gradient method but does not need the derivative of a Bessel
    function and is robust to the flat directions of this likelihood.  Two
    restarts, because Nelder-Mead reliably stalls once on this surface.

    See the module docstring before reading the output on daily data: at
    dt/nu << 1 the likelihood is dominated by the singularity at the origin and
    over-estimates nu.  `moment_estimates` is the more reliable estimator there.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 50:
        raise ValueError(f"need at least 50 returns, got {r.size}")

    if start is None or mu0 is None:
        mfit = moment_estimates(r, dt)
        start = start or mfit.params
        mu0 = mfit.mu if mu0 is None else mu0

    def neg_loglik(z: np.ndarray) -> float:
        mu, theta, log_sigma, log_nu = z
        if not np.isfinite(z).all() or abs(log_sigma) > 20 or abs(log_nu) > 20:
            return 1e12
        try:
            p = VGParams(float(theta), float(np.exp(log_sigma)), float(np.exp(log_nu)))
            ll = log_density(p, r - mu * dt, dt)
        except (ValueError, FloatingPointError):
            return 1e12
        if not np.isfinite(ll).all():
            return 1e12
        return -float(np.sum(ll))

    z0 = np.array([mu0, start.theta, np.log(start.sigma), np.log(start.nu)])
    sol = optimize.minimize(
        neg_loglik,
        z0,
        method="Nelder-Mead",
        options={"maxiter": max_eval, "maxfev": max_eval, "xatol": 1e-9, "fatol": 1e-9},
    )
    sol = optimize.minimize(
        neg_loglik,
        sol.x,
        method="Nelder-Mead",
        options={"maxiter": max_eval, "maxfev": max_eval, "xatol": 1e-11, "fatol": 1e-11},
    )

    mu, theta, log_sigma, log_nu = sol.x
    return MLEFit(
        params=VGParams(float(theta), float(np.exp(log_sigma)), float(np.exp(log_nu))),
        mu=float(mu),
        dt=dt,
        loglik=float(-sol.fun),
        n_obs=int(r.size),
        success=bool(sol.success),
        message=str(sol.message),
    )


def annualised_summary(p: VGParams, mu: float, dt: float) -> dict[str, float]:
    """Human-readable annualised diagnostics for a fitted parameter set."""
    one_year = cumulants(p, 1.0)
    m = cumulants(p, dt)
    return {
        "mu_annual": mu,
        "theta": p.theta,
        "sigma": p.sigma,
        "nu": p.nu,
        "annual_vol": float(np.sqrt(one_year["k2"])),
        "annual_skewness": float(one_year["k3"] / one_year["k2"] ** 1.5),
        "daily_excess_kurtosis": float(m["k4"] / m["k2"] ** 2),
    }

