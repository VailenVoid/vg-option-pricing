"""Esscher transform and the risk-neutral measure for the VG model.

Section 4.2 of the paper.  The Esscher measure Q^h is defined on F_t by

    dQ^h / dP |_{F_t} = exp(h X_t) / M(h, t),

and h = h* is fixed by requiring that the discounted price be a Q-martingale.

A remark on the fixed-point equation
------------------------------------
The paper states the martingale condition as e^{rt} = M(h*+1, t) / M(h*, t)
(eq. 4) and then defines H(h) = D_h / D_{h+1}, asserting H(h*) = e^{rt}.  Since
M(h, t) = D_h^{-t/nu}, the condition is really

    e^{rt} = (D_{h*} / D_{h*+1})^{t/nu} = H(h*)^{t/nu}   <=>   H(h*) = e^{r nu},

which is what the paper itself uses two paragraphs later.  The e^{r nu} form is
the correct one and it matters: h* must not depend on t (a single measure has to
price every maturity), and only the e^{r nu} version has that property.  This
module implements H(h*) = e^{r nu}.

The tilted process is again VG
------------------------------
Writing D_{h+iu} = D_h - i u nu (theta + sigma^2 h) + (1/2) sigma^2 nu u^2 and
dividing by D_h shows that under Q^h the process X is again a VG process, with

    theta' = (theta + sigma^2 h) / D_h,   sigma' = sigma / sqrt(D_h),   nu' = nu.

This is not in the paper but it is worth having: it turns the abstract change of
measure into an explicit reparametrisation, so the option can be priced by
simulating *directly* under Q instead of reweighting P-paths, and it gives a
completely independent check on the Fourier code.

It also exposes something that is easy to miss and that matters a great deal
for the empirical part -- see `identifiability_constraint` below.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from vg.process import VGParams, mgf_bracket, mgf_domain

__all__ = [
    "esscher_h_star",
    "esscher_tilt",
    "risk_neutral_params",
    "esscher_cf",
    "esscher_domain",
    "identifiability_constraint",
    "H_ratio",
]

_EDGE = 1e-9  # relative pull-back from the open ends of the admissible interval


def esscher_domain(p: VGParams) -> tuple[float, float]:
    """Admissible interval (h1, h2 - 1) for the Esscher parameter.

    Both M(h, t) and M(h+1, t) must be finite, so h and h+1 must lie in the MGF
    domain (h1, h2).  This requires h2 - h1 > 1, i.e.

        2 * sqrt(theta^2/sigma^4 + 2/(nu sigma^2)) > 1,

    which holds comfortably for any realistic calibration but is checked here.
    """
    h1, h2 = mgf_domain(p)
    lo, hi = h1, h2 - 1.0
    if hi <= lo:
        raise ValueError(
            f"empty Esscher domain for {p}: the MGF interval {(h1, h2)} is "
            "shorter than 1, so M(h,t) and M(h+1,t) cannot both be finite"
        )
    return lo, hi


def H_ratio(p: VGParams, h):
    """H(h) = D_h / D_{h+1} (Tvrdjenje 4.1).  Strictly increasing on (h1, h2-1)."""
    return mgf_bracket(p, h) / mgf_bracket(p, np.asarray(h) + 1.0)


def esscher_h_star(p: VGParams, rate: float, *, check: bool = True) -> float:
    """Solve H(h*) = exp(rate * nu) for h* in (h1, h2 - 1).

    `rate` is the cost of carry r - q (use r when there is no dividend yield);
    it is the drift the discounted, dividend-reinvested price must have.

    The equation D_{h} = E * D_{h+1} with E = exp(rate*nu) is quadratic in h,
    because both sides are quadratic.  With D_h = a h^2 + b h + c,
    a = -sigma^2 nu/2, b = -theta nu, c = 1, it reads

        a(1-E) h^2 + [b(1-E) - 2aE] h + [c(1-E) - E(a+b)] = 0.

    Tvrdjenje 4.1 guarantees exactly one root in (h1, h2-1); the other is
    discarded automatically.  When rate == 0 we have E = 1, the quadratic term
    vanishes and the equation degenerates to the linear one with the closed-form
    solution h* = -1/2 - theta/sigma^2.
    """
    lo, hi = esscher_domain(p)
    th, s, nu = p.theta, p.sigma, p.nu
    e = np.exp(rate * nu)

    a = -0.5 * s**2 * nu
    b = -th * nu
    c = 1.0

    qa = a * (1.0 - e)
    qb = b * (1.0 - e) - 2.0 * a * e
    qc = c * (1.0 - e) - e * (a + b)

    span = hi - lo
    tol = _EDGE * max(span, 1.0)

    if abs(qa) <= 1e-14 * max(abs(qb), abs(qc), 1.0):
        # Degenerate (rate ~ 0): linear equation, exact solution -1/2 - theta/sigma^2.
        roots = [-qc / qb] if qb != 0.0 else []
    else:
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            roots = []
        else:
            sqrt_disc = np.sqrt(disc)
            # Numerically stable quadratic: avoid cancellation in -b +- sqrt(D).
            helper = -0.5 * (qb + np.sign(qb) * sqrt_disc) if qb != 0.0 else -0.5 * sqrt_disc
            roots = []
            if helper != 0.0:
                roots = [helper / qa, qc / helper]
            else:
                roots = [0.0]

    inside = [float(x) for x in roots if lo + tol < x < hi - tol]

    if len(inside) == 1:
        h = inside[0]
    else:
        # Fall back on bisection.  F(h) = D_h - E*D_{h+1} has the sign of
        # H(h) - E because D_{h+1} > 0 on the interval, and H increases from 0
        # to +infinity, so a sign change is guaranteed.
        h = _h_star_bisect(p, e, lo, hi)

    if check:
        got = float(H_ratio(p, h))
        if not np.isclose(got, e, rtol=1e-8, atol=1e-12):
            h = _h_star_bisect(p, e, lo, hi)
            got = float(H_ratio(p, h))
            if not np.isclose(got, e, rtol=1e-6, atol=1e-10):
                raise RuntimeError(
                    f"Esscher solve failed for {p}, rate={rate}: "
                    f"H(h*)={got} vs target {e}"
                )
    return float(h)


def _h_star_bisect(p: VGParams, e: float, lo: float, hi: float) -> float:
    span = hi - lo

    def f(h: float) -> float:
        return float(mgf_bracket(p, h) - e * mgf_bracket(p, h + 1.0))

    a, b = lo + 1e-12 * span, hi - 1e-12 * span
    fa, fb = f(a), f(b)
    for _ in range(200):
        if fa * fb < 0.0:
            break
        a += 0.05 * (b - a)
        b -= 0.05 * (b - a)
        fa, fb = f(a), f(b)
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"could not bracket the Esscher root for {p}")
    return float(optimize.brentq(f, a, b, xtol=1e-15, rtol=1e-15, maxiter=200))


def esscher_tilt(p: VGParams, h: float) -> VGParams:
    """Parameters of X under Q^h.  Still a VG process, with nu unchanged.

        theta' = (theta + sigma^2 h) / D_h,   sigma' = sigma / sqrt(D_h),   nu' = nu.
    """
    d = float(mgf_bracket(p, h))
    if d <= 0.0:
        raise ValueError(f"h={h} is outside the MGF domain {mgf_domain(p)} of {p}")
    return VGParams(
        theta=(p.theta + p.sigma**2 * h) / d,
        sigma=p.sigma / np.sqrt(d),
        nu=p.nu,
    )


def risk_neutral_params(p: VGParams, rate: float) -> tuple[VGParams, float]:
    """(Q-parameters, h*) for the Esscher risk-neutral measure at carry `rate`.

    By construction E^Q[exp(X_t)] = exp(rate * t), which the tests verify.
    """
    h = esscher_h_star(p, rate)
    return esscher_tilt(p, h), h


def identifiability_constraint(nu: float, rate: float) -> float:
    """The value that theta_Q + sigma_Q^2/2 is forced to take: (1 - e^{-rate nu})/nu.

    Why this exists
    ---------------
    Under the model of Section 4.1, S_t = S_0 exp(X_t), the price carries *no*
    drift of its own: the whole burden of being a martingale falls on the law of
    X under Q.  Writing the martingale condition E^Q[exp(X_t)] = exp(rate t) in
    terms of the tilted parameters gives D'_1 = 1 - nu(theta_Q + sigma_Q^2/2)
    = exp(-rate nu), that is

        theta_Q + sigma_Q^2 / 2 = (1 - exp(-rate nu)) / nu.

    So the risk-neutral law has only **two** free parameters, (sigma_Q, nu), not
    three: theta_Q is whatever the constraint says.  Since option prices depend
    on the physical (theta, sigma, nu) only through (theta_Q, sigma_Q, nu), the
    physical theta is *not identifiable* from option prices in this
    specification -- entire one-parameter families of (theta, sigma) give
    bit-for-bit identical prices, which the test suite verifies directly.

    The practical consequences are large enough to state plainly:

      * Calibrating three parameters through the Esscher transform is fitting a
        flat direction.  `vg.calibration` therefore fixes theta = 0 and fits
        (sigma, nu) for this measure, which loses nothing.
      * The reachable smiles are nearly symmetric.  Because theta_Q is pinned
        near -sigma_Q^2/2 (exactly that when rate = 0), the model produces
        curvature but almost no skew -- under a fifth of a volatility point
        across 80%-125% moneyness -- while equity index smiles are steeply
        downward sloping.

    Note where the constraint comes from: the price process S_t = S_0 exp(X_t)
    of Section 4.1 carries no drift of its own, so the martingale condition has
    nothing to fall on but the law of X.  The limitation is therefore in that
    specification, not in the VG process and not in the Esscher transform.
    """
    if nu <= 0:
        raise ValueError(f"nu must be > 0, got {nu}")
    return float(-np.expm1(-rate * nu) / nu)



def esscher_cf(p: VGParams, u, t: float, h: float):
    """Characteristic function of X_t under Q^h -- eq. (7) of the paper.

        phi_hat(u, t, h) = M(h + iu, t) / M(h, t) = ( D_h / D_{h+iu} )^{t/nu}.

    The bracket D_{h+iu} = D_h + (1/2) sigma^2 nu u^2 - i u nu (theta + sigma^2 h)
    has strictly positive real part whenever D_h > 0, so the principal branch of
    the complex power is continuous here (see `vg.process.cf`).
    """
    u = np.asarray(u)
    d_h = mgf_bracket(p, h)
    if np.real(d_h) <= 0.0:
        raise ValueError(f"h={h} outside the MGF domain {mgf_domain(p)} of {p}")
    d_hu = mgf_bracket(p, h + 1j * u)
    if np.any(np.real(d_hu) <= 0.0):
        raise ValueError("characteristic-function bracket left the right half-plane")
    return (d_h / d_hu) ** (t / p.nu)

