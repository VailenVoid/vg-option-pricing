"""Pricing European calls in the VG model.

Two complementary routes, exactly as in Sections 4.4 and 4.5 of the paper:

  * Fourier inversion.  The tail probabilities in Theorem 4.1 are obtained from
    the Esscher-tilted characteristic function (eq. 7) through the Gil-Pelaez
    inversion formula (eq. 6).

  * Monte Carlo.  Either by importance sampling under P with the Esscher weight
    exp(h* X_T)/M(h*, T) (eq. 8), or -- using the explicit tilted parameters
    from `vg.esscher` -- by simulating directly under Q.


Black-Scholes and implied volatility live here too, since every comparison in
the paper is made on the implied-volatility scale.

Dividends.  All functions take a continuous dividend yield `q` (0 by default,
which reproduces the formulas of the paper verbatim).  Everything goes through
the cost of carry r - q: the Esscher parameter solves H(h*) = exp((r-q) nu) and
the stock leg is discounted by exp(-q T).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from vg.esscher import esscher_cf, esscher_h_star, esscher_tilt
from vg.process import VGParams, cf, density, mgf_bracket, simulate_terminal

__all__ = [
    "QuadInfo",
    "gil_pelaez_tail",
    "gil_pelaez_cdf",
    "call_price_from_cf",
    "vg_call_esscher_fourier",
    "vg_put_esscher_fourier",
    "vg_call_density_quad",
    "MCResult",
    "vg_call_mc_esscher",
    "vg_call_mc_direct",
    "bs_call",
    "bs_put",
    "bs_vega",
    "implied_vol",
]


# --------------------------------------------------------------------------- #
# Gil-Pelaez inversion
# --------------------------------------------------------------------------- #


@dataclass
class QuadInfo:
    """What the quadrature actually achieved -- returned on request.

    `error_estimate` is an estimate of the remaining truncation error in the
    tail probability, and `converged` says whether the requested tolerance was
    met inside the node budget.  Both are reported rather than assumed: for
    small T/nu the Gil-Pelaez integral converges slowly (see the note on
    `gil_pelaez_tail`), and a routine that silently returned three correct
    digits while claiming eleven would be worse than useless.
    """

    u_max: float
    n_nodes: int
    n_segments: int
    error_estimate: float
    converged: bool


def _segment_integral(cf_fn, x_arr, a, b, n_sub, n_gl, max_block):
    """Composite Gauss-Legendre integral of Im[e^{-iux} phi(u)]/u over [a, b].

    Returns the contribution for every x at once: the characteristic function is
    evaluated once on the shared node set and reused across strikes, which is
    what makes calibrating a whole chain affordable.

    Written in real arithmetic rather than complex.  With base = a + ib,

        Im[e^{-iux}(a + ib)] = b cos(ux) - a sin(ux),

    so the whole segment is two real matrix-vector products.  Done in complex
    arithmetic this needs several (n_x, n_u) complex temporaries, which on a
    full option chain runs to hundreds of megabytes and was enough to exhaust
    memory during calibration; this form allocates two real arrays of that shape
    and is faster besides.
    """
    edges = np.linspace(a, b, n_sub + 1)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    u = (mid[:, None] + half[:, None] * xg[None, :]).ravel()
    w = (half[:, None] * wg[None, :]).ravel()

    phi = np.asarray(cf_fn(u), dtype=complex)
    base = (w / u) * phi  # w/u is real, so it can be folded in before Im[.]
    base_re = np.ascontiguousarray(base.real)
    base_im = np.ascontiguousarray(base.imag)

    out = np.empty(x_arr.size, dtype=float)
    block = max(1, int(max_block // u.size))
    while True:
        try:
            for i in range(0, x_arr.size, block):
                phase = np.outer(x_arr[i : i + block], u)
                cos_p = np.cos(phase)
                np.sin(phase, out=phase)  # reuse the buffer; now holds sin
                out[i : i + block] = cos_p @ base_im - phase @ base_re
            return out, u.size
        except MemoryError:
            # A calibration run is long, and on a loaded machine the temporary
            # (n_x, n_u) arrays can fail to allocate even when they are small.
            # Halving the block trades a little speed for finishing at all;
            # giving up here would throw away the whole fit.
            if block == 1:
                raise
            block = max(1, block // 4)


def gil_pelaez_tail(
    cf_fn,
    x,
    *,
    tol: float = 1e-10,
    n_gl: int = 16,
    max_nodes: int = 400_000,
    max_block: int = 400_000,
    warn: bool = False,
    return_info: bool = False,
):
    """P(Y > x) from the characteristic function of Y (Gil-Pelaez, eq. 6):

        1 - F_Y(x) = 1/2 + (1/pi) * int_0^inf Im[ exp(-i u x) phi(u) ] / u  du.

    `x` may be an array.  The integrand is regular at u -> 0 (it tends to
    E[Y] - x) and Gauss-Legendre nodes never land on an endpoint, so the origin
    needs no special case.

    How far to integrate, and how the error is known
    ------------------------------------------------
    Two things have to be got right, and they pull against each other.  The
    integrand oscillates like exp(-iux), so subintervals must be at most a
    quarter period, pi/(2 max|x|), wide; and it decays only like
    u^{-(1 + 2T/nu)}, so the upper limit has to be pushed out until the tail is
    negligible.  Fixing a truncation point first and then rationing nodes to
    reach it is the trap: under-resolving the oscillation does not lose a digit,
    it returns a number with no digits at all.

    So the integral is instead accumulated over dyadic segments
    [0, U], [U, 2U], [2U, 4U], ..., each fully resolved, and what remains is
    estimated two independent ways, of which the smaller is reported:

      * From the decay of the segment contributions.  If they shrink by a factor
        rho per segment the remaining tail is about s*rho/(1-rho).  This is
        sharp but can be fooled: each segment spans many oscillations, so its
        contribution can pass accidentally close to zero and make convergence
        look better than it is.  Taking the largest of the last three segments
        as the base, and estimating rho across two segments rather than one,
        removes that failure mode.

      * From the characteristic function itself.  The remaining integral is
        bounded by int_U^inf |phi(u)|/u du = |phi(U)|/beta, with the decay
        exponent beta measured on the segment just computed.  Conservative, but
        it cannot be fooled by cancellation.

    Iteration stops when the estimate falls below `tol`, or when the node budget
    runs out -- in which case `converged` is False and `error_estimate` says how
    much accuracy was actually obtained.

    When T/nu is small (a short maturity with a fat-tailed nu) the true tail is
    genuinely heavy and no Fourier method converges quickly; this is a property
    of the VG law -- whose density is unbounded at the origin for T/nu <= 1/2 --
    not of this implementation.  `vg_call_density_quad` is the accurate route
    in that regime, and the two agree wherever both are reliable.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    x_scale = float(np.max(np.abs(x_arr)))
    quarter_period = np.pi / (2.0 * max(x_scale, 1e-3))

    total = np.zeros(x_arr.size, dtype=float)
    # Dyadic segments [0,1], [1,2], [2,4], ... .  The doubling matters twice
    # over: the oscillation is resolved by the quarter-period cap, and the
    # characteristic function's *own* decay -- which can be as steep as
    # u^{-2T/nu} with T/nu in the tens -- is resolved because each segment is
    # only as wide as its own left endpoint, so no subinterval ever straddles
    # more than a few percent of the scale on which phi varies.
    u_lo, u_hi = 0.0, 1.0
    nodes = 0
    n_segments = 0
    recent: list[float] = []
    err = float("inf")
    converged = False
    min_segments = 5

    while True:
        n_sub = max(24, int(np.ceil((u_hi - u_lo) / quarter_period)))
        if n_segments > 0 and nodes + n_sub * n_gl > max_nodes:
            break

        seg, used = _segment_integral(
            cf_fn, x_arr, u_lo, u_hi, n_sub, n_gl, max_block
        )
        total += seg
        nodes += used
        n_segments += 1

        recent.append(float(np.max(np.abs(seg))))
        recent = recent[-3:]

        # Estimate 1: geometric decay of the segment contributions.
        ratio_estimate = float("inf")
        if len(recent) >= 3 and recent[-3] > 0.0:
            rho = float(np.sqrt(recent[-1] / recent[-3]))
            if rho < 0.95:
                ratio_estimate = max(recent) * rho / (1.0 - rho)

        # Estimate 2: |phi(U)|/beta bounds the remaining integral outright.
        bound = float("inf")
        a_hi = abs(complex(np.asarray(cf_fn(np.array([u_hi]))).ravel()[0]))
        a_mid = abs(complex(np.asarray(cf_fn(np.array([0.5 * u_hi]))).ravel()[0]))
        if a_hi > 0.0 and a_mid > a_hi:
            beta = np.log(a_mid / a_hi) / np.log(2.0)
            bound = a_hi / max(beta, 1e-3)
        elif a_hi == 0.0:
            bound = 0.0

        err = min(ratio_estimate, bound)

        # The decay argument only applies past the body of the characteristic
        # function, hence the minimum segment count: two early segments that
        # happen to be similar in size are not convergence.
        if n_segments >= min_segments and err / np.pi < tol:
            converged = True
            break
        u_lo, u_hi = u_hi, 2.0 * u_hi

    # Floored at a few ulps: a probability computed in double precision cannot
    # be more accurate than this, whatever the tail argument says.
    error_estimate = max(err / np.pi, 1e-15)
    if warn and not converged:
        warnings.warn(
            f"Gil-Pelaez did not reach tol={tol:.1e} within {max_nodes} nodes; "
            f"estimated error {error_estimate:.2e} (u_max={u_hi:.3g}). "
            "For small T/nu use vg_call_density_quad instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    out = 0.5 + total / np.pi
    result = out.reshape(np.shape(x)) if np.ndim(x) else float(out[0])

    if return_info:
        return result, QuadInfo(
            u_max=float(u_hi),
            n_nodes=nodes,
            n_segments=n_segments,
            error_estimate=error_estimate,
            converged=converged,
        )
    return result


def gil_pelaez_cdf(cf_fn, x, **kwargs):
    """F_Y(x); the CDF written F-hat in the paper."""
    return 1.0 - np.asarray(gil_pelaez_tail(cf_fn, x, **kwargs))


# --------------------------------------------------------------------------- #
# Generic Fourier call price
# --------------------------------------------------------------------------- #


def call_price_from_cf(cf_logret, S0: float, K, r: float, T: float, q: float = 0.0, **kw):
    """European call from the characteristic function of Y = log(S_T / S_0) under Q.

        C = S_0 exp(-qT) * Pi_1  -  K exp(-rT) * Pi_2,
        Pi_2 = Q(Y > log k),
        Pi_1 = E^Q[ e^Y 1{Y > log k} ] / E^Q[e^Y],   with c.f.  phi(u - i) / phi(-i).

    `cf_logret` must accept complex arguments (Pi_1 needs u - i) and must satisfy
    the martingale condition phi(-i) = exp((r-q)T); this is checked.
    """
    k = np.asarray(K, dtype=float) / S0
    x = np.log(k)

    fwd = complex(np.asarray(cf_logret(np.array([-1j]))).ravel()[0])
    target = np.exp((r - q) * T)
    if not np.isclose(fwd.real, target, rtol=1e-8, atol=1e-10) or abs(fwd.imag) > 1e-8:
        raise ValueError(
            f"characteristic function is not risk-neutral: phi(-i) = {fwd}, "
            f"expected exp((r-q)T) = {target}"
        )

    pi2 = np.asarray(gil_pelaez_tail(cf_logret, x, **kw))
    pi1 = np.asarray(
        gil_pelaez_tail(lambda u: np.asarray(cf_logret(np.asarray(u) - 1j)) / fwd.real, x, **kw)
    )
    price = S0 * np.exp(-q * T) * pi1 - np.asarray(K, float) * np.exp(-r * T) * pi2
    return price if np.ndim(K) else float(price)


def vg_call_esscher_fourier(
    S0: float, K, r: float, T: float, p: VGParams, q: float = 0.0, **kw
):
    """Theorem 4.1:

        C = S_0 [1 - F-hat(log k, T, h*+1)] - K e^{-rT} [1 - F-hat(log k, T, h*)]

    (the S_0 leg carries an extra exp(-qT) when a dividend yield is supplied).
    The two tail probabilities are the same object under the Esscher measures
    Q^{h*} and Q^{h*+1}; tilting by one extra unit is precisely the change to the
    share measure, which is why the Black-Scholes structure survives.
    """
    h = esscher_h_star(p, r - q)
    x = np.log(np.asarray(K, dtype=float) / S0)
    pi2 = np.asarray(gil_pelaez_tail(lambda u: esscher_cf(p, u, T, h), x, **kw))
    pi1 = np.asarray(gil_pelaez_tail(lambda u: esscher_cf(p, u, T, h + 1.0), x, **kw))
    price = S0 * np.exp(-q * T) * pi1 - np.asarray(K, float) * np.exp(-r * T) * pi2
    return price if np.ndim(K) else float(price)


def vg_put_esscher_fourier(
    S0: float, K, r: float, T: float, p: VGParams, q: float = 0.0, **kw
):
    """European put, the mirror image of Theorem 4.1:

        P = K e^{-rT} F-hat(log k, T, h*) - S_0 e^{-qT} F-hat(log k, T, h*+1).

    Computed from the CDFs directly rather than through put-call parity, so
    that parity becomes a check on the implementation instead of an identity
    imposed by construction.
    """
    h = esscher_h_star(p, r - q)
    x = np.log(np.asarray(K, dtype=float) / S0)
    f2 = 1.0 - np.asarray(gil_pelaez_tail(lambda u: esscher_cf(p, u, T, h), x, **kw))
    f1 = 1.0 - np.asarray(gil_pelaez_tail(lambda u: esscher_cf(p, u, T, h + 1.0), x, **kw))
    price = np.asarray(K, float) * np.exp(-r * T) * f2 - S0 * np.exp(-q * T) * f1
    return price if np.ndim(K) else float(price)


def vg_call_density_quad(
    S0: float, K, r: float, T: float, p: VGParams, q: float = 0.0, *, limit: int = 400
):
    """European call by integrating the payoff against the closed-form density.

        C = e^{-rT} * int_{log k}^{inf} (S_0 e^x - K) f_Q(x) dx

    where f_Q is the Bessel-K density of the *tilted* process -- available in
    closed form precisely because the Esscher transform maps VG to VG
    (`vg.esscher.esscher_tilt`).

    This is the third, fully independent pricing route, and it is the accurate
    one exactly where Fourier inversion struggles: for T/nu <= 1/2 the density
    is unbounded at the origin, which makes the characteristic function decay
    slowly but is trivial for an adaptive quadrature told where the singularity
    is.  Slower than Fourier (no reuse across strikes), so it is used as a
    reference rather than inside the calibration loop.
    """
    from scipy.integrate import quad

    h = esscher_h_star(p, r - q)
    qp = esscher_tilt(p, h)

    # Exponential decay rate of e^x f_Q(x) in the right tail; positive because
    # the martingale condition forces E^Q[e^{X_T}] to be finite.
    a_scale = np.sqrt(2.0 * qp.sigma**2 / qp.nu + qp.theta**2)
    lam = (a_scale - qp.theta) / qp.sigma**2 - 1.0
    if lam <= 0:  # pragma: no cover - excluded by the martingale condition
        raise ValueError(f"E^Q[exp(X_T)] is infinite for tilted parameters {qp}")

    k_arr = np.atleast_1d(np.asarray(K, dtype=float))
    out = np.empty(k_arr.size, dtype=float)

    for j, strike in enumerate(k_arr):
        lb = float(np.log(strike / S0))
        ub = max(lb, 0.0) + 40.0 / lam + 10.0 * np.sqrt(T * qp.sigma**2)

        def integrand(xi: float, strike=strike) -> float:
            return (S0 * np.exp(xi) - strike) * float(density(qp, xi, T))

        # Split at 0: that is where the density is singular for T/nu <= 1/2.
        breaks = [lb] + ([0.0] if lb < 0.0 < ub else []) + [ub]
        value = 0.0
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            if hi > lo:
                value += quad(integrand, lo, hi, limit=limit, points=None)[0]
        out[j] = np.exp(-r * T) * value

    return out if np.ndim(K) else float(out[0])



# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #


@dataclass
class MCResult:
    """Monte Carlo price with its standard error."""

    price: np.ndarray | float
    stderr: np.ndarray | float
    n_paths: int
    label: str = ""

    def ci(self, level: float = 0.95) -> tuple:
        z = stats.norm.ppf(0.5 + level / 2.0)
        return (np.asarray(self.price) - z * np.asarray(self.stderr),
                np.asarray(self.price) + z * np.asarray(self.stderr))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        p = np.atleast_1d(self.price)
        s = np.atleast_1d(self.stderr)
        if p.size == 1:
            return f"{self.label}: {p[0]:.6f} +- {s[0]:.6f} (n={self.n_paths:,})"
        return f"{self.label}: {p.size} strikes, max se {s.max():.6f} (n={self.n_paths:,})"


def _mc_from_samples(
    discount: float, s_t: np.ndarray, K, weights: np.ndarray | None, label: str
) -> MCResult:
    k_arr = np.atleast_1d(np.asarray(K, dtype=float))
    n = s_t.size
    price = np.empty(k_arr.size)
    stderr = np.empty(k_arr.size)
    for j, k in enumerate(k_arr):
        vals = discount * np.maximum(s_t - k, 0.0)
        if weights is not None:
            vals = vals * weights
        price[j] = vals.mean()
        stderr[j] = vals.std(ddof=1) / np.sqrt(n)
    if np.ndim(K) == 0:
        return MCResult(float(price[0]), float(stderr[0]), n, label)
    return MCResult(price, stderr, n, label)


def vg_call_mc_esscher(
    S0: float,
    K,
    r: float,
    T: float,
    p: VGParams,
    n_paths: int,
    rng: np.random.Generator,
    q: float = 0.0,
) -> MCResult:
    """Equation (8): simulate under P, reweight by the Esscher density.

        C-hat_N = e^{-rT} (1/N) sum_i  [ e^{h* X_T^(i)} / M(h*, T) ] (S_0 e^{X_T^(i)} - K)^+

    The point of the estimator is that no change to the simulation code is
    needed: the paths are drawn with the statistical parameters and only a
    weight is attached.  The price of that convenience is variance -- the weight
    has a heavy right tail -- which is exactly what `vg_call_mc_direct` avoids.
    """
    h = esscher_h_star(p, r - q)
    x = simulate_terminal(p, T, n_paths, rng)
    # log M(h, T) = -(T/nu) log D_h, so the log-weight is h X_T + (T/nu) log D_h.
    log_w = h * x + (T / p.nu) * np.log(float(mgf_bracket(p, h)))
    return _mc_from_samples(
        np.exp(-r * T), S0 * np.exp(x), K, np.exp(log_w), "MC Esscher (importance sampling)"
    )


def vg_call_mc_direct(
    S0: float,
    K,
    r: float,
    T: float,
    p: VGParams,
    n_paths: int,
    rng: np.random.Generator,
    q: float = 0.0,
) -> MCResult:
    """Simulate directly under the Esscher measure Q^{h*}.

    Uses the fact that the tilted process is again VG with parameters
    (theta', sigma', nu) from `vg.esscher.esscher_tilt`, so the same exact
    sampler applies and no weights are needed.  Prices the same option as
    `vg_call_mc_esscher` but with markedly lower variance.
    """
    h = esscher_h_star(p, r - q)
    x = simulate_terminal(esscher_tilt(p, h), T, n_paths, rng)
    return _mc_from_samples(
        np.exp(-r * T), S0 * np.exp(x), K, None, "MC Esscher (direct under Q)"
    )



# --------------------------------------------------------------------------- #
# Black-Scholes
# --------------------------------------------------------------------------- #


def _d1_d2(S0: float, K, r: float, T: float, sigma, q: float = 0.0):
    k = np.asarray(K, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    vol = sig * np.sqrt(T)
    d1 = (np.log(S0 / k) + (r - q + 0.5 * sig**2) * T) / vol
    return d1, d1 - vol


def bs_call(S0: float, K, r: float, T: float, sigma, q: float = 0.0):
    """Black-Scholes call price -- eq. (1)-(2) of the paper, with dividend yield."""
    d1, d2 = _d1_d2(S0, K, r, T, sigma, q)
    return S0 * np.exp(-q * T) * stats.norm.cdf(d1) - np.asarray(K, float) * np.exp(
        -r * T
    ) * stats.norm.cdf(d2)


def bs_put(S0: float, K, r: float, T: float, sigma, q: float = 0.0):
    d1, d2 = _d1_d2(S0, K, r, T, sigma, q)
    return np.asarray(K, float) * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * np.exp(
        -q * T
    ) * stats.norm.cdf(-d1)


def bs_vega(S0: float, K, r: float, T: float, sigma, q: float = 0.0):
    """dC/dsigma -- used to weight the calibration objective."""
    d1, _ = _d1_d2(S0, K, r, T, sigma, q)
    return S0 * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    price: float,
    S0: float,
    K: float,
    r: float,
    T: float,
    q: float = 0.0,
    *,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Black-Scholes implied volatility of a call, or NaN if the price is not
    attainable (outside the no-arbitrage band).

    Every model/market comparison in the results is done on this scale: prices
    across strikes and maturities are not comparable, implied vols are.
    """
    if not np.isfinite(price) or price <= 0.0:
        return float("nan")
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper = S0 * np.exp(-q * T)
    if price <= intrinsic + 1e-12 or price >= upper - 1e-12:
        return float("nan")

    def f(s: float) -> float:
        return float(bs_call(S0, K, r, T, s, q)) - price

    try:
        return float(optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200))
    except ValueError:
        return float("nan")

