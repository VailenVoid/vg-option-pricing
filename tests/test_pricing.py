"""Sections 4.3-4.5: Fourier inversion against Monte Carlo, and no-arbitrage."""

import numpy as np
import pytest
from scipy.integrate import quad

from vg import (
    VGParams,
    bs_call,
    cf,
    density,
    gil_pelaez_tail,
    implied_vol,
    vg_call_esscher_fourier,
    vg_call_mc_direct,
    vg_call_mc_esscher,
)
from vg.esscher import esscher_cf, esscher_h_star
from vg.pricing import (
    call_price_from_cf,
    vg_call_density_quad,
    vg_put_esscher_fourier,
)

S0, R, Q = 100.0, 0.03, 0.0

CASES = [
    (VGParams(-0.14, 0.12, 0.20), 0.5),
    (VGParams(-0.30, 0.25, 0.05), 1.0),
    (VGParams(0.05, 0.30, 1.00), 0.25),
    (VGParams(-0.02, 0.18, 0.02), 0.75),
]


# --------------------------------------------------------------------------- #
# Gil-Pelaez inversion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p,t", CASES)
def test_gil_pelaez_matches_the_closed_form_density(p, t):
    """Two entirely independent routes to the same CDF.

    Gil-Pelaez inverts the characteristic function; the reference integrates the
    Bessel-K density.  The tolerance is the routine's *own* reported error
    estimate, so this simultaneously checks that the answer is right and that
    the error it advertises is honest -- which matters because for small T/nu it
    cannot reach full precision within any sane node budget.
    """
    tight = {"epsabs": 1e-14, "epsrel": 1e-14, "limit": 2000}
    for x in [-0.4, -0.15, -0.02, 0.08, 0.3]:
        tail, info = gil_pelaez_tail(lambda u: cf(p, u, t), x, return_info=True)
        ref = quad(lambda z: density(p, z, t), -15.0, min(x, 0.0), **tight)[0]
        if x > 0:
            ref += quad(lambda z: density(p, z, t), 0.0, x, **tight)[0]
        # The routine's own claim, plus a floor for the reference quadrature's
        # roundoff.  A factor of 3 of slack, not orders of magnitude: the point
        # is that the advertised accuracy is real.
        tol = max(3.0 * info.error_estimate, 1e-13)
        assert 1.0 - float(tail) == pytest.approx(ref, abs=tol), f"x={x}, info={info}"


@pytest.mark.parametrize("p,t", CASES)
def test_quadrature_reports_convergence_honestly(p, t):
    """converged=True must mean it really did converge."""
    x = np.array([-0.3, 0.0, 0.2])
    _, info = gil_pelaez_tail(lambda u: cf(p, u, t), x, return_info=True)
    if t / p.nu >= 1.0:
        assert info.converged, f"easy case reported non-convergence: {info}"
        assert info.error_estimate < 1e-10
    else:
        # Heavy-tailed regime: not converging is allowed, lying about it is not.
        assert info.error_estimate > 0.0
        assert np.isfinite(info.error_estimate)


@pytest.mark.parametrize("p,t", CASES)
def test_reported_error_is_an_upper_bound_on_the_real_error(p, t):
    """The claim the whole numerical section rests on.

    Everywhere else in the suite the reported `error_estimate` is used as the
    tolerance, so it has to actually bound the truth.  Measured against the
    independent density quadrature at machine-tight settings.
    """
    tight = {"epsabs": 1e-14, "epsrel": 1e-14, "limit": 2000}
    for x in [-0.35, -0.05, 0.12, 0.4]:
        tail, info = gil_pelaez_tail(lambda u: cf(p, u, t), x, return_info=True)
        ref = quad(lambda z: density(p, z, t), -15.0, min(x, 0.0), **tight)[0]
        if x > 0:
            ref += quad(lambda z: density(p, z, t), 0.0, x, **tight)[0]
        actual = abs((1.0 - float(tail)) - ref)
        budget = max(info.error_estimate, 1e-13)  # reference quad's own roundoff
        assert actual <= budget, (
            f"under-reported at x={x}: real error {actual:.3e} exceeds the "
            f"advertised {info.error_estimate:.3e} ({info})"
        )


@pytest.mark.parametrize("p,t", CASES)
def test_gil_pelaez_is_a_valid_cdf(p, t):
    """Range scaled to the law's own spread -- these distributions differ in
    standard deviation by a factor of three and in excess kurtosis by a factor
    of 150, so a fixed +-1.5 window would be a tail test for some and not for
    others."""
    from vg import moments

    sd = moments(p, t)["std"]
    x = np.linspace(-12.0 * sd, 12.0 * sd, 121)
    tail = np.asarray(gil_pelaez_tail(lambda u: cf(p, u, t), x))
    cdf = 1.0 - tail
    assert np.all(cdf > -1e-8) and np.all(cdf < 1.0 + 1e-8)
    assert np.all(np.diff(cdf) > -1e-8)  # non-decreasing
    assert cdf[0] < 1e-3 and cdf[-1] > 1.0 - 1e-3


def test_gil_pelaez_matches_the_normal_cdf():
    """Sanity anchor on a distribution with a known CDF."""
    from scipy import stats

    mu, sd = 0.02, 0.3
    normal_cf = lambda u: np.exp(1j * u * mu - 0.5 * sd**2 * np.asarray(u) ** 2)
    x = np.array([-1.0, -0.2, 0.0, 0.15, 0.9])
    got = 1.0 - np.asarray(gil_pelaez_tail(normal_cf, x))
    assert np.abs(got - stats.norm.cdf(x, mu, sd)).max() < 1e-12


# --------------------------------------------------------------------------- #
# Fourier vs Monte Carlo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p,t", CASES)
def test_fourier_matches_direct_monte_carlo(p, t):
    """Theorem 4.1 against simulation under Q. Tolerance is 4 standard errors."""
    k = np.array([80.0, 92.0, 100.0, 108.0, 125.0])
    ref = np.asarray(vg_call_esscher_fourier(S0, k, R, t, p, Q))
    rng = np.random.default_rng(11)
    mc = vg_call_mc_direct(S0, k, R, t, p, 3_000_000, rng, Q)
    z = (np.asarray(mc.price) - ref) / np.asarray(mc.stderr)
    assert np.abs(z).max() < 4.0, f"z-scores {z}"


@pytest.mark.parametrize("p,t", CASES[:2])
def test_fourier_matches_importance_sampling_monte_carlo(p, t):
    """Equation (8): reweighted P-paths must give the same price."""
    k = np.array([92.0, 100.0, 108.0])
    ref = np.asarray(vg_call_esscher_fourier(S0, k, R, t, p, Q))
    rng = np.random.default_rng(12)
    mc = vg_call_mc_esscher(S0, k, R, t, p, 3_000_000, rng, Q)
    z = (np.asarray(mc.price) - ref) / np.asarray(mc.stderr)
    assert np.abs(z).max() < 4.0, f"z-scores {z}"


@pytest.mark.parametrize("p,t", CASES[:2])
def test_direct_simulation_beats_importance_sampling_on_variance(p, t):
    """The tilted-parameter estimator should be the lower-variance one.

    Both are unbiased for the same price; the reweighted estimator pays for its
    convenience with the variance of the Esscher weight.
    """
    k = 100.0
    rng = np.random.default_rng(13)
    a = vg_call_mc_direct(S0, k, R, t, p, 500_000, rng, Q)
    b = vg_call_mc_esscher(S0, k, R, t, p, 500_000, rng, Q)
    assert a.stderr < b.stderr


@pytest.mark.parametrize("p,t", CASES)
def test_esscher_price_matches_the_generic_cf_pricer(p, t):
    """Theorem 4.1 is the general two-probability formula in disguise."""
    h = esscher_h_star(p, R - Q)
    generic = call_price_from_cf(
        lambda u: esscher_cf(p, u, t, h), S0, np.array([85.0, 100.0, 115.0]), R, t, Q
    )
    special = vg_call_esscher_fourier(S0, np.array([85.0, 100.0, 115.0]), R, t, p, Q)
    assert np.abs(np.asarray(generic) - np.asarray(special)).max() < 1e-9


@pytest.mark.parametrize("p,t", CASES)
def test_fourier_matches_the_density_quadrature(p, t):
    """The third, fully independent route: integrate the payoff against the
    closed-form tilted density.  Agreement across all four regimes -- including
    the two where the density is unbounded at the origin -- is the strongest
    single check in the suite.
    """
    k = np.array([85.0, 100.0, 115.0])
    fourier = np.asarray(vg_call_esscher_fourier(S0, k, R, t, p, Q))
    by_density = np.asarray(vg_call_density_quad(S0, k, R, t, p, Q))
    # Sub-penny everywhere; near machine precision when T/nu is not small.
    tol = 1e-9 if t / p.nu >= 1.0 else 5e-6
    assert np.abs(fourier - by_density).max() < tol


@pytest.mark.parametrize("p,t", CASES[:2])
def test_density_quadrature_matches_monte_carlo(p, t):
    k = np.array([92.0, 100.0, 110.0])
    ref = np.asarray(vg_call_density_quad(S0, k, R, t, p, Q))
    rng = np.random.default_rng(15)
    mc = vg_call_mc_direct(S0, k, R, t, p, 3_000_000, rng, Q)
    z = (np.asarray(mc.price) - ref) / np.asarray(mc.stderr)
    assert np.abs(z).max() < 4.0, f"z-scores {z}"


# --------------------------------------------------------------------------- #
# Structural properties
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p,t", CASES)
@pytest.mark.parametrize("q", [0.0, 0.018])
def test_no_arbitrage_bounds_monotonicity_and_convexity(p, t, q):
    k = np.linspace(60.0, 150.0, 46)
    c = np.asarray(vg_call_esscher_fourier(S0, k, R, t, p, q))
    lower = np.maximum(S0 * np.exp(-q * t) - k * np.exp(-R * t), 0.0)
    upper = S0 * np.exp(-q * t)
    assert np.all(c > lower - 1e-8)
    assert np.all(c < upper + 1e-8)
    assert np.all(np.diff(c) < 1e-8)                 # decreasing in K
    assert np.all(np.diff(c, 2) > -1e-7)             # convex in K


@pytest.mark.parametrize("p,t", CASES)
def test_put_call_parity(p, t):
    """C - P = S e^{-qT} - K e^{-rT}, with both legs computed independently."""
    k = np.array([80.0, 100.0, 125.0])
    for q in [0.0, 0.018]:
        c = np.asarray(vg_call_esscher_fourier(S0, k, R, t, p, q))
        put = np.asarray(vg_put_esscher_fourier(S0, k, R, t, p, q))
        expected = S0 * np.exp(-q * t) - k * np.exp(-R * t)
        assert np.abs(c - put - expected).max() < 1e-8


@pytest.mark.parametrize("t", [0.25, 1.0])
def test_vg_converges_to_black_scholes_as_nu_goes_to_zero(t):
    """Section 3.3: as nu -> 0 the Gamma clock becomes deterministic."""
    sigma = 0.20
    k = np.array([85.0, 100.0, 115.0])
    ref = np.asarray(bs_call(S0, k, R, t, sigma, Q))
    errs = []
    for nu in [0.2, 0.05, 0.01, 0.002]:
        c = np.asarray(vg_call_esscher_fourier(S0, k, R, t, VGParams(0.0, sigma, nu), Q))
        errs.append(np.abs(c - ref).max())
    assert errs == sorted(errs, reverse=True), f"not monotone: {errs}"
    assert errs[-1] < 5e-3


def test_dividend_yield_only_enters_through_the_cost_of_carry():
    """Everything about q flows through the carry r - q and the stock discount.

    Pricing with (r, q) must equal exp(-qT) times pricing with rate r-q and no
    dividend: both use the same h* (it depends on the carry alone), so the two
    tail probabilities are identical and only the discounting differs.

    Note this is *not* the same as pricing off a reduced spot S0*exp(-qT) at
    rate r -- that keeps the forward but changes the carry, hence h*, hence the
    law of X under Q.
    """
    p, t, q = VGParams(-0.14, 0.12, 0.20), 0.5, 0.02
    k = np.array([90.0, 100.0, 110.0])
    with_div = np.asarray(vg_call_esscher_fourier(S0, k, R, t, p, q))
    via_carry = np.exp(-q * t) * np.asarray(
        vg_call_esscher_fourier(S0, k, R - q, t, p, 0.0)
    )
    assert np.abs(with_div - via_carry).max() < 1e-10


def test_implied_vol_round_trips():
    checked = 0
    for sigma in [0.05, 0.15, 0.4, 1.2]:
        for k in [70.0, 100.0, 140.0]:
            price = float(bs_call(S0, k, R, 0.6, sigma, 0.01))
            intrinsic = max(S0 * np.exp(-0.01 * 0.6) - k * np.exp(-R * 0.6), 0.0)
            if price - intrinsic < 1e-8:
                continue  # no time value left in double precision; not invertible
            assert implied_vol(price, S0, k, R, 0.6, 0.01) == pytest.approx(sigma, rel=1e-8)
            checked += 1
    assert checked >= 9


def test_implied_vol_returns_nan_outside_the_no_arbitrage_band():
    assert np.isnan(implied_vol(0.0, S0, 100.0, R, 0.5))
    assert np.isnan(implied_vol(1e6, S0, 100.0, R, 0.5))
    assert np.isnan(implied_vol(float("nan"), S0, 100.0, R, 0.5))


def test_vg_smile_is_not_flat():
    """The whole point of the model: VG bends the implied-vol curve, Black-Scholes
    cannot.  The *slope* is a separate question, settled in test_esscher.py --
    under the Esscher specification the smile is nearly symmetric, so only
    curvature is asserted here."""
    p, t = VGParams(-0.20, 0.15, 0.35), 0.4
    k = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    c = np.asarray(vg_call_esscher_fourier(S0, k, R, t, p, Q))
    ivs = np.array([implied_vol(float(ci), S0, float(ki), R, t) for ci, ki in zip(c, k)])
    assert np.all(np.isfinite(ivs))
    assert ivs.max() - ivs.min() > 0.02        # a real smile, > 2 vol points
    assert ivs[2] < min(ivs[0], ivs[-1])       # convex: the wings sit above ATM


def test_scalar_and_array_strikes_agree():
    p, t = VGParams(-0.14, 0.12, 0.20), 0.5
    scalar = vg_call_esscher_fourier(S0, 103.0, R, t, p, Q)
    array = vg_call_esscher_fourier(S0, np.array([103.0]), R, t, p, Q)
    assert isinstance(scalar, float)
    assert scalar == pytest.approx(float(array[0]), rel=1e-12)


def test_non_risk_neutral_cf_is_rejected():
    """The generic pricer must refuse a characteristic function that is not a
    martingale measure -- silently pricing off one would be the worst failure."""
    p, t = VGParams(-0.14, 0.12, 0.20), 0.5
    with pytest.raises(ValueError, match="not risk-neutral"):
        call_price_from_cf(lambda u: cf(p, u, t), S0, 100.0, R, t, Q)

