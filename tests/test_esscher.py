"""Section 4.2: the Esscher transform and the risk-neutral measure."""

import numpy as np
import pytest

from vg import VGParams, cf, mgf, mgf_domain
from vg.esscher import (
    H_ratio,
    esscher_cf,
    esscher_domain,
    esscher_h_star,
    esscher_tilt,
    risk_neutral_params,
)
from vg.process import mgf_bracket, simulate_terminal

PARAMS = [
    VGParams(-0.14, 0.12, 0.20),
    VGParams(0.05, 0.30, 1.00),
    VGParams(-0.30, 0.25, 0.05),
    VGParams(0.00, 0.20, 0.50),
    VGParams(-0.02, 0.18, 0.02),
]
RATES = [0.0, 0.01, 0.03, 0.055, -0.005]


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("rate", RATES)
def test_h_star_solves_the_fixed_point(p, rate):
    """H(h*) = D_{h*}/D_{h*+1} must equal exp(rate * nu)."""
    h = esscher_h_star(p, rate)
    assert float(H_ratio(p, h)) == pytest.approx(np.exp(rate * p.nu), rel=1e-12)


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("rate", RATES)
def test_h_star_lies_in_the_admissible_interval(p, rate):
    lo, hi = esscher_domain(p)
    h = esscher_h_star(p, rate)
    assert lo < h < hi
    assert float(mgf_bracket(p, h)) > 0
    assert float(mgf_bracket(p, h + 1.0)) > 0


@pytest.mark.parametrize("p", PARAMS)
def test_h_star_does_not_depend_on_maturity(p):
    """The point of the e^{r nu} form of the fixed-point equation.

    A single measure has to price every maturity, so h* cannot depend on t.
    The martingale property is checked at several t with one h*.
    """
    rate = 0.03
    qp, h = risk_neutral_params(p, rate)
    for t in [0.05, 0.25, 1.0, 3.0]:
        # M(h*+1, t) / M(h*, t) == exp(rate * t) for every t with the same h*.
        ratio = float(mgf(p, h + 1.0, t) / mgf(p, h, t))
        assert ratio == pytest.approx(np.exp(rate * t), rel=1e-11)


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("rate", RATES)
def test_tilted_process_is_a_martingale(p, rate):
    """E^Q[exp(X_t)] = exp(rate * t) for the tilted parameters, at every t."""
    qp, _ = risk_neutral_params(p, rate)
    for t in [0.1, 0.5, 2.0]:
        assert float(mgf(qp, 1.0, t)) == pytest.approx(np.exp(rate * t), rel=1e-11)


@pytest.mark.parametrize("p", PARAMS)
def test_tilt_reparametrisation_matches_the_tilted_cf(p):
    """The closed-form tilted parameters must reproduce eq. (7) exactly.

    This is the independent check on the whole Esscher construction: eq. (7) is
    a ratio of MGF brackets, while the tilt is a claim about *which VG process*
    that ratio is the characteristic function of.
    """
    lo, hi = esscher_domain(p)
    u = np.linspace(-40.0, 40.0, 401)
    for frac in [0.2, 0.5, 0.8]:
        h = lo + frac * (hi - lo)
        tilted = esscher_tilt(p, h)
        for t in [0.1, 0.7, 2.0]:
            a = np.asarray(esscher_cf(p, u, t, h))
            b = np.asarray(cf(tilted, u, t))
            assert np.abs(a - b).max() < 1e-11


@pytest.mark.parametrize("p", PARAMS)
def test_tilt_preserves_nu_and_reduces_to_identity_at_h_zero(p):
    tilted = esscher_tilt(p, 0.0)
    assert tilted.nu == p.nu
    assert tilted.theta == pytest.approx(p.theta)
    assert tilted.sigma == pytest.approx(p.sigma)


@pytest.mark.parametrize("p", PARAMS)
def test_esscher_density_reweights_correctly(p):
    """Monte Carlo check of dQ^h/dP = exp(h X_t)/M(h, t).

    Reweighted P-samples must reproduce the moments of the tilted process.
    """
    rate, t = 0.03, 0.5
    h = esscher_h_star(p, rate)
    tilted = esscher_tilt(p, h)
    rng = np.random.default_rng(5)
    x = simulate_terminal(p, t, 2_000_000, rng)
    w = np.exp(h * x + (t / p.nu) * np.log(float(mgf_bracket(p, h))))
    assert w.mean() == pytest.approx(1.0, rel=0.02)
    from vg import moments

    assert np.average(x, weights=w) == pytest.approx(moments(tilted, t)["mean"], abs=0.01)


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("rate", RATES)
def test_martingale_condition_pins_one_combination_of_the_q_parameters(p, rate):
    """theta_Q + sigma_Q^2/2 = (1 - e^{-rate nu})/nu, for every physical input.

    Under S_t = S_0 exp(X_t) the price has no drift of its own, so the whole
    martingale condition falls on the law of X.  One combination of the tilted
    parameters is therefore fixed, leaving only two free.
    """
    from vg.esscher import identifiability_constraint

    qp, _ = risk_neutral_params(p, rate)
    assert qp.theta + 0.5 * qp.sigma**2 == pytest.approx(
        identifiability_constraint(p.nu, rate), rel=1e-10, abs=1e-14
    )


def test_physical_theta_is_not_identifiable_from_option_prices():
    """Different (theta, sigma) with the same (sigma_Q, nu) price identically.

    The consequence of the constraint above: option prices see the physical
    parameters only through the tilted ones, so an entire one-parameter family
    of physical parameters is observationally equivalent.  Calibrating three
    parameters through the Esscher transform would be fitting a flat direction,
    which is why `vg.calibration` fixes theta = 0 for this measure.
    """
    from scipy.optimize import brentq

    from vg import vg_call_esscher_fourier

    nu, rate, s0, t = 0.35, 0.03, 100.0, 0.5
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    target = risk_neutral_params(VGParams(-0.20, 0.15, nu), rate)[0].sigma

    prices, sigmas = [], []
    for theta in [-0.60, -0.35, -0.20, 0.0, 0.20]:
        sigma = brentq(
            lambda s: risk_neutral_params(VGParams(theta, s, nu), rate)[0].sigma - target,
            0.02, 1.5, xtol=1e-14,
        )
        sigmas.append(sigma)
        prices.append(np.asarray(vg_call_esscher_fourier(s0, strikes, rate, t, VGParams(theta, sigma, nu))))

    assert max(sigmas) - min(sigmas) > 0.05, "the family should be genuinely spread out"
    for other in prices[1:]:
        assert np.abs(other - prices[0]).max() < 1e-10


def test_esscher_smile_is_nearly_symmetric_whatever_the_physical_theta():
    """The practical cost of the identifiability constraint.

    Because theta_Q is pinned near -sigma_Q^2/2, this specification produces
    curvature but almost no skew -- under a fifth of a volatility point across
    80%-120% moneyness -- no matter what the physical theta is, including when
    the physical skew is strongly negative.  Index option markets are steeply
    downward sloping, so the model cannot reproduce their shape.
    """
    from vg import implied_vol, vg_call_esscher_fourier

    s0, rate, t, nu = 100.0, 0.03, 0.5, 0.35
    strikes = np.array([80.0, 120.0])

    def skew(prices):
        iv = [implied_vol(float(c), s0, float(k), rate, t) for c, k in zip(prices, strikes)]
        return iv[0] - iv[1]

    skews = []
    for theta in [-0.6, -0.3, 0.0, 0.3]:
        p = VGParams(theta, 0.15, nu)
        skews.append(skew(np.asarray(vg_call_esscher_fourier(s0, strikes, rate, t, p))))
        assert abs(skews[-1]) < 0.02, f"theta={theta}: skew {skews[-1]} unexpectedly large"

    # And it barely moves with theta at all -- theta is not a skew control here.
    assert max(skews) - min(skews) < 0.02


def test_zero_rate_has_the_closed_form_solution():
    """At rate = 0 the quadratic degenerates and h* = -1/2 - theta/sigma^2."""
    for p in PARAMS:
        h = esscher_h_star(p, 0.0)
        assert h == pytest.approx(-0.5 - p.theta / p.sigma**2, rel=1e-10)


def test_empty_esscher_domain_is_reported():
    """Needs h2 - h1 > 1; a wide, fat-tailed parameter set violates it."""
    p = VGParams(0.0, 3.0, 4.0)
    h1, h2 = mgf_domain(p)
    assert h2 - h1 < 1.0
    with pytest.raises(ValueError, match="empty Esscher domain"):
        esscher_domain(p)


def test_tilt_outside_domain_raises():
    p = VGParams(-0.1, 0.2, 0.3)
    _, hi = mgf_domain(p)
    with pytest.raises(ValueError):
        esscher_tilt(p, hi + 1.0)
