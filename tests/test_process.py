"""The VG process itself: MGF, characteristic function, moments, density."""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from vg import (
    VGParams,
    cf,
    cumulants,
    density,
    mgf,
    mgf_domain,
    moments,
    simulate_paths,
    simulate_terminal,
)

# Parameter sets spanning the regimes that behave differently numerically:
# t/nu large (nearly Gaussian), t/nu = 1/2 (log singularity at the origin),
# t/nu < 1/2 (power singularity), strong negative skew.
CASES = [
    (VGParams(-0.14, 0.12, 0.20), 0.5),
    (VGParams(0.00, 0.20, 0.50), 0.25),
    (VGParams(0.05, 0.30, 1.00), 0.10),
    (VGParams(-0.30, 0.25, 0.05), 1.00),
    (VGParams(-0.02, 0.18, 0.02), 0.75),
]


@pytest.mark.parametrize("p,t", CASES)
def test_mgf_matches_monte_carlo(p, t):
    """Restricted to |u| <= 2.5: exp(u X) has enormous Monte Carlo variance
    further out, so a mismatch there would be a statement about the sampler's
    tail, not about the MGF."""
    rng = np.random.default_rng(0)
    x = simulate_terminal(p, t, 2_000_000, rng)
    lo, hi = mgf_domain(p)
    for u in [-2.5, -1.0, 0.5, 1.0, 2.5]:
        if not (lo < u < hi):
            continue
        got = np.mean(np.exp(u * x))
        expected = float(mgf(p, u, t))
        assert got == pytest.approx(expected, rel=0.02), f"u={u}"


@pytest.mark.parametrize("p,t", CASES)
def test_cf_matches_monte_carlo(p, t):
    rng = np.random.default_rng(1)
    x = simulate_terminal(p, t, 2_000_000, rng)
    u = np.array([0.5, 2.0, 5.0, 15.0])
    got = np.mean(np.exp(1j * np.outer(u, x)), axis=1)
    expected = np.asarray(cf(p, u, t))
    assert np.abs(got - expected).max() < 5e-3


@pytest.mark.parametrize("p,t", CASES)
def test_moments_match_monte_carlo(p, t):
    rng = np.random.default_rng(2)
    x = simulate_terminal(p, t, 4_000_000, rng)
    m = moments(p, t)
    assert x.mean() == pytest.approx(m["mean"], abs=4e-3 * m["std"] + 1e-6)
    assert x.var(ddof=1) == pytest.approx(m["var"], rel=0.02)
    assert stats.skew(x) == pytest.approx(m["skewness"], abs=0.05)
    assert stats.kurtosis(x) == pytest.approx(m["excess_kurtosis"], rel=0.10, abs=0.05)


@pytest.mark.parametrize("p,t", CASES)
def test_cumulants_are_the_taylor_coefficients_of_log_mgf(p, t):
    """kappa_n = n! * [u^n] log M(u).

    Read off a Chebyshev fit of log M rather than by finite differences: third
    and fourth differences lose most of their significant digits to
    cancellation, so an FD check would be testing the differencing scheme, not
    the cumulants.
    """
    from numpy.polynomial import chebyshev as cheb
    from scipy.special import factorial

    k = cumulants(p, t)
    lo, hi = mgf_domain(p)
    half_width = 0.2 * min(abs(lo), hi, 5.0)

    n = 61
    u = half_width * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    poly = cheb.cheb2poly(cheb.chebfit(u, np.log(np.asarray(mgf(p, u, t))), 14))

    for order, key in enumerate(["k1", "k2", "k3", "k4"], start=1):
        got = float(factorial(order) * poly[order])
        assert got == pytest.approx(k[key], rel=1e-6, abs=1e-15), key


@pytest.mark.parametrize("p,t", CASES)
def test_density_integrates_to_one(p, t):
    """Split at 0: for t/nu <= 1/2 the density is singular there (but integrable)."""
    lo = quad(lambda z: density(p, z, t), -12.0, 0.0, limit=800)[0]
    hi = quad(lambda z: density(p, z, t), 0.0, 12.0, limit=800)[0]
    assert lo + hi == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize("p,t", CASES)
def test_density_reproduces_mean_and_variance(p, t):
    m = moments(p, t)
    mean = sum(
        quad(lambda z: z * density(p, z, t), a, b, limit=800)[0]
        for a, b in [(-12.0, 0.0), (0.0, 12.0)]
    )
    second = sum(
        quad(lambda z: z * z * density(p, z, t), a, b, limit=800)[0]
        for a, b in [(-12.0, 0.0), (0.0, 12.0)]
    )
    assert mean == pytest.approx(m["mean"], abs=1e-7)
    assert second - mean**2 == pytest.approx(m["var"], rel=1e-5)


@pytest.mark.parametrize("p,t", CASES)
def test_density_is_positive_and_finite_away_from_origin(p, t):
    x = np.concatenate([np.linspace(-2.0, -1e-6, 200), np.linspace(1e-6, 2.0, 200)])
    f = density(p, x, t)
    assert np.all(np.isfinite(f))
    assert np.all(f > 0)


def test_gamma_clock_has_the_right_law():
    """G(t) ~ Gamma(t/nu, nu): mean t, variance nu*t, monotone paths."""
    p = VGParams(-0.1, 0.2, 0.3)
    rng = np.random.default_rng(3)
    _, _, g = simulate_paths(p, T=1.0, n_steps=250, n_paths=20_000, rng=rng)
    assert np.all(np.diff(g, axis=1) >= 0)  # subordinator increases
    assert g[:, -1].mean() == pytest.approx(1.0, rel=0.02)
    assert g[:, -1].var(ddof=1) == pytest.approx(p.nu * 1.0, rel=0.05)


def test_paths_start_at_zero_and_terminal_law_matches():
    p = VGParams(-0.2, 0.25, 0.4)
    rng = np.random.default_rng(4)
    _, x, _ = simulate_paths(p, T=1.0, n_steps=50, n_paths=200_000, rng=rng)
    assert np.all(x[:, 0] == 0.0)
    m = moments(p, 1.0)
    assert x[:, -1].mean() == pytest.approx(m["mean"], abs=0.02)
    assert x[:, -1].var(ddof=1) == pytest.approx(m["var"], rel=0.03)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        VGParams(0.0, -1.0, 0.2)
    with pytest.raises(ValueError):
        VGParams(0.0, 0.2, 0.0)
    with pytest.raises(ValueError):
        VGParams(np.nan, 0.2, 0.2)


def test_mgf_outside_domain_raises():
    p = VGParams(-0.1, 0.2, 0.3)
    _, hi = mgf_domain(p)
    with pytest.raises(ValueError):
        mgf(p, hi + 1.0, 0.5)
