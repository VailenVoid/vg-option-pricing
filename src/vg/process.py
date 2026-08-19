"""The Variance-Gamma process.

    X(t) = theta * G(t) + sigma * W(G(t)),      G(t) ~ Gamma(shape = t/nu, scale = nu)

with W a standard Brownian motion independent of the Gamma subordinator G.
This is Definition 3.2 of the paper.

Everything in this module is under the *physical* parametrisation (theta, sigma, nu);
the change of measure lives in `vg.esscher`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

__all__ = [
    "VGParams",
    "mgf_bracket",
    "mgf",
    "mgf_domain",
    "cf",
    "cumulants",
    "moments",
    "simulate_terminal",
    "simulate_paths",
    "log_density",
    "density",
]


@dataclass(frozen=True)
class VGParams:
    """Parameters of a Variance-Gamma process.

    theta : drift of the subordinated Brownian motion (controls skewness)
    sigma : volatility of the subordinated Brownian motion (> 0)
    nu    : variance rate of the Gamma clock (> 0; controls tail thickness)
    """

    theta: float
    sigma: float
    nu: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.theta, self.sigma, self.nu]).all():
            raise ValueError(f"non-finite VG parameters: {self}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.nu <= 0.0:
            raise ValueError(f"nu must be > 0, got {self.nu}")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.theta, self.sigma, self.nu)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"VG(theta={self.theta:+.5f}, sigma={self.sigma:.5f}, nu={self.nu:.5f})"


# --------------------------------------------------------------------------- #
# Moment generating function
# --------------------------------------------------------------------------- #


def mgf_bracket(p: VGParams, z):
    """The bracket D_z = 1 - theta*nu*z - (1/2)*sigma^2*nu*z^2.

    The MGF of X(t) is D_z**(-t/nu) (Lemma 3.1).  Keeping the bracket as its own
    function is convenient because the whole Esscher machinery is expressed
    through it: the martingale condition, the tilted parameters and the
    characteristic function are all ratios of D at different arguments.

    Accepts real or complex `z`, scalar or array.
    """
    z = np.asarray(z)
    return 1.0 - p.theta * p.nu * z - 0.5 * p.sigma**2 * p.nu * z**2


def mgf_domain(p: VGParams) -> tuple[float, float]:
    """Open interval (h1, h2) on which the MGF of X(t) is finite.

    h1, h2 are the roots of D_h = 0:

        h_{1,2} = -theta/sigma^2 -+ sqrt(theta^2/sigma^4 + 2/(nu*sigma^2)).
    """
    centre = -p.theta / p.sigma**2
    radius = np.sqrt(p.theta**2 / p.sigma**4 + 2.0 / (p.nu * p.sigma**2))
    return float(centre - radius), float(centre + radius)


def mgf(p: VGParams, u, t: float):
    """M_X(u, t) = E[exp(u * X(t))] = D_u**(-t/nu), for u in (h1, h2)."""
    d = mgf_bracket(p, u)
    if np.any(np.real(d) <= 0.0) and not np.iscomplexobj(d):
        raise ValueError(
            f"MGF argument outside the domain {mgf_domain(p)}: bracket = {d}"
        )
    return d ** (-t / p.nu)


def cf(p: VGParams, u, t: float):
    """Characteristic function phi(u, t) = E[exp(i*u*X(t))] = D_{iu}**(-t/nu).

    Branch-cut note.  For a general Levy model, raising a complex bracket to a
    non-integer power with the *principal* branch of the logarithm can produce
    spurious 2*pi jumps, and the standard remedy is to unwrap the argument along
    the integration grid.  Here it is not needed and we can prove it:

        Re D_{h + iu} = D_h + (1/2) * sigma^2 * nu * u^2 > 0

    whenever D_h > 0, i.e. whenever h lies in the MGF domain.  A complex number
    with strictly positive real part has principal argument in (-pi/2, pi/2), so
    the principal branch is already the continuous one.  We assert this rather
    than assume it.
    """
    u = np.asarray(u)
    d = mgf_bracket(p, 1j * u)
    _assert_positive_real_part(d)
    return d ** (-t / p.nu)


def _assert_positive_real_part(d) -> None:
    d = np.asarray(d)
    if np.any(np.real(d) <= 0.0):
        raise ValueError(
            "characteristic-function bracket left the right half-plane; the "
            "principal branch is no longer continuous here"
        )


# --------------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------------- #


def cumulants(p: VGParams, t: float) -> dict[str, float]:
    """First four cumulants of X(t).

    Obtained by expanding the cumulant generating function
    K(u) = -(t/nu) * log(1 - theta*nu*u - (1/2)*sigma^2*nu*u^2) in powers of u.
    """
    th, s, nu = p.theta, p.sigma, p.nu
    k1 = th * t
    k2 = (s**2 + nu * th**2) * t
    k3 = (3.0 * s**2 * nu * th + 2.0 * nu**2 * th**3) * t
    k4 = (3.0 * s**4 * nu + 12.0 * s**2 * th**2 * nu**2 + 6.0 * th**4 * nu**3) * t
    return {"k1": k1, "k2": k2, "k3": k3, "k4": k4}


def moments(p: VGParams, t: float) -> dict[str, float]:
    """Mean, variance, skewness and excess kurtosis of X(t)."""
    k = cumulants(p, t)
    var = k["k2"]
    return {
        "mean": k["k1"],
        "var": var,
        "std": np.sqrt(var),
        "skewness": k["k3"] / var**1.5,
        "excess_kurtosis": k["k4"] / var**2,
    }


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def simulate_terminal(
    p: VGParams, t: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact i.i.d. draws of X(t).

    No time discretisation is involved: we draw the Gamma clock G(t) directly
    from Gamma(t/nu, nu) and then X(t) | G(t)=g ~ N(theta*g, sigma^2*g).
    """
    if t <= 0.0:
        raise ValueError("t must be > 0")
    g = rng.gamma(shape=t / p.nu, scale=p.nu, size=n)
    z = rng.standard_normal(n)
    return p.theta * g + p.sigma * np.sqrt(g) * z


def simulate_paths(
    p: VGParams,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate VG paths on a uniform grid of [0, T].

    Returns (times, X, G) with X and G of shape (n_paths, n_steps + 1).
    The increments are exact (the grid only controls how finely we *observe*
    the path, it introduces no discretisation error at the grid points).
    """
    dt = T / n_steps
    dg = rng.gamma(shape=dt / p.nu, scale=p.nu, size=(n_paths, n_steps))
    dz = rng.standard_normal((n_paths, n_steps))
    dx = p.theta * dg + p.sigma * np.sqrt(dg) * dz
    zeros = np.zeros((n_paths, 1))
    x = np.concatenate([zeros, np.cumsum(dx, axis=1)], axis=1)
    g = np.concatenate([zeros, np.cumsum(dg, axis=1)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), x, g


# --------------------------------------------------------------------------- #
# Closed-form density (Madan, Carr & Chang 1998, eq. 6)
# --------------------------------------------------------------------------- #


def log_density(p: VGParams, x, t: float) -> np.ndarray:
    """log f_{X(t)}(x) using the modified Bessel function of the second kind.

        f(x) = 2 exp(theta x / sigma^2)
               / ( nu^{t/nu} sqrt(2 pi) sigma Gamma(t/nu) )
               * ( x^2 / (2 sigma^2/nu + theta^2) )^{ t/(2 nu) - 1/4 }
               * K_{t/nu - 1/2}( |x| sqrt(2 sigma^2/nu + theta^2) / sigma^2 )

    Evaluated through `scipy.special.kve` (exponentially scaled Bessel K) so
    that the large-|x| regime does not underflow.

    Behaviour at x = 0.  Writing m = t/nu - 1/2, the two x-dependent factors are
    (|x|/A)^m and K_m(|x| A / sigma^2), and for m > 0 each blows up (or vanishes)
    as x -> 0 while the product tends to the finite limit

        (1/2) Gamma(m) (2 sigma^2 / A^2)^m .

    Evaluating them separately in floating point overflows, so near z = 0 we
    switch to the small-argument expansion K_m(z) ~ (1/2) Gamma(m) (2/z)^m,
    which cancels the |x| powers analytically.  For m < 0 the density really
    does diverge at the origin (the singularity is integrable), and for m = 0 it
    diverges logarithmically; both are returned as such rather than papered over.
    """
    th, s, nu = p.theta, p.sigma, p.nu
    a = t / nu
    m = a - 0.5
    scale = np.sqrt(2.0 * s**2 / nu + th**2)

    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    z = ax * scale / s**2
    # Only used inside branches where the coefficient multiplying it is either
    # zero or the branch itself is discarded; the floor keeps 0 * (-inf) = nan
    # from leaking out of the unselected branch of np.where.
    log_ax = np.log(np.maximum(ax, 1e-300))

    prefactor = (
        np.log(2.0)
        + th * x / s**2
        - a * np.log(nu)
        - 0.5 * np.log(2.0 * np.pi)
        - np.log(s)
        - special.gammaln(a)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        # Regular branch: (|x|/A)^m * K_m(z), computed in logs via kve.
        regular = m * (log_ax - np.log(scale)) + np.log(special.kve(m, z)) - z

        # Small-z branch, with the |x| powers cancelled analytically.
        abs_m = abs(m)
        if abs_m > 1e-12:
            small = (
                (m - abs_m) * log_ax
                - (m + abs_m) * np.log(scale)
                - np.log(2.0)
                + special.gammaln(abs_m)
                + abs_m * (np.log(2.0) + 2.0 * np.log(s))
            )
        else:  # m == 0: K_0(z) ~ -log(z/2) - Euler gamma
            small = np.log(np.maximum(-np.log(0.5 * z) - 0.5772156649015329, 1e-300))

        term = np.where(z < 1e-8, small, regular)

    return prefactor + term


def density(p: VGParams, x, t: float) -> np.ndarray:
    """f_{X(t)}(x); see `log_density`."""
    return np.exp(log_density(p, x, t))
