"""Variance-Gamma option pricing.

Reference implementation for the seminar paper
"Variance-Gamma model i njegova primena" (Ivanovic, Miladinovic, 2026).

Layout
------
vg.process      VG process: MGF, characteristic function, moments, exact
                simulation, closed-form (Bessel-K) density.
vg.esscher      Esscher transform: h*, the tilted (risk-neutral) parameters.
vg.pricing      Gil-Pelaez Fourier inversion, Monte Carlo (importance sampling
                and direct-under-Q), Black-Scholes and implied volatility.
vg.estimation   Physical-parameter estimation from returns (moments + MLE).
vg.calibration  Risk-neutral calibration to a market option chain.
vg.data         Market data: option chains, spot history, implied forwards.
"""

from vg.process import (
    VGParams,
    mgf,
    mgf_bracket,
    mgf_domain,
    cf,
    cumulants,
    moments,
    simulate_terminal,
    simulate_paths,
    log_density,
    density,
)
from vg.esscher import (
    esscher_h_star,
    esscher_tilt,
    risk_neutral_params,
    esscher_cf,
)
from vg.pricing import (
    QuadInfo,
    gil_pelaez_tail,
    gil_pelaez_cdf,
    vg_call_esscher_fourier,
    vg_put_esscher_fourier,
    vg_call_density_quad,
    vg_call_mc_esscher,
    vg_call_mc_direct,
    MCResult,
    call_price_from_cf,
    bs_call,
    bs_put,
    bs_vega,
    implied_vol,
)

__all__ = [
    "VGParams",
    "mgf",
    "mgf_bracket",
    "mgf_domain",
    "cf",
    "cumulants",
    "moments",
    "simulate_terminal",
    "simulate_paths",
    "log_density",
    "density",
    "esscher_h_star",
    "esscher_tilt",
    "risk_neutral_params",
    "esscher_cf",
    "QuadInfo",
    "gil_pelaez_tail",
    "gil_pelaez_cdf",
    "vg_call_esscher_fourier",
    "vg_put_esscher_fourier",
    "vg_call_density_quad",
    "vg_call_mc_esscher",
    "vg_call_mc_direct",
    "MCResult",
    "call_price_from_cf",
    "bs_call",
    "bs_put",
    "bs_vega",
    "implied_vol",
]

__version__ = "1.0.0"

