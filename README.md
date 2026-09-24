# Variance-Gamma model: pricing European options

Implementation of the seminar paper *"Variance-Gamma model i njegova primena"*
("The Variance-Gamma model and its applications", Faculty of Mathematics,
University of Belgrade, 2026): the price of a European call under the VG model
via the **Esscher transform**, with two independent routes to the same price —
**Fourier inversion** of the characteristic function and **Monte Carlo**
simulation — evaluated on real market data.

The code in this repository is my own: written from scratch and independent of
any code accompanying the paper. Reports and figures in `outputs/` are in
English as well, so they can be included in LaTeX directly.

---

## What it does

| # | Task | Where |
|---|---|---|
| 1 | Estimate VG and Black-Scholes parameters from historical returns (method of moments and MLE) | [`vg/estimation.py`](src/vg/estimation.py), `scripts/03` |
| 2 | Price an option from those parameters via the **inverse Fourier transform** of the Esscher-transformed process: `h*` from a quadratic equation, then numerical integration | [`vg/pricing.py`](src/vg/pricing.py), [`vg/esscher.py`](src/vg/esscher.py) |
| 3 | The same price via **Monte Carlo** estimation — in two variants | [`vg/pricing.py`](src/vg/pricing.py) |
| 4 | A **validation set** of real European options; for each input, estimate parameters, call both pricers, compare against the traded price, report MSE/RMSE/MAE/MAPE | `scripts/05`, [`outputs/validation_out_of_sample.md`](outputs/validation_out_of_sample.md) |
| 5 | Risk-neutral calibration to option prices, evaluated **out of sample** as well | `scripts/04`, `scripts/06` |

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe scripts\01_fetch_data.py          # download and cache market data
.\.venv\Scripts\python.exe scripts\02_validate_numerics.py   # numerical validation, no data needed
.\.venv\Scripts\python.exe scripts\03_physical_estimation.py # physical parameters from returns
.\.venv\Scripts\python.exe scripts\04_calibrate.py           # calibration to option prices
.\.venv\Scripts\python.exe scripts\05_validation.py          # prediction from returns + metrics
.\.venv\Scripts\python.exe scripts\06_calibration_split.py   # calibration scored out of sample

.\.venv\Scripts\python.exe -m pytest -q                      # 230 tests, ~2 min
```

Results (markdown reports, CSV tables, PNG + PDF figures) go to `outputs/`.
A data snapshot is already in `data/`, so scripts 02–06 run without step 01.

**Runtime.** Scripts 01–03 and 05–06 take a few minutes each. Script 04 takes
about 25 minutes, because it fits both globally and separately for each of the
12 maturities, with multi-start.

**Memory.** The quadrature holds temporary arrays of shape `(number of strikes ×
number of nodes)`; the block is tuned to ~400k elements, and `_segment_integral`
halves the block and retries if an allocation fails, so a long calibration
survives on a loaded machine.

Minimal example:

```python
import numpy as np
from vg import VGParams, vg_call_esscher_fourier, vg_call_mc_direct

p = VGParams(theta=-0.14, sigma=0.12, nu=0.20)
K = np.array([90.0, 100.0, 110.0])

vg_call_esscher_fourier(S0=100.0, K=K, r=0.03, T=0.5, p=p)   # Theorem 4.1
# array([11.56999845,  3.8172685 ,  0.77084887])

mc = vg_call_mc_direct(100.0, K, 0.03, 0.5, p, 10**6, np.random.default_rng(0))
mc.price, mc.stderr   # same price, with a standard error alongside it
```

---

## Layout

| Module | Contents | Section of the paper |
|---|---|---|
| [`vg.process`](src/vg/process.py) | VG process: MGF, characteristic function, cumulants, exact simulation, closed-form density (Bessel `K`) | 3.1–3.3, Lemma 3.1 |
| [`vg.esscher`](src/vg/esscher.py) | Esscher transform, `h*` from a quadratic equation, explicit Q-parameters | 4.2, Lemma 4.1, Proposition 4.1 |
| [`vg.pricing`](src/vg/pricing.py) | Gil-Pelaez Fourier inversion, Monte Carlo, quadrature against the density, Black-Scholes and implied volatility | 4.3–4.5, eq. (5)–(8) |
| [`vg.estimation`](src/vg/estimation.py) | Physical parameters from returns: method of moments and MLE | — |
| [`vg.calibration`](src/vg/calibration.py) | Risk-neutral calibration to an option chain | — |
| [`vg.data`](src/vg/data.py) | Market data, cleaning, forwards and discount factors from put-call parity | — |
| [`vg.plotting`](src/vg/plotting.py) | Figure style | — |

**Three independent routes to the same price** — Fourier inversion, quadrature
against the closed-form density, and Monte Carlo — share nothing but the
parameter object. That is the basic correctness check, and all three agree to
`5.4e−07` (finding B).

---

## Data: `^XSP`, not `SPY`

The paper prices a **European** option (Theorem 4.1), whereas `SPY` options are
**American** — model and data would describe different contracts, and the
early-exercise premium would have to be argued away. Hence the `^XSP` chain
(mini-S&P 500 index): European, cash-settled index options.

Snapshot of `2026-08-18`: **2,987** raw contracts → **617** calls across
**12 maturities** after filtering, **100%** of prices are two-sided quote
midpoints. The return history is taken from `^GSPC` (same process, far longer
sample; daily log returns correlate at `0.99933`, annualised volatilities agree
to 0.03 percentage points).

Instead of assuming a rate and a dividend yield, the forward `F` and the
discount factor `D` are extracted per maturity from put-call parity,
`C(K) − P(K) = D(F − K)`. All formulas are generalised to a cost of carry
`r − q`; for `q = 0` they reduce exactly to the formulas in the paper.

The source is `yfinance` (free, no API key). Every download is written to
`data/` with a timestamp, and everything downstream reads the cached snapshot,
so results stay reproducible even after the market moves.

---

## Additions to the paper

### 1. An Esscher-transformed VG process is again a VG process

From `D_{h+iu} = D_h − iuν(θ + σ²h) + ½σ²νu²` it follows that under `Q^h` the
process `X` remains VG, with

```
θ' = (θ + σ²h) / D_h,     σ' = σ / √D_h,     ν' = ν.
```

This turns an abstract change of measure into an explicit reparametrisation.
The payoff is twofold: the option can be simulated **directly under Q** instead
of reweighting P-paths, and it yields a fully independent check on the Fourier
code. How much that is worth — finding C.

### 2. A correction to equation (4)

The paper writes `e^{rt} = M(h*+1, t)/M(h*, t)` and then `H(h*) = e^{rt}`. Since
`M(h,t) = D_h^{−t/ν}`, the condition is in fact

```
e^{rt} = (D_{h*}/D_{h*+1})^{t/ν}   ⟺   H(h*) = e^{rν}.
```

The `e^{rν}` form is the correct one, and it matters: `h*` **must not** depend on
`t`, because a single measure has to price all maturities, and only the `e^{rν}`
version has that property. (The paper itself uses `e^{rν}` two paragraphs
later.) `H(h*) = e^{rν}` is what is implemented; a test checks this across
several `t` with the same `h*`.

---

## Findings

All with references to the reports in `outputs/`, where the numbers and the
evidence live.

### A. The Esscher specification has two free parameters, not three

The most important finding, and it bears directly on Chapter 4 of the paper.

Under the model `S_t = S₀ e^{X_t}` (Section 4.1) the price has **no drift of its
own**, so the entire martingale burden falls on the law of `X` under `Q`. The
condition `E^Q[e^{X_t}] = e^{rt}` gives

```
θ_Q + σ_Q²/2 = (1 − e^{−rν}) / ν.
```

One combination of the risk-neutral parameters is therefore **pinned**, and
option prices depend on the physical `(θ, σ, ν)` only through `(σ_Q, ν)`. Two
consequences, both verified numerically in
[`outputs/validation.md`](outputs/validation.md):

1. **The physical `θ` is not identifiable from option prices.** Five physical
   parameter sets with `θ` ranging from −0.60 to +0.20, constructed to share the
   same `(σ_Q, ν)`, produce **identical prices to machine precision**. This is
   why `vg.calibration` fixes `θ = 0` and fits `(σ, ν)` — nothing is lost, and
   optimising along a flat direction is avoided.

2. **The attainable smiles are nearly symmetric.** Because `θ_Q` is pinned close
   to `−σ_Q²/2` (exactly that when `r = 0`), the model produces curvature but
   almost no slope — under a fifth of a volatility point across 80%–125%
   moneyness, whatever the physical `θ`.

`fig_04_smile_shapes.png` puts the two halves side by side: four processes whose
densities under `P` differ visibly, and the four implied-volatility curves they
produce — which lie on top of one another. Everything that distinguishes them
under `P` is erased by the martingale condition on the way to `Q`.

This is a property of the *specification*, not of the VG process and not of the
numerics.

**Confirmation on market data.** Calibration to 617 `^XSP` calls
([`outputs/calibration.md`](outputs/calibration.md)):

| model | free params | σ | ν | RMSE price | RMSE IV | at bound |
|---|---|---|---|---|---|---|
| Black-Scholes, single flat vol | 1 | 0.13590 | — | $6.003 | 425.6 bp | |
| VG / Esscher | 2 | 0.13594 | 0.00100 | $6.000 | 423.9 bp | `ν` |

The market smile in this snapshot falls by **5.6 volatility points** on average
(range 3.5–8.2) between one standard move below and above the forward — a clean,
realistic equity skew.

The Esscher fit does not merely land *near* Black-Scholes — it lands **on** it:
`σ = 0.13594` against a flat BS volatility of `0.13590`, with `ν` at its lower
bound. The reason is exactly the finding above: a symmetric smile cannot
approach a monotonically decreasing one, because raising `ν` lifts *both* wings
— it helps on one side and hurts on the other. The best compromise is not to
bend at all. The second parameter buys 1.7 basis points (425.6 → 423.9). The
same holds maturity by maturity: even when allowed to fit each maturity
separately, `ν` ends up at its lower bound in **9 out of 12** maturities, and the
rank correlation of `ν` with `T` is `+0.03` — there is no term structure,
because the model refuses the curvature `ν` offers. `σ` carries the whole load,
rising from 0.105 at the front end to 0.186 at the back (rank correlation
`+0.96`): this is how a flat-volatility model encodes a smile it cannot
represent.

`fig_07_residuals.png` shows the shape of the failure: the residuals are not
noise around zero but a systematic tilt in moneyness — the missing skew forcing
its way through.

### B. Fourier inversion does not converge uniformly — and it says so

For `T/ν ≤ 1/2` the VG density is unbounded at zero, the characteristic function
decays only as `u^{−2T/ν}`, and no Fourier method converges quickly. That is a
property of the law, not of the implementation.

The trap is to fix a truncation point first and then ration nodes to reach it:
under-resolving the oscillation does not lose a digit, it returns a number
without a single correct digit. Instead the integral is accumulated over dyadic
segments `[0,U], [U,2U], [2U,4U], …`, each fully resolved, and the tail is
estimated two independent ways (geometric decay of the contributions, and the
bound `|φ(U)|/β`), of which the **smaller** is reported. The routine returns a
`QuadInfo` carrying `converged` and `error_estimate`.

Measured ([`outputs/validation.md`](outputs/validation.md)):

| regime | `T/ν` | Fourier vs density | reported error |
|---|---|---|---|
| near-Gaussian | 37.5 | `~3e−14` | `1e−13` |
| moderate tails | 2.5 | `~2e−11` | `3.1e−09` |
| heavy tails | 0.32 | `~3e−07` | `3.6e−05`, `converged=False` |

Largest Fourier-vs-density disagreement across all regimes: **`5.4e−07`** on a
$100 asset. The test `test_reported_error_is_an_upper_bound_on_the_real_error`
checks that the reported error really is an upper bound — the entire numerical
part rests on that.

### C. Equation (8) breaks down, and the fix is free

The paper proposes Monte Carlo via weighted sampling (eq. 8): simulate under `P`,
weight by `exp(h* X_T)/M(h*, T)`. Since the Esscher transform maps VG to VG, the
alternative is to simulate directly under `Q` with the transformed parameters and
drop the weight. Both are unbiased for the same price, but:

* on synthetic tests the weighted estimator has on average a **5.1×** larger
  standard error, i.e. it needs ~**27×** more paths for the same accuracy
  (`fig_03_mc_convergence.png`);
* on real data it **fails outright** when `ν` is small. A short estimation window
  gives a small `ν`, a small `ν` pushes `h*` far out (`h* = 0.52` on a five-year
  window, **over 29** on a one-year window), and the weight `exp(h* X_T)` becomes
  lognormal with enormous variance: a handful of paths carry all the mass. RMSE
  jumps to **$21.53** against **$6.79** for direct simulation under `Q`.

Same measure, same price, usable variance.

### D. Prediction from returns: the variance risk premium

This is the test the construction in the paper actually implies — estimate under
`P`, move to `Q` by the Esscher transform, price. **No option price is fitted**,
so every number is out of sample. Validation set: 617 European options, 12
maturities
([`outputs/validation_out_of_sample.md`](outputs/validation_out_of_sample.md)).

Best estimation window — 5 years (1,260 returns):

| method | MSE | RMSE ($) | MAE ($) | RMSE (vol points) |
|---|---|---|---|---|
| Black-Scholes (MLE from returns) | 16.05 | 4.007 | 3.146 | 4.280 |
| VG / Esscher, Fourier | 15.75 | 3.969 | 3.082 | 4.148 |
| VG / Esscher, Monte Carlo under `Q` | **15.04** | **3.878** | **3.036** | **4.119** |
| VG / Esscher, Monte Carlo eq. (8) | 15.79 | 3.974 | 3.082 | 4.178 |

VG beats Black-Scholes, but modestly, and both models err in the same direction
for the same reason:

* volatility from realised returns: **16.99%**
* average at-the-money implied volatility: **14.18%**

That gap is the whole story. Realised volatility and the volatility the market
charges are simply not the same number — this is the **variance risk premium**,
and it is precisely why one calibrates to option prices in practice instead of
estimating from returns. It is not a flaw in the estimator; it is being measured
here.

### E. Calibration scored out of sample

A calibrated model reproduces its own quotes almost by construction, so the
**in-sample** error measures how *flexible* the model is, not how *accurate*.
Every model here is therefore fitted on one part of the surface and scored on a
part it has not seen — and Black-Scholes is **fitted** to option prices under the
same conditions, not estimated from history, so that models are compared rather
than information sets
([`outputs/calibration_split.md`](outputs/calibration_split.md)).

RMSE in volatility points, `train → test`:

| split | what it tests | Black-Scholes | VG / Esscher |
|---|---|---|---|
| alternating strikes | nothing (control) | 4.31 → 4.16 | 4.29 → **4.14** |
| ATM → wings | extrapolating the *shape* of the smile | 3.18 → 5.19 | 2.58 → **4.77** |
| short → long maturities | extrapolating the *term structure* | 3.79 → 6.366 | 3.75 → 6.369 |

The second parameter buys something only when extrapolating the shape of the
smile (4.77 against 5.19). On the term structure it buys nothing — a difference
of 0.003 volatility points is noise, and `ν` again ends at its bound. For Lévy
processes the excess kurtosis of log returns at horizon `T` is `3ν/T` and must
decay as `1/T`, whereas the market's decays more slowly; but for that to be
visible at all, `ν` has to be free, and here it is not.

There is a methodological point here too. If a model is fitted by minimising
squared error against option prices, and that same error against those same
prices is then reported as a measure of quality and compared against a model
estimated from returns — the fitted model wins no matter what. The result is a
property of the setup, not of the model. The three comparisons that actually mean
something are kept separate:

| what is compared | which question it answers | where |
|---|---|---|
| both estimated from returns | which **predicts** better? | `validation_out_of_sample.md` |
| both calibrated, scored in sample | which is more **flexible**? | `calibration.md` |
| both calibrated, scored out of sample | which **generalises**? | `calibration_split.md` |

Only the first and the third are tests of a model.

### F. Method of moments and MLE disagree on daily data

On 2,510 daily `^GSPC` returns: moments give `ν = 0.0217`, MLE gives
`ν = 0.0048` — a factor of 4.6
([`outputs/physical_estimation_daily.md`](outputs/physical_estimation_daily.md)).

The cause is structural. The shape of the Gamma clock per step is `dt/ν`, here
0.18; below `1/2` the density has a power singularity at zero, so the likelihood
measures how sharply the parameters spike at the mode, not how well they describe
the spread of the data.

The model's claim about that spike is testable, and it **fails**: under the
moment fit, 19.5% of daily returns should fall within `1e−4` of the drift
("nothing happened today"); in the actual `^GSPC` series **1.3%** do. Markets
trade continuously and do not pile returns up at zero the way a Gamma clock of
this shape claims. This is a genuine limitation of VG as a description of the
physical measure, separate from how it prices options under `Q`.

The moment estimates are the ones carried forward, because option prices depend
on the variance, skewness and kurtosis of the return distribution, not on the
height of the mode. The disagreement does not simply vanish under coarser
sampling: the ratio of the two `ν` estimates is 4.6× daily, 1.6× weekly, 3.0×
monthly — the weekly case improves as the mechanism predicts, but the monthly
series has only 119 observations, with a sample kurtosis determined by a handful
of crisis months, so it is far too short to settle anything.

### G. Put-call parity on stale quotes

Outside US trading hours Yahoo returns `bid = ask = 0`, so the only available
prices are last trades — and for calls and puts those originate at different
moments. The two-parameter regression `C − P = D(F − K)` then reads timing noise
as curvature: on an earlier snapshot it produced discount factors **above 1**
(negative rates) and dividend yields of **−4%** on half the maturities. A robust
variant (`D` fixed from `^IRX`, `F` as the median of `K + (C−P)/D` near the
money) removes this. Both are implemented; the choice is automatic based on the
share of two-sided quotes, and the method is recorded in the `fwd_method` column
so that nothing enters the calibration on a silent assumption.

A side consequence: fitting `log F(T) = log S_eff + carry·T` across all
maturities gives an effective spot of **770.33** against a quoted last price of
**778.58** — a **1.06%** difference. That is not a modelling decision but a
timestamp: the last bar of the index history and the option quotes come from
different sessions. Initially that gap failed *every* short maturity in the
forward check, even though their forwards were excellent (two independent
estimators agreeing to two-thousandths of a dollar). Checking the curve against
itself rather than against a stale spot keeps them.

---

## Numerical validation

`scripts/02_validate_numerics.py` verifies, **with no market data at all**, that
the implementation is correct before it is pointed at the market:

- **three independent routes to a price** — largest disagreement `5.4e−07` on a
  $100 asset;
- **Monte Carlo** — largest z-score against the Fourier price `1.98` standard
  errors across 15 strike/regime combinations;
- **the martingale property** — `E^Q[e^{X_T}] = e^{rT}` to `1e−11` for every `t`;
- **the `ν → 0` limit** — monotone convergence VG → Black-Scholes;
- **identifiability** — the table from finding A.

The test suite (230 tests) additionally checks moments against Monte Carlo, the
density against quadrature (integral `= 1.0000000000` in all four regimes,
including the singular ones), cumulants against a Chebyshev expansion of `log M`,
put-call parity with an independently computed put, monotonicity and convexity in
the strike, and no-arbitrage bounds.

---

## References

- Madan, Carr, Chang, *The Variance Gamma Process and Option Pricing*, European Finance Review 2(1), 1998.
- Cont, Tankov, *Financial Modelling with Jump Processes*, Chapman & Hall/CRC, 2004.
- Gil-Pelaez, *Note on the inversion theorem*, Biometrika 38(3–4), 1951.
- Shenoy, Kempthorne, *The Variance Gamma Process for Option Pricing*, arXiv:2510.14093, 2024.

---

## License

MIT — see [LICENSE](LICENSE).
