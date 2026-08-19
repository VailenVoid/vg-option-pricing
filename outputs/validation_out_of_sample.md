# Out-of-sample validation

Parameters are estimated from the **underlying's realised returns** and then
used to **predict** option prices.  No option price is fitted, so every number
below is out of sample.  This is a much harder test than a calibration, and it
is the one the paper's construction actually implies: estimate under P, move to
Q with the Esscher transform, price.

* Validation set: **617 European calls** on `^XSP`
  (European (cash-settled index option)), 12 expiries,
  T from 0.016 to 1.332 years.
* Estimation data: daily log returns on `^GSPC`.
* Monte Carlo: 200,000 paths per expiry, importance sampling under P
  with the Esscher weight (eq. 8).

## 1. Estimated parameters, by estimation window

|   window (y) | model         |   n |   RMSE ($) |   MAE ($) |   MAPE (%) |   RMSE (vol pts) |
|-------------:|:--------------|----:|-----------:|----------:|-----------:|-----------------:|
|            1 | black_scholes | 617 |      6.715 |     3.899 |      28.71 |            4.615 |
|            1 | vg_fourier    | 617 |      6.884 |     3.997 |      27.53 |            4.687 |
|            1 | vg_mc_q       | 617 |      6.785 |     3.968 |      27.39 |            4.673 |
|            1 | vg_mc_is      | 617 |     21.53  |     9.483 |      37.16 |            4.248 |
|            2 | black_scholes | 617 |      4.195 |     3.032 |      75.18 |            4.023 |
|            2 | vg_fourier    | 617 |      4.176 |     2.915 |      75.79 |            3.792 |
|            2 | vg_mc_q       | 617 |      4.153 |     2.898 |      75.4  |            3.76  |
|            2 | vg_mc_is      | 617 |      4.201 |     2.932 |      75.85 |            3.858 |
|            5 | black_scholes | 617 |      4.007 |     3.146 |      89.41 |            4.28  |
|            5 | vg_fourier    | 617 |      3.969 |     3.082 |      89.43 |            4.148 |
|            5 | vg_mc_q       | 617 |      3.878 |     3.036 |      89.22 |            4.119 |
|            5 | vg_mc_is      | 617 |      3.974 |     3.082 |      89.54 |            4.178 |

## 2. Full scoring

|   window (y) | model         |   n |     MSE |   RMSE ($) |   MAE ($) |   MAPE (%) |   mean error ($) |   RMSE (vol pts) |   mean IV error (vol pts) |
|-------------:|:--------------|----:|--------:|-----------:|----------:|-----------:|-----------------:|-----------------:|--------------------------:|
|            1 | black_scholes | 617 |  45.094 |     6.7152 |    3.8989 |     28.708 |         -3.4662  |           4.6148 |                  -2.4439  |
|            1 | vg_fourier    | 617 |  47.384 |     6.8836 |    3.9974 |     27.528 |         -3.6289  |           4.6874 |                  -2.6061  |
|            1 | vg_mc_q       | 617 |  46.038 |     6.7851 |    3.9681 |     27.388 |         -3.6046  |           4.6726 |                  -2.6016  |
|            1 | vg_mc_is      | 617 | 463.54  |    21.53   |    9.4825 |     37.159 |         -9.0587  |           4.2479 |                  -1.6985  |
|            2 | black_scholes | 617 |  17.595 |     4.1946 |    3.0318 |     75.185 |         -0.25465 |           4.023  |                   0.92742 |
|            2 | vg_fourier    | 617 |  17.436 |     4.1756 |    2.9155 |     75.791 |         -0.4447  |           3.7922 |                   0.76638 |
|            2 | vg_mc_q       | 617 |  17.244 |     4.1525 |    2.8976 |     75.404 |         -0.44991 |           3.7604 |                   0.74858 |
|            2 | vg_mc_is      | 617 |  17.648 |     4.201  |    2.9323 |     75.855 |         -0.4832  |           3.8584 |                   0.71492 |
|            5 | black_scholes | 617 |  16.054 |     4.0067 |    3.1456 |     89.409 |          0.55963 |           4.28   |                   1.7304  |
|            5 | vg_fourier    | 617 |  15.752 |     3.9689 |    3.0819 |     89.43  |          0.49155 |           4.1479 |                   1.6675  |
|            5 | vg_mc_q       | 617 |  15.039 |     3.878  |    3.0359 |     89.22  |          0.54143 |           4.1191 |                   1.6812  |
|            5 | vg_mc_is      | 617 |  15.792 |     3.974  |    3.0818 |     89.54  |          0.48066 |           4.178  |                   1.6556  |

## 3. The pricers agree with each other -- but equation (8) can break

Fourier inversion and Monte Carlo share no code beyond the parameter object, and
across all 617 quotes they differ by at most **$0.7881**, a worst
z-score of **2.42** standard errors.  Whatever the models get wrong about
the market, they are not getting it wrong through an implementation error.

That holds for `vg_mc_q`, which simulates directly under Q using the tilted
parameters.  The estimator of **equation (8)** -- simulate under P, reweight by
`exp(h* X_T)/M(h*, T)` -- is a different matter, and the table above shows it
failing outright on the shortest estimation window.

The mechanism is worth stating because the paper proposes eq. (8) as the
practical route.  A short estimation window produces a small `nu`, a small `nu`
pushes `h*` far out (h* = 0.5 at the best window, but above 29 on the
one-year window), and the weight `exp(h* X_T)` is then lognormal with an
enormous variance: a handful of paths carry essentially all of the mass and the
average is meaningless at any feasible sample size.  Since the Esscher transform
maps VG to VG, the fix costs nothing -- simulate under Q with the tilted
parameters and drop the weight entirely, which is what `vg_mc_q` does.  Same
measure, same price, usable variance.

## 4. What the errors say

Best window: **5 years** (1260 returns).

* Physical parameters: `VG(theta=+0.00811, sigma=0.16989, nu=0.00836)`, drift +0.1044
* Esscher parameter solving the quadratic: `h* = 0.5189`
* Risk-neutral parameters after the transform: `VG(theta=+0.02309, sigma=0.16989, nu=0.00836)`
* Black-Scholes volatility from the same returns: **16.99%**
* Average at-the-money implied volatility in the market: **14.18%**

The gap between those last two numbers is the whole story.  Realised volatility
over the estimation window and the volatility the market is charging are simply
not the same number, and neither model can price the options correctly while
using the first to predict the second.  That is not a defect of the estimator --
it is the variance risk premium, and it is exactly why practitioners calibrate
to option prices instead of estimating from returns.  `calibration.md` does that
and gets errors an order of magnitude smaller.

The second panel of `fig_09_out_of_sample.png` shows this directly: the flat
line is the historical volatility, the scatter is the market's implied
volatility, and they do not meet.

### By maturity

| expiry     |       T |   n |   VG RMSE ($) |   BS RMSE ($) |
|:-----------|--------:|----:|--------------:|--------------:|
| 2026-08-24 | 0.01644 |  44 |         1.6   |         1.798 |
| 2026-08-27 | 0.02466 |  55 |         1.605 |         1.751 |
| 2026-08-31 | 0.03562 |  78 |         1.939 |         2.065 |
| 2026-09-08 | 0.05753 |  28 |         2.761 |         2.885 |
| 2026-09-18 | 0.08493 | 109 |         2.515 |         2.6   |
| 2026-10-02 | 0.1233  |  34 |         3.216 |         3.289 |
| 2026-10-16 | 0.1616  |  75 |         3.217 |         3.294 |
| 2026-11-20 | 0.2575  |  32 |         3.602 |         3.623 |
| 2027-01-15 | 0.411   |  40 |         4.492 |         4.489 |
| 2027-03-19 | 0.5836  |  34 |         5.832 |         5.817 |
| 2027-07-16 | 0.9096  |  49 |         6.519 |         6.492 |
| 2027-12-17 | 1.332   |  39 |         8.289 |         8.272 |
