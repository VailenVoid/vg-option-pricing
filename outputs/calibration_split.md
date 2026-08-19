# Calibration scored out of sample

617 European calls on `^XSP`, 12
expiries.  Snapshot 2026-08-18T08:43:40+00:00.

## The point of this table

A calibrated model reprices its own training quotes nearly by construction, so
an in-sample calibration error measures how *flexible* a model is, not how
*accurate*.  Every model below is therefore fitted on one part of the surface
and scored on a part it never saw -- and Black-Scholes is **fitted** to option
prices on identical terms rather than estimated from historical returns, so the
comparison is between models rather than between information sets.

That distinction is not pedantry.  Fit a model by minimising squared error
against option prices, report squared error against those same prices, and
compare it with a model estimated from returns, and the fitted model wins
whatever it is -- the result is a property of the setup, not of the model.

## Results

| split                  | model         |   free params | fitted                                        |   n train |   n test |   train RMSE ($) |   test RMSE ($) |   test MSE |   test MAE ($) |   train RMSE (vol pts) |   test RMSE (vol pts) |
|:-----------------------|:--------------|--------------:|:----------------------------------------------|----------:|---------:|-----------------:|----------------:|-----------:|---------------:|-----------------------:|----------------------:|
| interleaved strikes    | black_scholes |             1 | sigma=0.1364                                  |       311 |      306 |            5.962 |           5.957 |      35.49 |          3.495 |                  4.312 |                 4.159 |
| interleaved strikes    | esscher       |             2 | VG(theta=+0.00000, sigma=0.13643, nu=0.00100) |       311 |      306 |            5.958 |           5.953 |      35.44 |          3.491 |                  4.294 |                 4.143 |
| ATM -> wings           | black_scholes |             1 | sigma=0.1428                                  |       390 |      227 |            5.959 |           4.292 |      18.42 |          2.428 |                  3.175 |                 5.188 |
| ATM -> wings           | esscher       |             2 | VG(theta=+0.00000, sigma=0.16278, nu=0.17684) |       390 |      227 |            4.352 |           3.795 |      14.4  |          2.388 |                  2.576 |                 4.771 |
| short -> long maturity | black_scholes |             1 | sigma=0.1203                                  |       348 |      269 |            1.228 |          11.22  |     125.9  |          8.835 |                  3.79  |                 6.366 |
| short -> long maturity | esscher       |             2 | VG(theta=+0.00000, sigma=0.12024, nu=0.00100) |       348 |      269 |            1.23  |          11.23  |     126.1  |          8.841 |                  3.75  |                 6.369 |

## Reading it

The gap between the `train` and `test` columns is the honest cost of
calibration:

| split                  | model         |   train RMSE (vol pts) |   test RMSE (vol pts) |   ratio |
|:-----------------------|:--------------|-----------------------:|----------------------:|--------:|
| interleaved strikes    | black_scholes |                  4.312 |                 4.159 |  0.9644 |
| interleaved strikes    | esscher       |                  4.294 |                 4.143 |  0.9648 |
| ATM -> wings           | black_scholes |                  3.175 |                 5.188 |  1.634  |
| ATM -> wings           | esscher       |                  2.576 |                 4.771 |  1.852  |
| short -> long maturity | black_scholes |                  3.79  |                 6.366 |  1.68   |
| short -> long maturity | esscher       |                  3.75  |                 6.369 |  1.698  |

* **Interleaved strikes** is the easy split -- neighbouring strikes carry almost
  the same information, so any model that fits in-sample also fits out. A model
  failing here would be broken.
* **ATM to wings** asks the model to extrapolate the smile's *shape* outward
  from the money.  This is where a specification that cannot bend pays for it.
* **Short to long maturity** asks it to extrapolate the *term structure*, which
  a Levy model cannot do freely: its excess kurtosis must decay like `1/T`.

The three splits together say more than any single number, and none of them is
the number a calibration paper usually quotes.

## The full picture

Three comparisons, all now on the same footing:

| what is compared | question answered | where |
|---|---|---|
| both estimated from returns | which **predicts** better? | `validation_out_of_sample.md` |
| both calibrated, scored in sample | which is more **flexible**? | `calibration.md` |
| both calibrated, scored out of sample | which **generalises**? | this file |

Only the first and third are tests of a model.  The second is a test of a
parametrisation, and it is the one most often reported as if it were the first.
