# Calibration to the ^XSP option chain

Snapshot: **2026-08-18T08:43:40+00:00**, quoted spot **778.58**, 2,987
raw contracts downloaded, **617** calls surviving the screen across
12 expiries.

**Exercise style: European (cash-settled index option).**  This is the reason the chain is
`^XSP` and not the more liquid SPY: the paper's Theorem 4.1 prices a
European call, and SPY options are American.  Options on the index are
European and cash-settled, so model and data describe the same contract and no
early-exercise premium has to be argued away.

## 1. Building the quote set

| filter                                  |   kept |   dropped |
|:----------------------------------------|-------:|----------:|
| raw contracts                           |   2987 |         0 |
| valid strike/maturity                   |   2987 |         0 |
| finite positive price                   |   2748 |       239 |
| moneyness in [0.8, 1.2]                 |   2002 |       746 |
| relative spread <= 0.35                 |   1969 |        33 |
| traded or quoted (volume/OI/two-sided)  |   1960 |         9 |
| last-trade price fresher than 5.0d      |   1881 |        79 |
| option type == C                        |    833 |      1048 |
| inside no-arbitrage band                |    706 |       127 |
| implied vol in [2%, 200%]               |    706 |         0 |
| |log(K/F)| <= 3.0 ATM sigma*sqrt(T)     |    636 |        70 |
| IV within 4.0 MAD of the expiry's smile |    617 |        19 |

Prices are mids where a two-sided quote exists and last traded prices otherwise;
**100%** of the surviving quotes are
mids in this snapshot.

Rather than assume a rate and a dividend yield, each expiry's forward `F` and
discount factor `D` come from put-call parity, `C(K) - P(K) = D(F - K)`, fitted
across the liquid strikes.  That absorbs rate, dividend and borrow into the two
numbers the model actually needs:

| expiry     |        T |   forward |   discount |        r |          q | method                  |   pairs |   n |
|:-----------|---------:|----------:|-----------:|---------:|-----------:|:------------------------|--------:|----:|
| 2026-08-24 | 0.016438 |    770.99 |    0.9994  | 0.036303 | -0.015129  | parity (robust forward) |      44 |  44 |
| 2026-08-27 | 0.024658 |    771.14 |    0.99968 | 0.012777 | -0.029478  | parity (regression)     |      39 |  55 |
| 2026-08-31 | 0.035616 |    771.38 |    0.99751 | 0.069861 |  0.031657  | parity (regression)     |      56 |  78 |
| 2026-09-08 | 0.057534 |    771.87 |    0.99826 | 0.030213 | -0.0045157 | parity (regression)     |      15 |  28 |
| 2026-09-18 | 0.084932 |    772.59 |    0.99692 | 0.036303 |  0.0018529 | parity (robust forward) |      61 | 109 |
| 2026-10-02 | 0.12329  |    773.79 |    0.99485 | 0.041884 |  0.0056175 | parity (regression)     |      17 |  34 |
| 2026-10-16 | 0.16164  |    774.66 |    0.99132 | 0.053913 |  0.01929   | parity (regression)     |      57 |  75 |
| 2026-11-20 | 0.25753  |    777.15 |    0.98564 | 0.056155 |  0.021942  | parity (regression)     |      14 |  32 |
| 2027-01-15 | 0.41096  |    782.18 |    0.98115 | 0.046294 |  0.0091676 | parity (regression)     |       8 |  40 |
| 2027-03-19 | 0.58356  |    786.83 |    0.97304 | 0.046841 |  0.010524  | parity (regression)     |       8 |  34 |
| 2027-07-16 | 0.90959  |    796.77 |    0.95798 | 0.047201 |  0.010111  | parity (regression)     |      24 |  49 |
| 2027-12-17 | 1.3315   |    807.98 |    0.96625 | 0.025785 | -0.010051  | parity (regression)     |       8 |  39 |

### The spot is stale, and the options say so

Fitting `log F(T) = log S_eff + carry * T` across expiries gives an effective
spot of **770.33**, which is **-1.059%** away from the quoted
last close of 778.58.  That gap is not a modelling choice, it is a
timestamp: the index history's last bar and the option quotes come from
different sessions.

It matters more than its size suggests.  Screening the parity forwards against
the stale spot rejected every short expiry -- their forwards were excellent (the
two independent estimators agreed to two thousandths of a dollar) but sat 1%
away from where the stale spot said they should be.  Checking the curve against
itself instead keeps them, and everything downstream prices off `S_eff`.

## 2. Fit

| model                        |   free params |   theta |   sigma |      nu |   RMSE price ($) |   RMSE IV (bp) |   max |IV err| (bp) | at bound   |
|:-----------------------------|--------------:|--------:|--------:|--------:|-----------------:|---------------:|--------------------:|:-----------|
| Black-Scholes (one flat vol) |             1 |     nan | 0.1359  | nan     |           6.0027 |         425.56 |              1291.8 |            |
| VG / Esscher                 |             2 |     nan | 0.13594 |   0.001 |           5.9999 |         423.94 |              1289.9 | nu         |

The `free params` column is not decoration.  Under the Esscher specification of
Section 4.1 the martingale condition forces
`theta_Q + sigma_Q^2/2 = (1 - e^{-r nu})/nu`, so only **two** parameters are
free and `theta` is not identifiable from prices at all -- see `validation.md`
section 3, where five physical parameter sets spanning `theta` from -0.60 to
+0.20 are shown to give identical prices to machine precision.

## 3. What this says about the model

The ^XSP smile on this snapshot slopes down by 5.6 volatility points on average (range 3.5 to 8.2) between one standard
move below and one above the forward.  That is a real, clean equity skew, and
reproducing it requires a free skew parameter.

That is exactly what the Esscher specification gives up.  With `theta_Q` pinned
near `-sigma_Q^2/2` its smile is close to symmetric, and a symmetric smile
cannot approximate a monotonically falling one -- raising `nu` to add curvature
lifts *both* wings, which helps on one side and hurts on the other.  The
optimiser's best compromise is therefore to add no curvature at all: `nu` runs
to its lower bound and the model collapses onto flat Black-Scholes.  That is
what the `at bound` column reports, and it is why the Esscher row does not beat
the one-parameter benchmark despite having two parameters.  The collapse is literal, not approximate: the fitted `sigma = 0.13594` against the flat Black-Scholes volatility of `0.13590`, agreeing to four decimal places, with `nu` resting on its lower bound of 0.001.  The second parameter buys nothing at all -- the volatility error moves from 425.6 bp to 423.9 bp.

`fig_07_residuals.png` shows the shape of the failure: the residuals are not
noise around zero but a systematic tilt across moneyness, which is the missing
skew showing through.  A specification whose smile can bend the other way would
remove it; this one cannot, because the martingale condition has already spent
that degree of freedom.

## 4. Term structure

VG is a Levy process: increments are stationary and independent, so one
parameter set must price every maturity.  Fitting each expiry separately tests
that directly -- if the model were right, the fitted parameters would be
constant in `T`.

* **VG / Esscher**: sigma runs 0.1046-0.2583, nu runs 0.0010-3.2223, across T = 0.016-1.332y; nu sits on its lower bound at 9 of 12 expiries.

They are not constant (`fig_08_term_structure.png`):

* VG / Esscher sigma: rises with T (rank corr +0.96); 0.1046 at the front, 0.1859 at the back, range 0.1046-0.2583
* VG / Esscher nu: no monotone trend in T (rank corr +0.03); 0.0024 at the front, 0.0010 at the back, range 0.0010-3.2223

`sigma` carries the whole term structure here, and `nu` carries none of it:
it rests on its lower bound at 9 of the 12 expiries.  That is the
identifiability constraint of section 2 reappearing one maturity at a time.
Freed to fit a single expiry the specification still cannot bend its smile, so
it still declines the curvature `nu` would buy, and what is left is a flat
Black-Scholes fit whose level has to be re-chosen at every maturity.  A rising
`sigma` is exactly how a flat-vol model encodes a smile it cannot represent.

The comparison this section is meant to make therefore cannot be made on this
fit.  For a Levy process the excess kurtosis of the log return at horizon `T` is
`3 nu / T`, so it must decay like `1/T` by construction, and a market whose
implied kurtosis decays more slowly forces the per-expiry `nu` upward with
maturity.  That is the standard limitation of pure Levy models -- but reading it
off requires a specification whose `nu` is free, and this one's is not.  What
this fit does establish is the prior obstacle: the per-expiry `sigma` spans
0.1046-0.2583 while the model is committed to a
single number, so no one parameter set prices every maturity regardless of what
the tails are doing.

## 5. Per-expiry parameters

### VG / Esscher

| expiry     |        T |   n |   theta |   sigma |        nu |   rmse_price |   rmse_iv_bp |
|:-----------|---------:|----:|--------:|--------:|----------:|-------------:|-------------:|
| 2026-08-24 | 0.016438 |  44 |       0 | 0.10457 | 0.0023513 |      0.3357  |       256.1  |
| 2026-08-27 | 0.024658 |  55 |       0 | 0.11494 | 0.001     |      0.5354  |       312.12 |
| 2026-08-31 | 0.035616 |  78 |       0 | 0.11485 | 0.001     |      0.65258 |       354    |
| 2026-09-08 | 0.057534 |  28 |       0 | 0.11821 | 0.001     |      0.98384 |       284.87 |
| 2026-09-18 | 0.084932 | 109 |       0 | 0.13879 | 0.055483  |      1.3987  |       355.89 |
| 2026-10-02 | 0.12329  |  34 |       0 | 0.12957 | 0.001     |      1.5415  |       193.44 |
| 2026-10-16 | 0.16164  |  75 |       0 | 0.143   | 0.001     |      1.7674  |       190.96 |
| 2026-11-20 | 0.25753  |  32 |       0 | 0.14311 | 0.001     |      2.8386  |       310.62 |
| 2027-01-15 | 0.41096  |  40 |       0 | 0.15018 | 0.001     |      4.7787  |       544.75 |
| 2027-03-19 | 0.58356  |  34 |       0 | 0.25825 | 3.2223    |      4.1797  |       189.86 |
| 2027-07-16 | 0.90959  |  49 |       0 | 0.18693 | 0.001     |      5.3626  |       246.23 |
| 2027-12-17 | 1.3315   |  39 |       0 | 0.18589 | 0.001     |      7.8965  |       295.81 |

