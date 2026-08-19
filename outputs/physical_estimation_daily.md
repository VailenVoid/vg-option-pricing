# Physical parameters from realised returns

`^GSPC`, 2,510 daily log returns,
dt = 0.003968.  Snapshot 2026-08-18T08:43:40+00:00.

The option chain is on `^XSP` (European (cash-settled index option)); the return
history is taken from `^GSPC`, which is the same underlying
process with a far longer sample -- over their common window the daily log
returns correlate at 0.99933 and the annualised volatilities agree to 0.03
percentage points.

## Estimates

| estimator   |   mu_annual |    theta |   sigma |        nu |   annual_vol |   annual_skewness |   daily_excess_kurtosis |
|:------------|------------:|---------:|--------:|----------:|-------------:|------------------:|------------------------:|
| moments     |     0.2473  | -0.11982 | 0.18055 | 0.021666  |      0.18141 |        -0.042797  |                 16.688  |
| MLE         |     0.22854 | -0.10102 | 0.16781 | 0.0047606 |      0.16795 |        -0.0085852 |                  3.6114 |

The method of moments reproduces the sample moments it targets, as it must:

| moment          |       sample |   fitted (moments) |
|:----------------|-------------:|-------------------:|
| var             |  0.000130588 |        0.000130588 |
| skewness        | -0.679376    |       -0.679376    |
| excess_kurtosis | 16.688       |       16.688       |

## Why the two estimators disagree

They differ by a factor of 4.6 in `nu` (0.02167 by
moments, 0.00476 by likelihood), and the reason is structural rather
than numerical.

The Gamma clock's shape over one step is `dt/nu`: 0.183 at the moment
estimate, 0.834 at the MLE.  Below `1/2` the VG density has a power
singularity at the origin, so the likelihood is dominated by how sharply a
parameter set spikes at the mode rather than by how well it matches the spread
of the data -- and the two estimators are then answering different questions.
The moment estimator targets the variance, skewness and kurtosis, which is what
matters for pricing; the likelihood targets the peak.

The model's own claim about the peak is testable, and it fails.  Simulating from
the moment fit, 19.5% of daily returns should land within
1e-4 of the drift -- essentially "nothing happened today".  In the actual
`^GSPC` series 1.3% do.  Markets trade continuously
and do not pile up at the origin the way a Gamma clock of this shape says they
should, which is a real limitation of VG as a description of the physical
measure, quite separate from how well it prices options under Q.

The moment estimates are the ones carried forward, because option prices depend
on the variance, skewness and kurtosis of the return law rather than on the
height of its mode.

Lowering the sampling frequency raises `dt/nu` and should weaken the
singularity; run this script at `--frequency weekly` and `--frequency monthly`
to see how far that goes on this sample.  Note that the monthly series is only
119 observations long and its sample kurtosis is driven by a
handful of crisis months, so it is too short to settle the question on its own.
What the exercise does settle is that the two estimators are answering different
questions here, and that the one aligned with pricing is the moment estimator.

## What the fit does capture

`fig_05_physical_fit_daily.png` shows the point of the whole exercise.
Against the Gaussian benchmark the VG law matches both the sharp peak and the
heavy tails of the return distribution -- sample excess kurtosis
16.69 against 0 for the normal, sample skewness
-0.68 against 0.  The QQ panel shows where it still
fails, which is in the extreme tails.

## Note on the link to option prices

These are physical parameters.  Turning them into option prices needs a
martingale measure, and under the specification of Section 4.1 the Esscher
transform makes the physical `theta` unidentifiable -- see `validation.md`
section 4.  So the numbers above are the right answer to "what does the index do",
and deliberately not the numbers used to price; `04_calibrate.py` fits the
risk-neutral parameters to option prices directly.
