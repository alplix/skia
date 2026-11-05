# P1 provisional review finding 1: slope-constrained scan baseline

The 200 original free-slope draws are unchanged. `tests/bench/constrained_recovery.py`
repeats the same PCG64 parameter draws, observation settings and per-index noise seeds,
then fits both methods to each exact image. The prior branch adds a fixed Gaussian
penalty `(gamma-2)^2 / 0.1^2` to the objective; the reported chi-square remains the
image-data term only. Fixed lens/source bounds, start count, seed and stopping rules
are shared. This changes the inference model; it may bias gamma toward 2, so it is
reported alongside the free model. This is a stable-prototype baseline suggestion,
not a scientific justification that the real-lens slope is isothermal.

Run `JAX_PLATFORMS=cpu .venv/bin/python tests/bench/constrained_recovery.py`.
It writes paired raw outcomes under `docs/review-fix1/` and refuses overwrite.
Theta_E, both shear components and shear magnitude have paired bias/scatter/RMSE;
convergence, bound contacts and predeclared gross errors are counted for both.
The result will determine whether the prior improves the quantities requested by
review. No parameter ranges or thresholds will be changed after seeing outcomes.
