# P1 provisional review finding 2: estimated noise and PSF mismatch

Run 200 paired simulated lenses on the same PCG64 lens/source draws and noise seeds
as the original P1 protocol. The smoother source family, geometry, exposure and
pixel size are identical across each pair. The true generator PSF and matched control
fit use Gaussian FWHM 0.16 arcsec. This is an unverified approximate VIS width used
to define a controlled synthetic perturbation, not a claim about the measured VIS PSF.

Both branches use gamma~Normal(2, 0.1), the same 2 starts, seed 731 and bounds.
The control receives the simulator's ideal expected noise map and exact PSF. The
comparison branch estimates a constant sky sigma from the fixed outer 15% border
using 1.4826*MAD, subtracts that border's median, and fits with Gaussian FWHM
0.176 arcsec (+10%). It masks NaNs and values more than 8 estimated sigmas below
the median sky, while preserving positive source pixels. Gain and exposure calibration
are unavailable; no Poisson source term is added.

No lens is replaced, tuned or filtered. Record every fit, estimated sigma, mask,
convergence, boundary and error. Report bias/scatter/RMSE of all parameters,
gross flags, chi2/dof and image S/N. This is a joint noise-estimation and PSF
perturbation; the two effects are not separated by this run.

The original raw fit outcomes are retained in `docs/review-fix2/paired.jsonl`.
Optimizer amplitude/radius parameters are packed as natural logarithms in that raw
file. `paired-physical.jsonl` adds corresponding physical source values for the
summary without refitting; original packed values are retained alongside them.
