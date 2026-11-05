# Library choice (task 1.1, 2026-10-31)

| Candidate | License file checked | JAX / GPU | Determinism | Size evidence | Maturity evidence |
|---|---|---|---|---|---|
| herculens 0.3.0 | MIT, installed notice retained | JAX analytical profiles; upstream CPU/GPU support | Local seeded arrays tested with CPU/float64; other devices untested | Installed package measured by audit script; JAX native runtime is a separate large dependency | Maintained specialist code; upstream publications and examples; local profile tests pass |
| lenstronomy 1.13.2 | BSD-3-Clause, installed notice retained | Chosen profile implementation is NumPy/CPU; no GPU claim | Analytical profiles without random draws; local injection tests pass; cross-platform untested | Installed package measured separately from its broad dependency closure | Established modelling package with tests/publications; local TNFW/PJAFFE tests pass |
| PyAutoLens 2026.10.16.1 | MIT, upstream LICENSE snapshot retained | Current upstream README advertises JAX/GPU | Not tested here; seeded inference/device agreement requires measurement | PyPI archive size recorded; full installation not measured | Established package with workspace/tutorials/publications; not installed or locally tested |

**Choice:** herculens is the primary smooth-model backend: it preserves the planned
Python/JAX direction and supplies SPLE (`EPL`), external shear and Sérsic profiles.
The installed profile mapping has no TNFW/PJAFFE entries, so lenstronomy supplies
only those two spherical injection profiles rather than reimplementing them.
The simulator converts deflections to NumPy and uses SciPy convolution and NumPy
noise: its complete rendering path is CPU-only and not differentiable. Task 1.3 adds a separate JAX forward path for the smooth fitter, checked against
the simulator; optional subhaloes remain outside that differentiable path.
PyAutoLens remains a comparison candidate; installing another inference framework
is unnecessary for these simulator tasks. This is a scope/compatibility choice,
not a measured performance ranking.

Installed MiB per distribution (including metadata/cache files) are actual local
measurements in [LICENSES.md](LICENSES.md), produced by
`scripts/audit_environment.py`. The PyPI versioned metadata measurements in
[library-archives.json](library-archives.json) record compressed archive bytes and
hashes for all three candidates. Archive size is not full environment size;
PyAutoLens' dependency footprint remains unknown. No timing comparison is claimed.

Both installed closures require a release-specific audit: pyxdg includes LGPL-2.0,
Astropy bundles LGPL WCSLIB and native wheels carry additional notices. This local
development lock must not become the BOINC payload unchanged. No proprietary
source licensing change or binary release is made.

Sources checked: [herculens README](https://github.com/Herculens/herculens),
[herculens profile mapping](https://github.com/Herculens/herculens/blob/main/herculens/MassModel/profile_mapping.py),
[lenstronomy repository](https://github.com/lenstronomy/lenstronomy),
[PyAutoLens README](https://github.com/PyAutoLabs/PyAutoLens),
[PyAutoLens license](https://github.com/PyAutoLabs/PyAutoLens/blob/main/LICENSE).
Upstream main pages can change; installed notices and pinned profile code are the
local evidence. No cross-device determinism or science validation is established.
