# DATA.md — what is in data/ (P0 provenance, updated 2026-11-03)

Everything here is public Euclid Quick Data Release 1 (Q1, released 19 March 2025). data/ is git-ignored; `data/q1_cutouts/`
holds 185 FITS images (12,254,400 bytes in total, excluding filesystem metadata). Nothing in data/ may be redistributed outside this project until the Q1 terms of use have been read (NOT
checked yet: see "Not verified").

## data/q1_cutouts/ — 185 real VIS cutouts (24 first, 129 added 2026-11-01, 32 added 2026-11-03)

- 185 files `<id_str>_VIS.fits` + `manifest.csv` (all grade A; the script only selects candidates that have an Euclid reference model, so NO non-lens negatives are included). The first 153 were fetched before 2026-11-03; the remaining 32 were fetched on 2026-11-03 with the same documented script and query. Candidates: Q1 discovery-engine lens catalogue, grade A, that also
  have an Euclid reference lens model; sorted by the reference Einstein radius and sampled at a regular stride, so they
  span small to large lenses (theta_E ~0.3-1.5 arcsec, see manifest). Reproduce / fetch more: `docs/fetch_q1_cutouts.py`.
- Product: MER background-subtracted VIS mosaic (`EUC_MER_BGSUB-MOSAIC-VIS`, Q1_R1), ESA cutout service, radius 6 arcsec
  -> 121x121 or 122x122 pixels (the size varies by one pixel with the WCS rounding). One image extension only (PRIMARY).
- Pixel scale 0.1 arcsec/pixel (CD1_1 = -2.7778e-5 deg), north up. `MAGZERO = 24.6` (AB zero point: mag = 24.6 - 2.5 log10 flux).
  `BUNIT` is absent: treat flux as the MER mosaic's native units, consistent with MAGZERO. Typical sky noise in the sample
  cutout: MAD-sigma ~0.0031 per pixel, median sky ~0.0007 (background-subtracted, so ~0).
- NOT in the cutout (the archive product has them elsewhere or not at all): no noise/RMS map, no weight map, no bad-pixel
  flag, no PSF, no gain/exposure time. Consequences for the code:
  - Noise map must be ESTIMATED: sky sigma from the MAD of source-free pixels (outer ring), optionally plus a Poisson term
    for bright pixels (gain unknown: start with sky-only and report it as a limitation). The estimate is part of the model,
    document it, do not tune it per lens.
  - PSF is not supplied. Euclid VIS PSF FWHM is roughly 0.16 arcsec (from memory, not verified here): use a documented
    parametric PSF (Gaussian or Moffat) and say so; a per-position PSF is a known improvement left for later.
  - Masks: none. Mask only what you detect yourself (NaNs, extreme outliers), document the rule.

## data/ref/ — Euclid's own published tables (Zenodo 10.5281/zenodo.15025832, licence cc-by-4.0)

- `q1_discovery_engine_lens_catalog.csv` — 2584 candidates: columns subset, id_str, object_id, tile_index, right_ascension,
  declination, grade (A/B/C), expert_score, notes. Grades: discovery_engine A=250, B=247, C=1918; other subsets exist.
- `modeling_lens_mass.csv` (335 lenses) and `modeling_lens_sersic.csv` — the Euclid consortium's own PyAutoLens fits
  (power-law mass + shear, Sersic light). Use them ONLY as a cross-check of our fitter on real lenses
  (P1 task 1.5), not as input and not as truth: they are another model's estimates, with their own degeneracies.
  Columns kept in `manifest.csv`: ref_einstein_radius_arcsec, ref_ell_comp_0/1, ref_shear_g1/2, ref_lens_light_reff_arcsec,
  ref_lens_light_sersic_n. Coordinate/ellipticity conventions are PyAutoLens's (ell_comps = (e1, e2) elliptical components
  with q = (1-|e|)/(1+|e|)); the centre columns are in arcsec relative to an origin that is NOT documented in the files.
  Compare theta_E, shear magnitude and axis ratio first; compare centres/angles only after you have established the convention.
- The 3 GB `lens.zip` of the same Zenodo record (images + model outputs) has NOT been downloaded and is not needed.

## How the cutouts were fetched (reproducible)

astroquery 0.4.11, `EuclidClass(environment="PDR")`, TAP+ query `SELECT file_path, file_name FROM sedm.mosaic_product
WHERE tile_index=<tile> AND instrument_name='VIS'`, then `get_cutout(file_path=<dir>/<file>, instrument='VIS', id=<tile>,
coordinate, radius)`. `docs/fetch_q1_cutouts.py` does exactly this.

## Not verified

- Terms of use / acknowledgement text of the Euclid Q1 data (must be read before any publication or redistribution).
- The DR1 release date and whether DR1 is public from day one.
- Rate limits of the cutout service (24 cutouts took about a minute; do not hammer it: keep the 200-cutout cap).
- The exact units of the mosaic flux (BUNIT missing) and the VIS PSF FWHM quoted above.
