#!/usr/bin/env python3
"""Fetch small Euclid Q1 VIS cutouts of strong-lens candidates from the public ESA archive (written by Claude, P0).

Selects Q1 discovery-engine lens candidates of the given grade that also have an Euclid reference lens model
(data/ref/modeling_lens_mass.csv), sorted by the reference Einstein radius and taken at a regular stride so the
sample spans small to large lenses. Skips files that already exist, appends to data/q1_cutouts/manifest.csv.

  pip install astroquery astropy
  python docs/fetch_q1_cutouts.py --n 24 --offset 0          # the 24 cutouts that ship in data/
  python docs/fetch_q1_cutouts.py --n 100 --offset 1         # 100 more, a different stride phase

Needs network access to the ESA Euclid archive (TAP+ query of sedm.mosaic_product, then the cutout service).
Each cutout is ~66 kB. Limit for the coding agent: at most 200 cutouts in total (data/ must stay small).
"""
import argparse, csv, os, sys
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.esa.euclid import EuclidClass

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--offset", type=int, default=0, help="stride phase: take candidates offset, offset+step, ...")
ap.add_argument("--grade", default="A")
ap.add_argument("--radius", type=float, default=6.0, help="cutout radius in arcsec")
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
a = ap.parse_args()
if a.n > 200:
    sys.exit("refusing more than 200 cutouts")
REF, OUT = os.path.join(a.data, "ref"), os.path.join(a.data, "q1_cutouts")
os.makedirs(OUT, exist_ok=True)
cat = {r["id_str"]: r for r in csv.DictReader(open(os.path.join(REF, "q1_discovery_engine_lens_catalog.csv")))}
mass = {r["id_str"]: r for r in csv.DictReader(open(os.path.join(REF, "modeling_lens_mass.csv")))}
ser = {r["id_str"]: r for r in csv.DictReader(open(os.path.join(REF, "modeling_lens_sersic.csv")))}
cand = [i for i in mass if i in cat and cat[i]["grade"] == a.grade and mass[i]["einstein_radius_median_pdf"].strip()]
cand.sort(key=lambda i: float(mass[i]["einstein_radius_median_pdf"]))
step = max(1, len(cand) // max(1, a.n))
pick = cand[a.offset::step][: a.n]
e = EuclidClass(environment="PDR", verbose=False)
man = os.path.join(OUT, "manifest.csv")
cols = ["id_str", "file", "ra_deg", "dec_deg", "tile_index", "grade", "expert_score", "ref_einstein_radius_arcsec", "ref_ell_comp_0", "ref_ell_comp_1",
        "ref_shear_g1", "ref_shear_g2", "ref_lens_light_reff_arcsec", "ref_lens_light_sersic_n"]
have = set(r["id_str"] for r in csv.DictReader(open(man))) if os.path.exists(man) else set()
new = not os.path.exists(man)
with open(man, "a", newline="") as mf:
    w = csv.writer(mf)
    if new:
        w.writerow(cols)
    for i in pick:
        dest = os.path.join(OUT, f"{i}_VIS.fits")
        if i in have and os.path.exists(dest):
            continue
        r, m, s = cat[i], mass[i], ser.get(i, {})
        tile = int(r["tile_index"])
        t = e.launch_job(f"SELECT file_path, file_name FROM sedm.mosaic_product WHERE tile_index={tile} AND instrument_name='VIS'").get_results()
        if len(t) == 0:
            print(i, "no VIS mosaic"); continue
        fp, fn = str(t["file_path"][0]), str(t["file_name"][0])
        coord = SkyCoord(float(r["right_ascension"]) * u.deg, float(r["declination"]) * u.deg)
        files = None
        for path in (fp.rstrip("/") + "/" + fn, fp):
            files = e.get_cutout(file_path=path, instrument="VIS", id=tile, coordinate=coord, radius=a.radius * u.arcsec, output_file=dest)
            if files:
                break
        if not files:
            print(i, "cutout failed"); continue
        got = files[0] if isinstance(files, list) else files
        if got != dest:
            os.replace(got, dest)
        w.writerow([i, os.path.basename(dest), r["right_ascension"], r["declination"], r["tile_index"], r["grade"], r["expert_score"],
                    m["einstein_radius_median_pdf"], m["ell_comps_0_median_pdf"], m["ell_comps_1_median_pdf"], m["shear_gamma_1_median_pdf"],
                    m["shear_gamma_2_median_pdf"], s.get("effective_radius_median_pdf", ""), s.get("sersic_index_median_pdf", "")])
        mf.flush()
        print(i, "ok", os.path.getsize(dest))
