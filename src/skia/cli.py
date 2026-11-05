"""Command-line smooth fitting; wall-clock timing is emitted only on stderr."""
import argparse
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from skia.cutout import load_cutout
from skia.fitter import FitConfig, fit_cutout, save_fit


def main(argv: list[str] | None = None) -> int:
    """Run skia-fit cutout.npz -o fit.json; exits 0 converged, 2 failed, 3 unconverged."""
    parser = argparse.ArgumentParser(description='Fit an EPL+shear lens and Sersic source on CPU')
    parser.add_argument('cutout')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--starts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--max-nfev', type=int, default=200)
    parser.add_argument('--gamma-prior-sigma', type=float, default=None,
                        help='add Gaussian gamma prior N(2,sigma); data chi2 is reported separately')
    parser.add_argument('--source-components',type=int,choices=(1,2),default=1,
                        help='fit one Sersic or two shared-geometry Sersic components')
    args = parser.parse_args(argv)
    try:
        cutout = load_cutout(args.cutout)
        result = fit_cutout(cutout, FitConfig(args.starts, args.seed, args.max_nfev,
                                               args.gamma_prior_sigma,args.source_components))
        save_fit(result, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'skia-fit: {error}', file=sys.stderr)
        return 2
    print(f'elapsed_seconds={result.elapsed_seconds:.6f} chi2/dof={result.chi2/result.dof:.6f} '
          f'converged={result.converged}', file=sys.stderr)
    return 0 if result.converged else 3


if __name__ == '__main__':
    raise SystemExit(main())
