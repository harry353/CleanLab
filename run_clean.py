import argparse
from astropy.io import fits
import numpy as np
from clean import clean
from clean.core_spectral import clean_spectral

def main():
    parser = argparse.ArgumentParser(description="Run CLEAN deconvolution using CleanLab.")
    parser.add_argument("--dirty_image", required=True, help="Path to the dirty image or cube FITS file.")
    parser.add_argument("--psf", required=True, help="Path to the PSF FITS file.")
    parser.add_argument("--threshold", type=float, required=True, help="Noise threshold (sigma).")
    parser.add_argument("--max_iter", type=int, required=True, help="Maximum number of iterations.")
    parser.add_argument("--mode", required=True, choices=["clark", "sinc", "cluster", "multi", "spectral"], help="CLEAN strategy.")
    parser.add_argument("--mask", default="none", help="Mask type (none | manual | bgs).")
    parser.add_argument("--iter_per_cycle", type=int, default=100, help="Iterations per major cycle.")
    parser.add_argument("--show_plots", action="store_true", help="Show intermediate plots.")
    parser.add_argument("--print_results", action="store_true", help="Print iteration statistics.")
    parser.add_argument("--gain", type=float, default=0.2, help="Loop gain or gain function.")
    parser.add_argument("--peak_detection", default="regular", help="Peak detection method.")
    parser.add_argument("--debug_results", action="store_true", help="Enable debug output.")
    args = parser.parse_args()

    # Load data
    dirty_data = fits.getdata(args.dirty_image, ext=0).astype("f8")
    psf = fits.getdata(args.psf, ext=0).astype("f8")

    dirty_data = np.squeeze(dirty_data)
    psf = np.squeeze(psf)

    # --- Select between 2D CLEAN or Spectral Cube CLEAN ---
    if args.mode == "spectral":
        print("Running Spectral CLEAN (cube mode)...")
        clean_image, residual_image, n_iter = clean_spectral(
            dirty_cube=dirty_data,
            psf=psf,
            gain=args.gain,
            threshold=args.threshold,
            max_iter=args.max_iter,
            iter_per_cycle=args.iter_per_cycle,
            show_plots=args.show_plots,
            print_results=args.print_results,
            debug_results=args.debug_results,
        )
        # Save results to FITS
        fits.writeto("images/clean_cube.fits", clean_image, overwrite=True)
        fits.writeto("images/residual_cube.fits", residual_image, overwrite=True)
    else:
        print(f"Running {args.mode.capitalize()} CLEAN (2D mode)...")
        clean_image, residual_image, n_iter = clean(
            dirty_image=dirty_data,
            psf=psf,
            threshold=args.threshold,
            max_iter=args.max_iter,
            mode=args.mode,
            mask=args.mask,
            iter_per_cycle=args.iter_per_cycle,
            show_plots=args.show_plots,
            print_results=args.print_results,
            gain=args.gain,
            peak_detection=args.peak_detection,
            debug_results=args.debug_results,
        )
        # Save results to FITS
        fits.writeto("images/clean_image.fits", clean_image, overwrite=True)
        fits.writeto("images/residual_image.fits", residual_image, overwrite=True)

    print(f"\nCLEAN completed in {n_iter} iterations.")

if __name__ == "__main__":
    main()
