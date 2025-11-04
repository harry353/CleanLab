from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from clean_utils import gain_function as gf
from clean_utils import detect_peak as dp
from clean_utils import exit_conditions as ec
from .factory import get_clean_strategy
from clean_utils.utils import parse_gain, show_results


GAIN_MAP = {
    "logistic": gf.scaled_logistic,
    "constant": lambda x: x,
}


def clean_spectral(
    dirty_cube, psf, threshold, max_iter, iter_per_cycle,
    show_plots, print_results, gain, debug_results=False
):
    """
    Perform spectral CLEAN deconvolution on a 3D image cube.

    This function applies the CLEAN algorithm independently to each
    frequency channel of a spectral cube. The algorithm iteratively
    subtracts scaled and shifted versions of the point spread function (PSF)
    from the dirty cube to reconstruct the underlying source structure
    and produce a clean component cube and a residual cube.

    Parameters
    ----------
    dirty_cube : np.ndarray
        Input dirty image cube of shape (nchan, ny, nx).
        Each slice along the first axis corresponds to a frequency channel.
    psf : np.ndarray
        2D point spread function (PSF) used for deconvolution.
    threshold : float
        Noise stopping threshold; cleaning stops when the peak signal-to-noise
        ratio falls below this value.
    max_iter : int
        Maximum number of CLEAN iterations to perform.
    iter_per_cycle : int
        Number of iterations per major cycle (after which a full residual
        recomputation via convolution is performed).
    show_plots : bool
        If True, display visual diagnostic plots after cleaning.
    print_results : bool
        If True, print performance metrics and basic statistics at the end.
    gain : str, float, or callable
        CLEAN loop gain. May be specified as:
        - a float constant,
        - a string such as "logistic" or "constant:0.1", or
        - a custom callable gain function.
    debug_results : bool, optional
        If True, enables extra diagnostic output and visualization from
        the active CLEAN strategy (default: False).

    Returns
    -------
    comps_cube : np.ndarray
        Clean component cube of the same shape as `dirty_cube`.
        Represents the recovered model of the sky brightness.
    residual_cube : np.ndarray
        Residual cube after subtraction of the CLEAN components.
    iter : int
        The number of iterations performed before reaching the stopping condition.

    Notes
    -----
    - This implementation currently treats each spectral channel independently,
      using a shared PSF for all channels.
    - The results are automatically saved as FITS files:
        * `images/clean_cube.fits`
        * `images/residual_cube.fits`
    """

    nchan = dirty_cube.shape[0]
    comps_cube = np.zeros_like(dirty_cube)
    residual_cube = dirty_cube.copy()

    # Dummy 2D mask (ignored)
    mask = np.ones_like(dirty_cube[0], dtype=bool)

    gain_fn = parse_gain(gain)
    config = {
        "threshold": threshold,
        "iter_per_cycle": iter_per_cycle,
        "max_iter": max_iter,
        "gain": gain_fn,
        "debug_results": debug_results,
    }

    strategy = get_clean_strategy("spectral", psf, mask, config)

    # -----------------------------
    # Main loop
    # -----------------------------
    for iter in tqdm(range(max_iter)):
        if iter % iter_per_cycle == 0 and iter > 0:
            for ch in range(nchan):
                residual_cube[ch] = dirty_cube[ch] - correlate(comps_cube[ch], psf, mode="same")

        peak_flux = np.zeros(nchan)
        my = np.zeros(nchan, dtype=int)
        mx = np.zeros(nchan, dtype=int)

        for ch in range(nchan):
            pf, y, x = dp.regular(residual_cube[ch], mask)
            peak_flux[ch], my[ch], mx[ch] = pf, y, x

        if ec.noise(residual_cube, threshold, peak_flux):
            break

        gain_value = np.mean(peak_flux / np.std(residual_cube))
        gain = config["gain"](gain_value)

        comps_cube, residual_cube, should_stop = strategy.step(
            comps_cube, residual_cube, peak_flux, my, mx, gain
        )
        if should_stop:
            break

    if hasattr(strategy, "finalize"):
        strategy.finalize()

    # -----------------------------
    # Save results for spectral cube
    # -----------------------------
    output_path = "images/clean_cube.fits"
    residual_path = "images/residual_cube.fits"

    fits.writeto(output_path, comps_cube.astype("float32"), overwrite=True)
    fits.writeto(residual_path, residual_cube.astype("float32"), overwrite=True)

    print(f"\nSaved clean cube to: {output_path}")
    print(f"Saved residual cube to: {residual_path}")

    if print_results:
        # Use first channel's mask (dummy)
        mask = np.ones_like(dirty_cube[0], dtype=bool)
        print_stats(comps_cube[0], psf, residual_cube[0], dirty_cube[0], iter, mask)

    # if print_results:
    #     print_stats(comps_cube, psf, residual_cube, dirty_cube, iter, mask)

    return comps_cube, residual_cube, iter
