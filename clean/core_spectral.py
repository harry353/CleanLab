import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from clean_utils import gain_function as gf
from clean_utils.utils import parse_gain, show_results, print_stats
from clean_utils import detect_peak as dp
from clean_utils import exit_conditions as ec
from .factory import get_clean_strategy


GAIN_MAP = {
    "logistic": gf.scaled_logistic,
    "constant": lambda x: x,
}


def clean_spectral(
    dirty_cube, psf, threshold, max_iter, iter_per_cycle,
    show_plots, print_results, gain, debug_results=False
):
    """
    Spectral CLEAN: iterates through frequency channels independently.
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

    from astropy.io import fits

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
        from clean_utils.utils import print_stats
        # Use first channel's mask (dummy)
        mask = np.ones_like(dirty_cube[0], dtype=bool)
        print_stats(comps_cube[0], psf, residual_cube[0], dirty_cube[0], iter, mask)

    # if print_results:
    #     print_stats(comps_cube, psf, residual_cube, dirty_cube, iter, mask)

    return comps_cube, residual_cube, iter
