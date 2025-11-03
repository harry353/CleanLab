import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from clean_utils import apply_mask as am
from .factory import get_clean_strategy
from clean_utils import gain_function as gf
from clean_utils.utils import parse_gain, show_results, print_stats
from clean_utils import detect_peak as dp
from clean_utils import exit_conditions as ec


GAIN_MAP = {
    "logistic": gf.scaled_logistic,
    "constant": lambda x: x,
}


def clean(
    dirty_image, psf, mask, threshold, max_iter, iter_per_cycle,
    mode, show_plots, print_results, gain, peak_detection="regular",
    debug_results=False
):
    # -----------------------------
    # Mask selection
    # -----------------------------
    if isinstance(mask, str):
        if mask == "bgs":
            am.bg_subtr(dirty_image, "mask.npy")
            mask = np.load("mask.npy")
        elif mask == "manual":
            am.manual(dirty_image, "mask.npy")
            mask = np.load("mask.npy")
        elif mask == "none":
            mask = np.ones_like(dirty_image, dtype=bool)
        else:
            raise ValueError(f"Unknown mask mode: {mask}")

    # -----------------------------
    # Initialization
    # -----------------------------
    comps_image = np.zeros_like(dirty_image)
    residual_image = dirty_image.copy()

    gain_fn = parse_gain(gain)
    config = {
        "threshold": threshold,
        "iter_per_cycle": iter_per_cycle,
        "max_iter": max_iter,
        "gain": gain_fn,
        "peak_detection": peak_detection,
        "debug_results": debug_results,
    }

    strategy = get_clean_strategy(mode, psf, mask, config)

    # --- Ensure images are 2D ---
    dirty_image = np.squeeze(dirty_image)
    psf = np.squeeze(psf)
    mask = np.squeeze(mask)
    comps_image = np.squeeze(comps_image)
    residual_image = np.squeeze(residual_image)

    # -----------------------------
    # Main CLEAN loop (2D only)
    # -----------------------------
    for iter in tqdm(range(max_iter)):
        if iter % iter_per_cycle == 0 and iter > 0:
            residual_image = dirty_image - correlate(comps_image, psf, mode="same")

        # ---- Peak detection ----
        pd = config.get("peak_detection", "regular")

        if pd == "regular":
            peak_flux, my, mx = dp.regular(residual_image, mask)
        elif pd == "matched":
            peak_flux, my, mx = dp.matched_filtering(residual_image, psf, mask)
        elif pd == "multi":
            peaks = dp.multi(residual_image, mask, N=5)
            if not peaks:
                print("No peaks found — stopping CLEAN.")
                break
            peak_flux, my, mx = zip(*peaks)
            peak_flux = np.array(peak_flux)
            my = np.array(my, dtype=int)
            mx = np.array(mx, dtype=int)
        else:
            raise ValueError(f"Unknown peak detection method: {pd}")

        # ---- Noise stop check ----
        if ec.noise(residual_image, threshold, peak_flux):
            break

        # ---- Gain ----
        gain_value = peak_flux / np.std(residual_image)
        gain = config["gain"](gain_value)

        # ---- CLEAN step ----
        comps_image, residual_image, should_stop = strategy.step(
            comps_image, residual_image, peak_flux, my, mx, gain
        )
        if should_stop:
            break

    if hasattr(strategy, "finalize"):
        strategy.finalize()

    if show_plots:
        show_results(dirty_image, comps_image, mask, residual_image, psf)
    if print_results:
        print_stats(comps_image, psf, residual_image, dirty_image, iter, mask)

    return comps_image, residual_image, iter
