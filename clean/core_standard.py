import numpy as np
from tqdm import tqdm
from scipy.signal import correlate
from clean_utils import apply_mask as am
from .factory import get_clean_strategy
from clean_utils import gain_function as gf
from clean_utils import detect_peak as dp
from clean_utils import exit_conditions as ec
from clean_utils.utils import parse_gain, show_results


GAIN_MAP = {
    "logistic": gf.scaled_logistic,
    "constant": lambda x: x,
}


def clean(
    dirty_image, psf, mask, threshold, max_iter, iter_per_cycle,
    mode, show_plots, print_results, gain, peak_detection="regular",
    debug_results=False
):
    """
    Perform 2D CLEAN deconvolution on a dirty image.

    This function implements the classical CLEAN workflow used in radio
    interferometric imaging. It iteratively subtracts scaled and shifted
    versions of the point spread function (PSF) at the location of detected
    peaks, building up a clean component image and reducing the residuals.
    The process stops when the image reaches the specified noise threshold
    or the maximum number of iterations.

    Parameters
    ----------
    dirty_image : np.ndarray
        Input dirty image to be deconvolved (2D array).
    psf : np.ndarray
        Point spread function (PSF) corresponding to the dirty image.
    mask : str or np.ndarray
        Mask defining regions allowed for cleaning. Accepts:
        - `"bgs"` : interactively define background subtraction region.
        - `"manual"` : manually select regions to clean.
        - `"none"` : clean all pixels (default behavior).
        - `np.ndarray` : boolean mask array.
    threshold : float
        Stopping criterion; cleaning halts when the peak signal-to-noise
        ratio falls below this threshold.
    max_iter : int
        Maximum number of iterations to perform.
    iter_per_cycle : int
        Number of minor iterations between full residual recomputations
        (major cycles).
    mode : str
        CLEAN strategy to use. Options correspond to available strategies
        registered in `factory.get_clean_strategy`, e.g.:
        `"clark"`, `"sinc"`, `"cluster"`, `"multi"`, etc.
    show_plots : bool
        If True, displays diagnostic plots of the dirty, clean, and residual images.
    print_results : bool
        If True, prints performance statistics and image metrics at completion.
    gain : str, float, or callable
        CLEAN gain parameter controlling subtraction strength per iteration.
        Accepts:
        - float constant (e.g. `0.1`),
        - string (e.g. `"logistic"`, `"constant:0.05"`),
        - or callable gain function.
    peak_detection : {"regular", "matched", "multi"}, optional
        Method for detecting the next CLEAN peak:
        - `"regular"` : select the global maximum within the mask.
        - `"matched"` : use PSF-matched filtering to enhance detection.
        - `"multi"` : detect multiple peaks simultaneously per iteration.
    debug_results : bool, optional
        If True, enables debug plotting and per-step diagnostics within the
        selected CLEAN strategy (default: False).

    Returns
    -------
    comps_image : np.ndarray
        Clean component image representing the reconstructed sky model.
    residual_image : np.ndarray
        Residual image after PSF subtraction.
    iter : int
        Number of iterations completed before stopping.

    Notes
    -----
    - This function supports interactive mask creation and multiple
      peak-detection modes for testing advanced CLEAN variants.
    - The active CLEAN algorithm is selected dynamically via
      `get_clean_strategy(mode, psf, mask, config)`, allowing easy switching
      between implementations (e.g., Clark, Sinc, Cluster CLEAN, etc.).
    - Results can be visualized (`show_plots=True`) or printed
      (`print_results=True`) after the run completes.
    """

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
