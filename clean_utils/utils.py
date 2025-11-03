from . import gain_function as gf
from contextlib import contextmanager
from collections import defaultdict
from scipy.signal import correlate
import time
import numpy as np
import matplotlib.pyplot as plt

timings = defaultdict(float)

@contextmanager
def timed_block(name):
    start = time.perf_counter()
    yield
    timings[name] += time.perf_counter() - start

def crop_psf(psf, size):
    center = np.array(psf.shape) // 2
    half = size // 2
    y0, y1 = center[0] - half, center[0] + half + 1
    x0, x1 = center[1] - half, center[1] + half + 1
    return psf[y0:y1, x0:x1]

def adjust_bounds(center, patch_size, image_shape):
    cy, cx = center
    ph, pw = patch_size
    ih, iw = image_shape

    half_ph = ph // 2
    half_pw = pw // 2

    # image patch indices
    min_y = max(0, cy - half_ph)
    max_y = min(ih, cy + half_ph + 1)
    min_x = max(0, cx - half_pw)
    max_x = min(iw, cx + half_pw + 1)

    # kernel patch indices
    patch_y_start = max(0, half_ph - cy) if cy - half_ph < 0 else 0
    patch_y_end = patch_y_start + (max_y - min_y)

    patch_x_start = max(0, half_pw - cx) if cx - half_pw < 0 else 0
    patch_x_end = patch_x_start + (max_x - min_x)

    return min_y, max_y, min_x, max_x, patch_y_start, patch_y_end, patch_x_start, patch_x_end

def parse_gain(gain):
    if isinstance(gain, str):
        key, *arg = gain.lower().split(":")
        if key == "constant":
            value = float(arg[0]) if arg else 0.1
            return lambda _: value
        elif key in ("logistic", "linear"):  # extendable
            return {
                "logistic": gf.scaled_logistic,
            }[key]
        else:
            raise ValueError(f"Unknown gain function: '{gain}'")
    elif isinstance(gain, (int, float)):
        return lambda _: float(gain)
    elif callable(gain):
        return gain
    else:
        raise TypeError(f"Unsupported gain type: {type(gain)}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import correlate

# -----------------------------
# 2D Gaussian model
# -----------------------------
def gaussian_2d(coords, amp, x0, y0, sigma_x, sigma_y, theta):
    x, y = coords
    x0, y0 = float(x0), float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return amp * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

# -----------------------------
# Clean beam fitting
# -----------------------------
def fit_clean_beam(psf, fit_radius=15):
    """
    Fit a 2D Gaussian to the central lobe of the PSF and return the clean beam.
    """
    ny, nx = psf.shape
    y, x = np.mgrid[0:ny, 0:nx]
    cy, cx = ny // 2, nx // 2

    # Extract central region
    ymin, ymax = cy - fit_radius, cy + fit_radius + 1
    xmin, xmax = cx - fit_radius, cx + fit_radius + 1
    psf_cut = psf[ymin:ymax, xmin:xmax]
    y_cut, x_cut = np.mgrid[0:psf_cut.shape[0], 0:psf_cut.shape[1]]

    # Initial parameter guesses: amplitude, center, sigmas, rotation
    p0 = [psf_cut.max(), psf_cut.shape[1]/2, psf_cut.shape[0]/2, 3, 3, 0]

    popt, _ = curve_fit(
        gaussian_2d,
        (x_cut, y_cut),
        psf_cut.ravel(),
        p0=p0,
        maxfev=10000
    )

    amp, x0, y0, sigma_x, sigma_y, theta = popt

    # Create clean beam on full PSF grid
    clean_beam = gaussian_2d((x, y), 1.0, cx, cy, sigma_x, sigma_y, theta)
    clean_beam /= clean_beam.max()  # Normalize to unit peak

    return clean_beam

# -----------------------------
# Visualization
# -----------------------------
def show_results(dirty, clean_components, mask, residual_image, clean_beam):
    """
    Display the dirty image, clean components, residuals, mask, clean beam,
    and the final clean image (components convolved with clean beam).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Clean image = clean components convolved with clean beam
    clean_map = correlate(clean_components, clean_beam, mode='same')

    vmin_val = min(dirty.min(), clean_map.min())
    vmax_val = max(dirty.max(), clean_map.max())

    axes[0, 0].imshow(dirty, origin='lower', cmap='viridis', vmin=vmin_val, vmax=vmax_val)
    axes[0, 0].set_title("Dirty Image")

    axes[0, 1].imshow(clean_components, origin='lower', cmap='viridis', vmin=vmin_val, vmax=vmax_val)
    axes[0, 1].set_title("Clean Components")

    axes[0, 2].imshow(clean_map, origin='lower', cmap='viridis', vmin=vmin_val, vmax=vmax_val)
    axes[0, 2].set_title("Clean Image (Components * Clean Beam)")

    axes[1, 0].imshow(residual_image, origin='lower', cmap='viridis')
    axes[1, 0].set_title("Residual Image")

    axes[1, 1].imshow(mask, origin='lower', cmap='gray')
    axes[1, 1].set_title("Mask")

    axes[1, 2].imshow(clean_beam, origin='lower', cmap='viridis')
    axes[1, 2].set_title("Clean Beam")

    plt.tight_layout()
    plt.show()

def print_stats(comp, psf, res, dirty_image, iters, mask):
    iters += 1
    reconv = correlate(comp, psf, mode='same')
    rmse = np.sqrt(np.mean((reconv - dirty_image)**2))
    std = np.std(res)
    peak_flux = np.max(res[mask])
    res_std = np.std(res)
    noise = peak_flux / res_std

    print(f"\n{'Iterations':15} {'RMSE':15} {'Standard Dev':15} {'Noise Sigma':15}")
    print("-" * 85)
    print(f"{iters:<15d} {rmse:<15.8f} {std:<15.8f} {noise:<15.8f}")