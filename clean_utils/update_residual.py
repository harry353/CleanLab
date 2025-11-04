import numpy as np
from scipy.signal import correlate
from astropy.io import fits
from clean_utils.utils import adjust_bounds
import matplotlib.pyplot as plt
from . import perform_fit
from . import add_component as ac


def point_source(peak_flux, my_corr, mx_corr, my, mx, gain, res, trunc_psf):
    """
    Subtract a sinc-shaped point-source component from the residual image.

    Generates a PSF-convolved sinc kernel corresponding to a subpixel-corrected
    source position, then subtracts it from the residual map. This function
    provides a smoother, more physically accurate subtraction than the simple
    delta-function approach.

    Parameters
    ----------
    peak_flux : float
        Peak flux of the detected source.
    my_corr, mx_corr : float
        Subpixel-corrected centroid coordinates of the fitted source (y, x).
    my, mx : int
        Integer coordinates of the detected peak.
    gain : float
        CLEAN gain factor controlling subtraction strength.
    res : np.ndarray
        Current residual image to update.
    trunc_psf : np.ndarray
        Truncated point spread function (PSF) used for convolution.

    Returns
    -------
    res : np.ndarray
        Updated residual image after subtracting the convolved point-source model.

    Notes
    -----
    - The subtraction uses a PSF-convolved version of the sinc-shaped CLEAN
      component to account for subpixel offsets.
    - This version is computationally heavier than delta CLEAN but provides
      improved residual structure near bright point sources.
    """
    windowed_sinc = ac.point_source(peak_flux, my_corr, mx_corr, my, mx, gain,
                                    np.zeros((res.shape[0], res.shape[0])))
    conv_windowed_sinc = correlate(windowed_sinc, trunc_psf, mode="same")

    res -= conv_windowed_sinc
    return res


def delta(mx, my, gain, peak_flux, res, trunc_psf):
    """
    Subtract a delta-function (pixel-centered) component from the residual image.

    Performs a standard Clark-style subtraction using a scaled and shifted PSF
    centered on the detected peak. This is the simplest and fastest form of
    residual update in the CLEAN algorithm.

    Parameters
    ----------
    mx, my : int
        Pixel coordinates (x, y) of the detected peak.
    gain : float
        CLEAN gain factor controlling subtraction strength.
    peak_flux : float
        Flux of the detected peak.
    res : np.ndarray
        Current residual image.
    trunc_psf : np.ndarray
        Truncated point spread function used for subtraction.

    Returns
    -------
    res : np.ndarray
        Updated residual image after subtracting the scaled PSF.

    Notes
    -----
    - The subtraction is spatially bounded using :func:`utils.adjust_bounds`
      to ensure correct centering and edge handling.
    """
    res_subtr = gain * peak_flux * trunc_psf
    y_min, y_max, x_min, x_max, psf_y_start, psf_y_end, psf_x_start, psf_x_end = adjust_bounds(
        center=(my, mx), patch_size=res_subtr.shape, image_shape=res.shape
    )
    res[y_min:y_max, x_min:x_max] -= res_subtr[psf_y_start:psf_y_end, psf_x_start:psf_x_end]
    return res


def full(dirty, comp, psf):
    """
    Compute the full residual image via convolution.

    Reconstructs the residual image as the difference between the dirty image
    and the convolution of the CLEAN components with the PSF. This represents
    the exact residual state after a complete major cycle.

    Parameters
    ----------
    dirty : np.ndarray
        Original dirty image.
    comp : np.ndarray
        CLEAN component image.
    psf : np.ndarray
        Point spread function (PSF) used for convolution.

    Returns
    -------
    res : np.ndarray
        Fully recomputed residual image.
    """
    reconv = correlate(comp, psf, mode='same')
    res = dirty - reconv
    return res


def main():
    """
    Demonstration of residual update methods.

    Compares three residual computation approaches:
    - `point_source()`: sinc-based subpixel residual subtraction.
    - `delta()`: pixel-centered delta subtraction.
    - `full()`: full recomputation via convolution.

    Displays the residual maps and reports their standard deviations
    to illustrate accuracy and consistency across methods.

    Notes
    -----
    This function is used for visualization and testing only.
    It is not invoked by the main CLEAN pipeline.
    """
    mx, my = 80, 50
    mx_corr, my_corr = 80.2, 50.3
    pre_conv_amp = 1
    comp = np.zeros((100, 100))
    comp = ac.point_source(pre_conv_amp, my_corr, mx_corr, my, mx, 1, np.zeros((100, 100)))

    psf = fits.getdata('wsclean-psf.fits', ext=0)[-1][-1].astype("f8")

    dirty = correlate(comp, psf, mode='same')
    dirty_copy = np.copy(dirty)

    fit_image_size = 10
    popt = perform_fit.gauss(dirty_copy, my, mx, fit_image_size)
    true_y, true_x = my - fit_image_size + popt[2], mx - fit_image_size + popt[1]
    peak_flux = dirty.max()

    trunc_psf_size = 201
    trunc_psf = psf[psf.shape[0]//2-(trunc_psf_size-1):psf.shape[0]//2+trunc_psf_size,
                    psf.shape[1]//2-(trunc_psf_size-1):psf.shape[1]//2+trunc_psf_size]
    gain = 1
    res_ps = point_source(peak_flux, true_y, true_x, my, mx, gain, np.copy(dirty), trunc_psf)
    res_delta = delta(mx, my, gain, peak_flux, np.copy(dirty), trunc_psf)
    res_full = full(np.copy(dirty), comp, psf)
    
    std_ps = np.std(res_ps)
    std_delta = np.std(res_delta)
    std_full = np.std(res_full)

    print(f"Point source residual standard deviation: {std_ps}")
    print(f"Delta residual standard deviation: {std_delta}")
    print(f"Full residual standard deviation: {std_full}")

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0][0].imshow(dirty_copy, vmin=dirty_copy.min(), vmax=dirty_copy.max())
    axes[0][0].set_title("Dirty")

    axes[0][1].imshow(res_delta, vmin=dirty_copy.min(), vmax=dirty_copy.max())
    axes[0][1].set_title("Residual (Delta)")

    axes[1][0].imshow(res_ps, vmin=dirty_copy.min(), vmax=dirty_copy.max())
    axes[1][0].set_title("Residual (PS)")

    axes[1][1].imshow(res_full, vmin=dirty_copy.min(), vmax=dirty_copy.max())
    axes[1][1].set_title("Residual (Full)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
