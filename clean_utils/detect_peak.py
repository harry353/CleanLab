from clean_utils.utils import crop_psf
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter
from scipy.signal import correlate
from clean_utils import add_component
from astropy.io import fits
import matplotlib.pyplot as plt


def regular(res, mask):
    """
    Detect the brightest pixel (peak) within the masked region.

    This is the simplest form of peak detection used in classical CLEAN.
    It identifies the global maximum value in the residual image restricted
    to the valid mask area.

    Parameters
    ----------
    res : np.ndarray
        Residual image from which to detect the brightest pixel.
    mask : np.ndarray or bool array
        Boolean mask defining valid regions for source detection.

    Returns
    -------
    peak_flux : float
        Flux value at the brightest pixel within the masked region.
    my, mx : int
        Pixel coordinates (y, x) of the detected peak.
    """
    peak_flux = np.max(res[mask])
    my, mx = np.where(res == peak_flux)
    my, mx = my[0], mx[0]
    return peak_flux, my, mx


def matched_filtering(res, psf, mask, debug=False):
    """
    Perform PSF-matched filtering for enhanced peak detection.

    Correlates the residual image with a truncated version of the PSF
    to enhance signal-to-noise ratio for faint sources. The response is
    normalized by the local RMS, producing a whitened map from which
    the brightest pixel is selected as the detected peak.

    Parameters
    ----------
    res : np.ndarray
        Input residual image.
    psf : np.ndarray
        Point spread function used for matched filtering.
    mask : np.ndarray or bool array
        Boolean mask defining valid detection regions.
    debug : bool, optional
        If True, displays diagnostic plots of intermediate filtering steps
        (default: False).

    Returns
    -------
    peak_flux : float
        Flux value at the detected peak location.
    my, mx : int
        Pixel coordinates (y, x) of the detected peak.

    Notes
    -----
    - This method is more robust to noise and sidelobes than direct maximum search.
    - The PSF is truncated to size (101 × 101) for computational efficiency.
    - The local RMS map is estimated via a moving window filter.
    """
    def local_std(img, size=7):
        mean = uniform_filter(img, size)
        mean_sq = uniform_filter(img**2, size)
        return np.sqrt(np.maximum(mean_sq - mean**2, 1e-10))
   
    trunc_psf = crop_psf(psf, size=101)

    mf_response = correlate(res, trunc_psf, mode='same')
    res_masked = np.copy(res)
    res_masked[mask] = 0
    rms_map = local_std(res_masked, size=8)
    rms_floor = np.percentile(rms_map, 20)
    rms_map = np.maximum(rms_map, rms_floor)
    whitened = mf_response / (rms_map + 1e-6)
    masked = np.where(mask, whitened, -np.inf)
    peak_score = np.max(masked)
    my, mx = np.unravel_index(np.argmax(masked), masked.shape)
    peak_flux = res[my, mx]

    if debug:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(res, origin='lower', cmap='viridis')
        axes[0].set_title("Residual")

        axes[1].imshow(mf_response, origin='lower', cmap='viridis')
        axes[1].set_title("Matched Filter Response")
        
        axes[2].imshow(rms_map, origin='lower', cmap='viridis')
        axes[2].set_title("Local RMS")
        
        axes[3].imshow(whitened, origin='lower', cmap='viridis')
        axes[3].plot(mx, my, 'rx')
        axes[3].set_title("Whitened (Peak Marked)")

        plt.tight_layout()
        plt.show()

    return peak_flux, my, mx


def multi(residual, mask, N=5, radius=5, plot_peaks=False):
    """
    Detect multiple bright peaks in parallel without sequential suppression.

    Identifies up to `N` strongest local maxima in the residual image
    within a given radius, using a fast maximum filter operation.
    This version is optimized for parallel CLEAN approaches such as
    :class:`MultiPeakClean`, where several peaks are cleaned
    concurrently in each iteration.

    Parameters
    ----------
    residual : np.ndarray
        Input residual image.
    mask : np.ndarray or bool array
        Boolean mask defining valid search regions.
    N : int, optional
        Number of brightest peaks to detect per iteration (default: 5).
    radius : int, optional
        Radius (in pixels) defining local neighborhood for peak detection (default: 5).
    plot_peaks : bool, optional
        If True, displays detected peaks overlaid on the residual image (default: False).

    Returns
    -------
    peaks : list of tuple
        List of detected peaks in the form (peak_flux, y, x).

    Notes
    -----
    - This method uses a non-overlapping maximum filter, allowing multiple
      detections per iteration without explicit suppression.
    - It is well suited for parallelized CLEAN implementations.
    """
    working_mask = (mask & np.isfinite(residual))
    local_max = (maximum_filter(residual, size=2*radius+1, mode='constant') == residual) & working_mask
    peak_vals = residual[local_max]
    if peak_vals.size == 0:
        return []

    # Sort and take top N safely
    N = min(N, peak_vals.size)
    flat_indices = np.argpartition(peak_vals, -N)[-N:]

    yx_indices = np.argwhere(local_max)
    top_peaks = yx_indices[flat_indices[np.argsort(peak_vals[flat_indices])[::-1]]]

    peaks = [(residual[y, x], y, x) for y, x in top_peaks]

    if plot_peaks:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(residual, origin='lower', cmap='viridis')
        for _, y, x in peaks:
            ax.plot(x, y, 'rx', markersize=8, markeredgewidth=2)
        ax.set_title("Top Peaks (Parallel)")
        plt.tight_layout()
        plt.show()

    return peaks


def main():
    """
    Demonstration of peak detection algorithms.

    Loads test data and visualizes detected peaks using the multi-peak
    detector. Used for standalone testing of the detection functions.

    Notes
    -----
    This function is for visualization and debugging purposes only and
    is not called within the main CLEAN pipeline.
    """
    mx, my = 80, 50
    mx_corr, my_corr = 80.2, 50.3
    pre_conv_amp = 1
    comp = np.zeros((100, 100))
    comp = add_component.point_source(pre_conv_amp, my_corr, mx_corr, my, mx, 1, np.zeros((100, 100)))

    psf = fits.getdata('wsclean-psf.fits', ext=0)[-1][-1].astype("f8")
    dirty = np.load("images/ps_image_dense.npy")

    multi(dirty, True, N=25, radius=5, plot_peaks=True)


if __name__ == "__main__":
    main()
