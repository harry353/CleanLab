import numpy as np
from scipy.signal.windows import hann
from .func_kernels import sinc_kernel
import matplotlib.pyplot as plt
from clean_utils.utils import adjust_bounds


def point_source(peak_flux, my_corr, mx_corr, my, mx, gain, comps, sinc_kernel_size=3):
    """
    Add a sinc-shaped point-source component to the CLEAN component image.

    Creates a subpixel-shifted, Hann-windowed 2D sinc kernel centered at
    the fitted position `(my_corr, mx_corr)` and adds it to the CLEAN component
    image `comps`. The kernel is scaled by the CLEAN gain and the detected peak
    flux, and normalized to preserve total flux.

    Parameters
    ----------
    peak_flux : float
        Flux value of the detected peak.
    my_corr, mx_corr : float
        Subpixel-corrected centroid coordinates of the fitted source (y, x).
    my, mx : int
        Integer pixel coordinates of the detected peak in the residual image.
    gain : float
        CLEAN gain factor controlling subtraction strength.
    comps : np.ndarray
        CLEAN component image to which the sinc kernel will be added.
    sinc_kernel_size : int, optional
        Size of the sinc kernel window (default: 3).

    Returns
    -------
    comps : np.ndarray
        Updated CLEAN component image with the added point-source contribution.

    Notes
    -----
    - The sinc kernel is windowed by a 2D Hann function to minimize edge artifacts.
    - The resulting kernel is flux-normalized before scaling.
    - Bounds are adjusted using :func:`utils.adjust_bounds` to ensure
      the kernel remains within image boundaries.
    """
    offset_y, offset_x = my_corr - my, mx_corr - mx

    sinc_comp = sinc_kernel(sinc_kernel_size, offset_y, offset_x)

    hann_window = np.outer(hann(sinc_kernel_size), hann(sinc_kernel_size))
    windowed_sinc = sinc_comp * hann_window
    windowed_sinc /= np.sum(windowed_sinc)
    windowed_sinc *= peak_flux * gain

    (y_min, y_max, x_min, x_max,
     kernel_y_start, kernel_y_end, kernel_x_start, kernel_x_end) = adjust_bounds(
         center=(my, mx),
         patch_size=(sinc_kernel_size, sinc_kernel_size),
         image_shape=comps.shape
     )

    comps[y_min:y_max, x_min:x_max] += windowed_sinc[
        kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end
    ]

    return comps


def delta(peak_flux, gain, comps, y, x):
    """
    Add a delta-function (point) component to the CLEAN component image.

    Adds a single-pixel contribution scaled by the peak flux and CLEAN gain
    at the specified coordinates `(y, x)`.

    Parameters
    ----------
    peak_flux : float
        Flux value of the detected peak.
    gain : float
        CLEAN gain factor controlling subtraction strength.
    comps : np.ndarray
        CLEAN component image to update.
    y, x : int
        Pixel coordinates of the detected peak.

    Returns
    -------
    comps : np.ndarray
        Updated CLEAN component image with the added delta component.
    """
    comp_val = peak_flux * gain
    comps[y, x] += comp_val
    return comps


def main():
    """
    Demonstration of the point-source component generation.

    Generates a test image with a single sinc-shaped point-source component
    added to a blank image. Displays both the integer and subpixel-corrected
    peak positions for visual verification.

    Notes
    -----
    This function is intended for visualization and testing purposes.
    It is not used within the main CLEAN pipeline.
    """
    comps = np.zeros((100, 100))
    mx_corr, my_corr = 60.5, 60.5
    mx, my = 60, 60
    peak_flux, gain = 1, 1

    comps = point_source(peak_flux, my_corr, mx_corr, my, mx, gain, comps)

    plt.imshow(comps, origin='lower')
    plt.scatter(mx, my, color='r', marker='+', label='Peak')
    plt.scatter(mx_corr, my_corr, color='g', marker='+', label='Corrected Peak')
    plt.legend()
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
