import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import correlate
from clean_utils import add_component
from astropy.io import fits
import matplotlib.pyplot as plt
from clean_utils.utils import crop_psf
from scipy.ndimage import maximum_filter

def regular(res, mask):
    peak_flux = np.max(res[mask])
    my, mx = np.where(res == peak_flux)
    my, mx = my[0], mx[0]
    return peak_flux, my, mx

def matched_filtering(res, psf, mask, debug=False):
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

import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

def multi(residual, mask, N=5, radius=5, plot_peaks=False):
    """Parallel-friendly multi-peak detection (no sequential suppression)."""
    working_mask = (mask & np.isfinite(residual))
    local_max = (maximum_filter(residual, size=2*radius+1, mode='constant') == residual) & working_mask
    peak_vals = residual[local_max]
    if peak_vals.size == 0:
        return []

    # Sort and take top N
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
    mx, my = 80, 50
    mx_corr, my_corr = 80.2, 50.3
    pre_conv_amp = 1
    comp = np.zeros((100, 100))
    comp = add_component.point_source(pre_conv_amp, my_corr, mx_corr, my, mx, 1, np.zeros((100, 100)))

    psf = fits.getdata('wsclean-psf.fits', ext=0)[-1][-1].astype("f8")
    # dirty = fits.getdata('wsclean-dirty.fits', ext=0)[-1][-1].astype("f8")
    dirty = np.load("images/ps_image_dense.npy")

    # peak_flux, my, mx = matched_filtering(dirty, psf, True, debug=True)
    # print(peak_flux, my, mx)

    multi(dirty, True, N=25, radius=5, plot_peaks=True)

if __name__ == "__main__":
    main()