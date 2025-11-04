import numpy as np
from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy


class SpectralClean(CleanStrategy):
    """
    Multi-channel (spectral) CLEAN strategy.

    This strategy applies the standard CLEAN algorithm independently
    to each spectral channel of a 3D image cube. Each channel is cleaned
    using its corresponding PSF (if available) or a shared PSF for all
    channels. It is intended for spectral cubes where channels are
    approximately independent (e.g., radio data cubes with narrow frequency spacing).

    Parameters
    ----------
    psf : np.ndarray
        Point spread function. Can be either:
        - 2D array: shared PSF for all channels.
        - 3D array: per-channel PSF with shape (nchan, ny, nx).
    mask : np.ndarray or bool array
        Boolean mask defining valid cleaning regions in each channel.
    config : dict
        Configuration dictionary containing parameters such as
        gain, threshold, and iteration settings.
    """

    def __init__(self, psf, mask, config):
        """
        Initialize the SpectralClean strategy.

        Crops each PSF (or a single shared PSF) to improve computational
        efficiency and stores them for use during iterative subtraction.

        Parameters
        ----------
        psf : np.ndarray
            The point spread function array (2D or 3D).
        mask : np.ndarray or bool array
            Boolean cleaning mask.
        config : dict
            Dictionary of configuration options (gain, threshold, debug, etc.).
        """
        super().__init__(psf, mask, config)

        # If PSF is 3D (nchan, ny, nx), store each plane;
        # otherwise use same PSF for all channels.
        if psf.ndim == 3:
            self.trunc_psfs = [utils.crop_psf(p, size=101) for p in psf]
        else:
            self.trunc_psfs = [utils.crop_psf(psf, size=101)]

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        """
        Perform a single CLEAN iteration across all spectral channels.

        Applies delta-component addition and PSF subtraction independently
        for each channel in the cube. Each channel’s residual is updated using
        its corresponding PSF (if available).

        Parameters
        ----------
        clean_image : np.ndarray
            Current accumulated clean components cube of shape (nchan, ny, nx).
        residual_image : np.ndarray
            Current residual cube of shape (nchan, ny, nx).
        peak_flux : array_like
            Detected peak flux per channel.
        my, mx : array_like
            Pixel coordinates of detected peaks per channel (y, x).
        gain : float
            CLEAN gain factor controlling subtraction strength.

        Returns
        -------
        clean_image : np.ndarray
            Updated clean component cube.
        residual_image : np.ndarray
            Updated residual cube after PSF subtraction.
        should_stop : bool
            Always False; iteration continues until threshold or max iterations are reached.

        Notes
        -----
        - If the PSF is 2D, it is reused for all channels.
        - If a per-channel PSF cube is provided, each channel uses its own PSF.
        - This approach assumes that channels are independent and do not share
          cross-frequency correlations.
        """
        nchan = residual_image.shape[0]

        for ch in range(nchan):
            psf_ch = self.trunc_psfs[ch % len(self.trunc_psfs)]

            # Add delta component for this channel
            clean_image[ch] = ac.delta(peak_flux[ch], gain, clean_image[ch], my[ch], mx[ch])

            # Subtract scaled PSF from residual for this channel
            residual_image[ch] = ur.delta(mx[ch], my[ch], gain, peak_flux[ch],
                                          residual_image[ch], psf_ch)

        return clean_image, residual_image, False
