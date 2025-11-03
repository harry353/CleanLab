import numpy as np
from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy

class SpectralClean(CleanStrategy):
    """
    Spectral (multi-channel) CLEAN.
    Applies standard CLEAN independently to each channel in the cube.
    """

    def __init__(self, psf, mask, config):
        super().__init__(psf, mask, config)

        # If PSF is 3D (nchan, ny, nx), store each plane;
        # otherwise use same PSF for all channels.
        if psf.ndim == 3:
            self.trunc_psfs = [utils.crop_psf(p, size=101) for p in psf]
        else:
            self.trunc_psfs = [utils.crop_psf(psf, size=101)]

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        """
        Perform a CLEAN step for a spectral cube.
        Arguments are expected to include:
        - clean_image: 3D array (nchan, ny, nx)
        - residual_image: 3D array (nchan, ny, nx)
        - peak_flux, my, mx: coordinates of the brightest pixel per channel
        - gain: CLEAN gain value
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
