import numpy as np
from concurrent.futures import ThreadPoolExecutor
from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy

class MultiPeakClean(CleanStrategy):
    """
    Parallel version of MultiPeakClean.
    Detects N strongest peaks externally (via dp.multi)
    and subtracts them concurrently.
    """

    def __init__(self, psf, mask, config):
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=101)
        self.N = config.get("multi_N", 25)
        self.radius = config.get("multi_radius", 5)
        self.debug_results = config.get("debug_results", False)
        self.num_threads = config.get("threads", 4)

    def subtract_peak(self, args):
        """Worker function for parallel PSF subtraction."""
        val, y, x, gain, residual_copy = args
        # Explicitly name arguments for safety
        return ur.delta(mx=x, my=y, gain=gain, peak_flux=val,
                        res=residual_copy, trunc_psf=self.trunc_psf)

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        # Handle the case where no peaks were detected
        if peak_flux is None or len(np.atleast_1d(peak_flux)) == 0:
            return clean_image, residual_image, True

        # Convert to arrays for safety
        peak_flux = np.atleast_1d(peak_flux)
        my = np.atleast_1d(my)
        mx = np.atleast_1d(mx)

        # ---- Add all delta components ----
        for val, y, x in zip(peak_flux, my, mx):
            clean_image = ac.delta(val, gain, clean_image, y, x)

        # ---- Parallel PSF subtraction ----
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            residuals = list(
                executor.map(
                    self.subtract_peak,
                    [(val, y, x, gain, residual_image.copy())
                     for val, y, x in zip(peak_flux, my, mx)]
                )
            )

        # ---- Combine updated residuals (mean is physically consistent) ----
        residual_image = np.mean(residuals, axis=0)

        return clean_image, residual_image, False
