import numpy as np
from concurrent.futures import ThreadPoolExecutor
from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy

class MultiPeakClean(CleanStrategy):
    """
    Multi-peak CLEAN strategy with parallel PSF subtraction.

    This CLEAN variant subtracts multiple strong peaks per iteration instead
    of one. The `detect_peak.multi()` function identifies the top `N` brightest
    peaks within the mask, which are then cleaned simultaneously. Each peak’s
    PSF subtraction is performed in parallel across multiple threads, improving
    speed on modern CPUs. The resulting residuals are averaged to maintain
    physical consistency in the final image.

    Parameters
    ----------
    psf : np.ndarray
        The point spread function (PSF) of the observation.
    mask : np.ndarray or bool array
        Boolean mask defining valid cleaning regions.
    config : dict
        Configuration dictionary containing algorithm parameters such as
        gain, threshold, number of peaks per iteration, and thread count.
    """

    def __init__(self, psf, mask, config):
        """
        Initialize the MultiPeakClean strategy.

        Crops the PSF for faster convolution and retrieves user-defined
        configuration values controlling the number of peaks, suppression
        radius, debug mode, and thread count.

        Parameters
        ----------
        psf : np.ndarray
            Input point spread function.
        mask : np.ndarray or bool array
            Boolean cleaning mask.
        config : dict
            Configuration dictionary containing:
            - ``multi_N`` (int): Number of peaks to clean per iteration (default: 25).
            - ``multi_radius`` (int): Suppression radius for peak detection (default: 5).
            - ``threads`` (int): Number of worker threads for parallel subtraction (default: 4).
            - ``debug_results`` (bool): Enables extra plotting or diagnostics.
        """
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=101)
        self.N = config.get("multi_N", 25)
        self.radius = config.get("multi_radius", 5)
        self.debug_results = config.get("debug_results", False)
        self.num_threads = config.get("threads", 4)

    def subtract_peak(self, args):
        """
        Subtract a single CLEAN component from the residual map.

        This helper function performs PSF subtraction for one detected
        point source. It is designed to be called concurrently by multiple
        threads in :meth:`step`.

        Parameters
        ----------
        args : tuple
            Tuple containing:
            (peak_flux, y, x, gain, residual_copy)

        Returns
        -------
        np.ndarray
            Updated residual image after PSF subtraction for this source.
        """
        val, y, x, gain, residual_copy = args
        # Explicitly name arguments for safety
        return ur.delta(mx=x, my=y, gain=gain, peak_flux=val,
                        res=residual_copy, trunc_psf=self.trunc_psf)

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        """
        Perform one iteration of multi-peak CLEAN.

        Adds delta components for all detected peaks, then performs PSF
        subtraction for each peak concurrently using a thread pool.
        The residuals from all threads are averaged to produce the updated
        residual map.

        Parameters
        ----------
        clean_image : np.ndarray
            Current accumulated clean components.
        residual_image : np.ndarray
            Current residual image.
        peak_flux : array_like
            Flux values of the detected peaks.
        my, mx : array_like
            Pixel coordinates of the detected peaks (y, x).
        gain : float
            CLEAN gain factor controlling subtraction strength.

        Returns
        -------
        clean_image : np.ndarray
            Updated clean component image.
        residual_image : np.ndarray
            Updated residual image after concurrent subtraction.
        should_stop : bool
            Always False; iteration continues until stopping criteria are met.
        """
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
