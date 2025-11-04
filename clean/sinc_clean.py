from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy
from clean_utils import perform_fit


class PointSourceClean(CleanStrategy):
    """
    Point-source CLEAN strategy using subpixel Gaussian fitting.

    This variant refines the position of detected peaks via a 2D Gaussian fit
    to estimate subpixel coordinates of compact sources. The fitted coordinates
    are then used to place sinc-shaped CLEAN components that better represent
    the true source centroids. It is particularly suited for datasets with
    bright, unresolved point sources.

    Parameters
    ----------
    psf : np.ndarray
        Point spread function (PSF) of the observation.
    mask : np.ndarray or bool array
        Boolean mask defining valid cleaning regions.
    config : dict
        Configuration dictionary containing parameters such as gain, threshold,
        and debug options.
    """

    def __init__(self, psf, mask, config):
        """
        Initialize the PointSourceClean strategy.

        Crops the PSF to a specified size for faster convolution and sets
        up internal configuration parameters.

        Parameters
        ----------
        psf : np.ndarray
            Input point spread function used for PSF subtraction.
        mask : np.ndarray or bool array
            Boolean array defining regions to clean.
        config : dict
            Configuration dictionary containing runtime parameters.
        """
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=1023)

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        """
        Perform one CLEAN iteration using Gaussian-fitted point-source subtraction.

        Fits a 2D Gaussian to the region around the current residual peak to
        determine its subpixel position. The resulting fitted coordinates are
        used to add a sinc-weighted CLEAN component and subtract the corresponding
        PSF contribution from the residual image.

        Parameters
        ----------
        clean_image : np.ndarray
            Current accumulated clean component image.
        residual_image : np.ndarray
            Current residual image after previous iterations.
        peak_flux : float
            Flux value of the detected peak.
        my, mx : int
            Pixel coordinates (y, x) of the detected peak in the residual image.
        gain : float
            CLEAN gain factor controlling subtraction strength.

        Returns
        -------
        clean_image : np.ndarray
            Updated clean component image.
        residual_image : np.ndarray
            Updated residual image after PSF subtraction.
        should_stop : bool
            Always False; iteration continues until threshold or max iterations are reached.

        Notes
        -----
        - The fitted centroid `(true_y, true_x)` provides subpixel accuracy
          in component placement, improving reconstruction quality for unresolved sources.
        - The truncation size (1023) ensures that PSF convolution is efficient
          while maintaining high spatial fidelity.
        """
        fit_image_size = 10
        popt = perform_fit.gauss(residual_image, my, mx, fit_image_size)
        y_min = max(0, my - fit_image_size)
        x_min = max(0, mx - fit_image_size)
        true_y, true_x = y_min + popt[2], x_min + popt[1]

        clean_image = ac.point_source(peak_flux, true_y, true_x, my, mx, gain, clean_image)
        residual_image = ur.point_source(peak_flux, true_y, true_x, my, mx, gain, residual_image, self.trunc_psf)
        return clean_image, residual_image, False
