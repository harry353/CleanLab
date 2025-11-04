from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy

class RegularClean(CleanStrategy):
    """
    Standard implementation of the classical Clark CLEAN algorithm.

    This strategy performs iterative subtraction of scaled and shifted PSFs
    at the location of the strongest residual peak in the image. Each iteration
    adds a point-source component to the clean image and updates the residual map.

    Parameters
    ----------
    psf : np.ndarray
        The point spread function (PSF) of the imaging system.
    mask : np.ndarray or bool array
        Boolean mask defining regions allowed for cleaning.
    config : dict
        Configuration dictionary containing algorithm parameters such as
        gain, threshold, and iteration limits.
    """

    def __init__(self, psf, mask, config):
        """
        Initialize the RegularClean strategy.

        Crops the PSF to a fixed size for faster convolution operations.

        Parameters
        ----------
        psf : np.ndarray
            The input PSF used during the CLEAN deconvolution process.
        mask : np.ndarray or bool array
            Boolean array defining valid regions for source extraction.
        config : dict
            Dictionary of configuration options (gain function, threshold, etc.).
        """
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=101)

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        """
        Perform a single CLEAN iteration step.

        Subtracts a scaled and shifted PSF at the peak location from the residual
        image and adds the corresponding component to the clean image.

        Parameters
        ----------
        clean_image : np.ndarray
            Current accumulated clean components.
        residual_image : np.ndarray
            Current residual image from which the next peak will be subtracted.
        peak_flux : float
            Flux value of the detected peak in the residual image.
        my, mx : int
            Pixel coordinates of the detected peak (y, x).
        gain : float
            CLEAN gain factor controlling subtraction strength per iteration.

        Returns
        -------
        clean_image : np.ndarray
            Updated clean component image.
        residual_image : np.ndarray
            Updated residual image after PSF subtraction.
        should_stop : bool
            Always False for this method, indicating cleaning should continue.
        """
        clean_image = ac.delta(peak_flux, gain, clean_image, my, mx)
        residual_image = ur.delta(mx, my, gain, peak_flux, residual_image, self.trunc_psf)

        return clean_image, residual_image, False
