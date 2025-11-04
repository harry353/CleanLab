class CleanStrategy:
    """
    Abstract base class for CLEAN algorithm strategies.

    All CLEAN strategy implementations (e.g., Clark, Sinc, Cluster,
    MultiPeak, Spectral) inherit from this class and implement their
    own version of the :meth:`step` method. This provides a unified
    interface for performing one iteration of the CLEAN deconvolution
    process, regardless of the underlying algorithmic details.

    Parameters
    ----------
    psf : np.ndarray
        Point spread function (PSF) used for PSF subtraction.
    mask : np.ndarray or bool array
        Boolean array defining valid regions for CLEANing.
    config : dict
        Dictionary of configuration parameters controlling the algorithm
        (e.g., gain, threshold, iteration limits, peak detection mode).
    """

    def __init__(self, psf, mask, config):
        """
        Initialize the base CLEAN strategy.

        Stores the PSF, mask, and configuration dictionary that
        derived strategies will use during CLEAN iterations.

        Parameters
        ----------
        psf : np.ndarray
            Input point spread function (PSF).
        mask : np.ndarray or bool array
            Boolean mask specifying allowed CLEANing regions.
        config : dict
            Configuration dictionary containing algorithm parameters.
        """
        self.psf = psf
        self.mask = mask
        self.config = config

    def step(self, clean_image, residual_image, dirty_image, iter, peak_flux, my, mx):
        """
        Perform a single iteration step of the CLEAN algorithm.

        This method must be implemented by all subclasses. It defines
        the algorithm-specific logic for updating the clean and residual
        images based on detected peaks.

        Parameters
        ----------
        clean_image : np.ndarray
            Current accumulated clean components.
        residual_image : np.ndarray
            Current residual image after previous iterations.
        dirty_image : np.ndarray
            Original dirty image (used for reference or recomputation).
        iter : int
            Current iteration number.
        peak_flux : float or array-like
            Flux of the detected peak(s) in the residual image.
        my, mx : int or array-like
            Pixel coordinates (y, x) of the detected peak(s).

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the step() method")
