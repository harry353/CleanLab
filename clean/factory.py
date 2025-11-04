from .clark import RegularClean
from .sinc_clean import PointSourceClean
from .cluster_clean import ClusterClean
from .multi_peak_clean import MultiPeakClean
from .spectral_clean import SpectralClean

def get_clean_strategy(mode, psf, mask, config):
    """
    Return the appropriate CLEAN strategy class based on the selected mode.

    This factory function dynamically instantiates and returns a CLEAN
    strategy object corresponding to the requested algorithm. Each strategy
    class implements a variant of the CLEAN deconvolution process, enabling
    flexible switching between different algorithms through a single interface.

    Parameters
    ----------
    mode : str
        Name of the CLEAN algorithm to use. Supported options are:
        - "clark"   : Classical Clark CLEAN (point-source subtraction).
        - "sinc"    : Sinc-shaped point-source CLEAN using continuous kernels.
        - "cluster" : Gaussian-fit and cluster-based CLEAN.
        - "multi"   : Multi-peak parallel CLEAN detecting several peaks per iteration.
        - "spectral": Channel-wise spectral CLEAN for 3D image cubes.
    psf : np.ndarray
        Point spread function (PSF) of the observation.
    mask : np.ndarray or bool array
        Boolean mask defining valid cleaning regions.
    config : dict
        Configuration dictionary containing algorithm parameters such as
        threshold, gain, iteration limits, and debug settings.

    Returns
    -------
    strategy : CleanStrategy
        Instantiated CLEAN strategy object implementing the selected algorithm.

    Raises
    ------
    ValueError
        If the provided mode does not correspond to a supported CLEAN strategy.

    Notes
    -----
    This function allows for modular extension of the CLEAN framework:
    simply define a new subclass of `CleanStrategy` and register it here
    to make it selectable through the main `clean()` or `clean_spectral()` functions.
    """

    if mode == "clark":
        return RegularClean(psf, mask, config)
    elif mode == "sinc":
        return PointSourceClean(psf, mask, config)
    elif mode == "cluster":
        return ClusterClean(psf, mask, config)
    elif mode == "multi":
        return MultiPeakClean(psf, mask, config)
    elif mode == "spectral":
        return SpectralClean(psf, mask, config)
    else:
        raise ValueError(f"Unknown cleaning mode: {mode}")

