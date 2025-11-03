from .clark import RegularClean
from .sinc_clean import PointSourceClean
from .cluster_clean import ClusterClean
from .multi_peak_clean import MultiPeakClean
from .spectral_clean import SpectralClean

def get_clean_strategy(mode, psf, mask, config):
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

