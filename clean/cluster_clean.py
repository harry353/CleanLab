import matplotlib.pyplot as plt
import numpy as np
from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy
from clean_utils import perform_fit
from clean_utils.cluster import cluster_to_first_ps


class ClusterClean(CleanStrategy):
    """
    CLEAN variant that fits and clusters detected point sources.

    This strategy first fits a 2D Gaussian to refine the subpixel position
    of each detected peak, then clusters nearby fitted positions to reduce
    redundant detections of the same source. The final cluster centers are used
    as component positions during the deconvolution process.

    Parameters
    ----------
    psf : np.ndarray
        The point spread function (PSF) of the observation.
    mask : np.ndarray or bool array
        Boolean mask defining valid regions for cleaning.
    config : dict
        Configuration dictionary containing parameters such as gain, threshold,
        and debug settings.
    """

    def __init__(self, psf, mask, config):
        """
        Initialize the ClusterClean strategy.

        Crops the PSF for faster convolution, initializes coordinate tracking
        lists, and stores debug flags.

        Parameters
        ----------
        psf : np.ndarray
            Input point spread function.
        mask : np.ndarray or bool array
            Boolean array defining valid cleaning regions.
        config : dict
            Configuration dictionary containing runtime parameters (e.g. gain, threshold).
        """
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=101)
        self.ps_coords = []            # clustered (output) positions
        self.fit_coords = []           # fitted (pre-cluster) positions
        self.debug_results = config.get("debug_results", False)
        self.initial_dirty = None      # for visualization later

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        """
        Perform a single CLEAN iteration using Gaussian fitting and clustering.

        Fits a 2D Gaussian around the current peak to determine its subpixel
        position, then clusters it with previous fitted points to avoid
        duplicate detections. Updates both the clean and residual images
        based on the clustered coordinates.

        Parameters
        ----------
        clean_image : np.ndarray
            Current accumulated clean components.
        residual_image : np.ndarray
            Current residual image.
        peak_flux : float
            Flux of the detected peak.
        my, mx : int
            Pixel coordinates of the detected peak (y, x).
        gain : float
            CLEAN gain controlling subtraction strength.

        Returns
        -------
        clean_image : np.ndarray
            Updated clean component image.
        residual_image : np.ndarray
            Updated residual image after subtraction.
        should_stop : bool
            Always False; iteration continues until threshold or max iterations reached.
        """
        if self.initial_dirty is None:
            self.initial_dirty = residual_image.copy()

        fit_image_size = 10
        popt = perform_fit.gauss(residual_image, my, mx, fit_image_size)
        true_y = my - fit_image_size + popt[2]
        true_x = mx - fit_image_size + popt[1]

        self.fit_coords.append((true_y, true_x))

        cl_y, cl_x = cluster_to_first_ps(true_y, true_x, self.ps_coords)
        self.ps_coords.append((cl_y, cl_x))

        clean_image = ac.point_source(peak_flux, cl_y, cl_x, my, mx, gain, clean_image)
        residual_image = ur.point_source(peak_flux, cl_y, cl_x, my, mx, gain, residual_image, self.trunc_psf)

        return clean_image, residual_image, False

    def finalize(self):
        """
        Display fitted and clustered point positions for debugging.

        Generates a plot showing fitted (orange) vs clustered (red) points
        over the initial dirty image, helping visualize the clustering
        behavior and verify position refinement.

        Only runs if `debug_results` is True.

        Returns
        -------
        None
        """
        if not self.debug_results or self.initial_dirty is None:
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.initial_dirty, origin='lower', cmap='viridis')
        ax.set_title("Fitted vs Clustered Points on Initial Dirty Image")

        if self.fit_coords:
            fy, fx = np.array(self.fit_coords).T
            ax.scatter(fx, fy, s=40, color='orange', label='Fitted points', marker='x')

        if self.ps_coords:
            cy, cx = np.array(self.ps_coords).T
            ax.scatter(cx, cy, s=40, color='red', label='Clustered points', marker='o', facecolors='none')

        ax.legend()
        plt.tight_layout()
        plt.show()
