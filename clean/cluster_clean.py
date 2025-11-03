import matplotlib.pyplot as plt
import numpy as np
from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy
from clean_utils import perform_fit
from clean_utils.cluster import cluster_to_first_ps


class ClusterClean(CleanStrategy):
    def __init__(self, psf, mask, config):
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=101)
        self.ps_coords = []            # clustered (output) positions
        self.fit_coords = []           # fitted (pre-cluster) positions
        self.debug_results = config.get("debug_results", False)
        self.initial_dirty = None      # for visualization later

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        if self.initial_dirty is None:
            # Store the initial dirty image only once
            self.initial_dirty = residual_image.copy()

        fit_image_size = 10
        popt = perform_fit.gauss(residual_image, my, mx, fit_image_size)
        true_y = my - fit_image_size + popt[2]
        true_x = mx - fit_image_size + popt[1]

        # store fitted point
        self.fit_coords.append((true_y, true_x))

        # cluster point
        cl_y, cl_x = cluster_to_first_ps(true_y, true_x, self.ps_coords)
        self.ps_coords.append((cl_y, cl_x))

        # update clean/residual images
        clean_image = ac.point_source(peak_flux, cl_y, cl_x, my, mx, gain, clean_image)
        residual_image = ur.point_source(peak_flux, cl_y, cl_x, my, mx, gain, residual_image, self.trunc_psf)

        return clean_image, residual_image, False

    def finalize(self):
        """Called at end of CLEAN if debug mode is on."""
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
