from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy
from clean_utils import perform_fit

class PointSourceClean(CleanStrategy):
    def __init__(self, psf, mask, config):
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=1023)

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        fit_image_size = 10
        popt = perform_fit.gauss(residual_image, my, mx, fit_image_size)
        y_min = max(0, my - fit_image_size)
        x_min = max(0, mx - fit_image_size)
        true_y, true_x = y_min + popt[2], x_min + popt[1]

        clean_image = ac.point_source(peak_flux, true_y, true_x, my, mx, gain, clean_image)
        residual_image = ur.point_source(peak_flux, true_y, true_x, my, mx, gain, residual_image, self.trunc_psf)
        # residual_image = ur.full(residual_image, clean_image, self.trunc_psf)
        return clean_image, residual_image, False