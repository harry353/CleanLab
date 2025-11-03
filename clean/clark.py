from clean_utils import utils
from clean_utils import update_residual as ur
from clean_utils import add_component as ac
from .strategy_base import CleanStrategy

class RegularClean(CleanStrategy):
    def __init__(self, psf, mask, config):
        super().__init__(psf, mask, config)
        self.trunc_psf = utils.crop_psf(psf, size=101)

    def step(self, clean_image, residual_image, peak_flux, my, mx, gain):
        clean_image = ac.delta(peak_flux, gain, clean_image, my, mx)
        residual_image = ur.delta(mx, my, gain, peak_flux, residual_image, self.trunc_psf)

        return clean_image, residual_image, False