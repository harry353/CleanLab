class CleanStrategy:
    def __init__(self, psf, mask, config):
        self.psf = psf
        self.mask = mask
        self.config = config

    def step(self, clean_image, residual_image, dirty_image, iter, peak_flux, my, mx):
        raise NotImplementedError("Subclasses must implement the step() method")

