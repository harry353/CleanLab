import numpy as np
from scipy.signal.windows import hann
from .utils import adjust_bounds
from .func_kernels import sinc_kernel
import matplotlib.pyplot as plt

def point_source(peak_flux, my_corr, mx_corr, my, mx, gain, comps, sinc_kernel_size=3):

    offset_y, offset_x = my_corr - my, mx_corr - mx

    sinc_comp = sinc_kernel(sinc_kernel_size, offset_y, offset_x)

    hann_window = np.outer(hann(sinc_kernel_size), hann(sinc_kernel_size))
    windowed_sinc = sinc_comp * hann_window
    windowed_sinc /= np.sum(windowed_sinc)
    windowed_sinc *= peak_flux * gain

    (y_min, y_max, x_min, x_max,
     kernel_y_start, kernel_y_end, kernel_x_start, kernel_x_end) = adjust_bounds(
         center=(my, mx),
         patch_size=(sinc_kernel_size, sinc_kernel_size),
         image_shape=comps.shape
     )

    comps[y_min:y_max, x_min:x_max] += windowed_sinc[
        kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end
    ]

    return comps

def delta(peak_flux, gain, comps, y, x):
    comp_val = peak_flux * gain 
    comps[y, x] += comp_val
    return comps

def main():
    comps = np.zeros((100, 100))
    mx_corr, my_corr = 60.5, 60.5
    mx, my = 60, 60
    peak_flux, gain = 1, 1

    comps = point_source(peak_flux, my_corr, mx_corr, my, mx, gain, comps)

    plt.imshow(comps, origin='lower')
    plt.scatter(mx, my, color='r', marker='+', label='Peak')
    plt.scatter(mx_corr, my_corr, color='g', marker='+', label='Corrected Peak')
    plt.legend()
    # plt.xlim(55, 65)
    # plt.ylim(55, 65)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
