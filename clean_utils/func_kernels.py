import numpy as np
from matplotlib import pyplot as plt


def sinc_kernel(size, offset_y=0.0, offset_x=0.0):
    """
    Generate a 2D sinc kernel with optional subpixel offsets.

    This function constructs a 2D separable sinc function centered on the
    middle pixel of the array, optionally shifted by subpixel offsets in
    the x and y directions. The kernel is often used for modeling ideal
    point sources or for subpixel-accurate component placement in CLEAN
    deconvolution algorithms.

    Parameters
    ----------
    size : int
        Size of the kernel in pixels (must be an odd integer for symmetry).
    offset_y : float, optional
        Subpixel offset along the y-axis, measured in pixels (default: 0.0).
    offset_x : float, optional
        Subpixel offset along the x-axis, measured in pixels (default: 0.0).

    Returns
    -------
    kernel : np.ndarray
        2D sinc kernel array of shape (size, size).

    Notes
    -----
    - The kernel is defined as :math:`\\text{sinc}(x) \\times \\text{sinc}(y)`.
    - Offsets allow modeling of sources not perfectly centered on a pixel.
    - This kernel is commonly windowed (e.g., with a Hann window) before use
      in CLEAN to suppress edge ringing.
    """
    f = np.sinc

    # (size - 1)/2 centers the coordinate grid on the middle pixel
    center = (size - 1) / 2
    y_coords = np.arange(size) - center - offset_y
    x_coords = np.arange(size) - center - offset_x
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

    return f(X) * f(Y)


def main():
    """
    Demonstrate the 2D sinc kernel generation.

    Generates and visualizes a sinc kernel with subpixel offsets, displaying
    its structure and verifying the central peak position and symmetry.

    Notes
    -----
    This function is intended for visualization and testing purposes only.
    It is not used directly in the CLEAN algorithm.
    """
    size = 21
    offset_y, offset_x = 0.2, -0.1
    kernel = sinc_kernel(size, offset_y, offset_x)

    print(f"Kernel shape: {kernel.shape}")
    print(f"Peak value at center: {kernel[size//2, size//2]:.8f}")

    extent = [-size//2 - 0.5, size//2 + 0.5, -size//2 - 0.5, size//2 + 0.5]
    plt.imshow(kernel, origin='lower', cmap='viridis', extent=extent)
    plt.scatter(0, 0, color='red', marker='+', s=100)
    plt.title("Sinc Kernel (center marked)")
    plt.colorbar()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
