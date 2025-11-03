import numpy as np
from matplotlib import pyplot as plt

def sinc_kernel(size, offset_y=0.0, offset_x=0.0):
    f = np.sinc

    # (size - 1)/2 centers the coordinate grid on the middle pixel
    center = (size - 1) / 2
    y_coords = np.arange(size) - center - offset_y
    x_coords = np.arange(size) - center - offset_x
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

    return f(X) * f(Y)

def main():
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
