import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from astropy.io import fits
import concurrent.futures
import time


def sinc_kernel(size, offset_x=0, offset_y=0):
    """
    Generate a 2D sinc kernel with optional subpixel offsets.

    Constructs a separable 2D sinc kernel centered on the image,
    optionally shifted by subpixel offsets in the x and y directions.
    This kernel is used to model ideal point-source responses.

    Parameters
    ----------
    size : int
        Size of the output square kernel in pixels.
    offset_x, offset_y : float, optional
        Subpixel offsets applied to the x and y axes (default: 0).

    Returns
    -------
    kernel : np.ndarray
        3D array of shape (1, size, size) containing the sinc kernel.

    Notes
    -----
    - The sinc function is defined as :math:`\\text{sinc}(x) = \\sin(\\pi x) / (\\pi x)`.
    - Offsets simulate subpixel source positions for more realistic
      point-source distributions.
    """
    f = np.sinc
    total_offset_x = (size + 1) // 2 + np.atleast_1d(offset_x)[:, None]
    total_offset_y = (size + 1) // 2 + np.atleast_1d(offset_y)[:, None]
    x_coords = np.arange(size)[None, :] - total_offset_y
    y_coords = np.arange(size)[None, :] - total_offset_x
    kernel = f(x_coords)[:, :, None] * f(y_coords)[:, None, :]
    return kernel


def compute_ps_contribution(ps_coord, intensity, image_size):
    """
    Compute the image contribution from a single point source.

    Generates a subpixel-shifted sinc kernel centered on the given
    coordinates, scaled by the source intensity, and places it
    into the full image frame.

    Parameters
    ----------
    ps_coord : np.ndarray or tuple of float
        (x, y) coordinates of the point source (in pixels).
    intensity : float
        Source flux or amplitude.
    image_size : int
        Size of the full image in pixels.

    Returns
    -------
    contrib : np.ndarray
        2D array representing this point source’s contribution to the image.

    Notes
    -----
    - The sinc kernel ensures smooth interpolation between pixel centers.
    - Edge clipping is automatically handled to avoid array overflows.
    """
    ps_int = ps_coord.astype(int)
    ps_frac = ps_coord - ps_int
    kernel = sinc_kernel(image_size, offset_x=ps_frac[0], offset_y=ps_frac[1])[0]
    kcenter = image_size // 2
    y0 = ps_int[1] - kcenter
    x0 = ps_int[0] - kcenter
    contrib = np.zeros((image_size, image_size))
    y1, x1 = y0 + image_size, x0 + image_size
    bg_y0, bg_x0 = max(0, y0), max(0, x0)
    bg_y1 = min(image_size, y1)
    bg_x1 = min(image_size, x1)
    k_y0 = bg_y0 - y0
    k_x0 = bg_x0 - x0
    k_y1 = k_y0 + (bg_y1 - bg_y0)
    k_x1 = k_x0 + (bg_x1 - bg_x0)
    contrib[bg_y0:bg_y1, bg_x0:bg_x1] = intensity * kernel[k_y0:k_y1, k_x0:k_x1]
    return contrib


def compute_chunk_contributions(chunk_args):
    """
    Compute summed image contributions from a chunk of point sources.

    Helper function for parallelized image generation. Processes a subset
    of the total source list to accelerate large-scale simulations.

    Parameters
    ----------
    chunk_args : tuple
        (ps_coords_chunk, intensities_chunk, image_size)

    Returns
    -------
    contributions : np.ndarray
        2D array representing the total contribution from this chunk of sources.
    """
    ps_coords_chunk, intensities_chunk, image_size = chunk_args
    contributions = np.zeros((image_size, image_size))
    for ps_coord, intensity in zip(ps_coords_chunk, intensities_chunk):
        contributions += compute_ps_contribution(ps_coord, intensity, image_size)
    return contributions


def generate_ps_image(n, image_size, num_workers=4):
    """
    Generate a synthetic 2D point-source sky image.

    Creates an image composed of `n` randomly distributed point sources,
    each represented by a sinc-shaped kernel with subpixel accuracy.
    Parallelized across multiple processes for scalability.

    Parameters
    ----------
    n : int
        Number of point sources to generate.
    image_size : int
        Size of the square output image in pixels.
    num_workers : int, optional
        Number of parallel worker processes (default: 4).

    Returns
    -------
    ps_image : np.ndarray
        Synthetic dirty sky image (2D array) after PSF convolution.

    Notes
    -----
    - Uses sinc kernels for realistic subpixel interpolation.
    - The resulting image is convolved with the PSF to simulate
      a dirty image as seen by an interferometer.
    """
    ps_coords = np.random.uniform(0, 1, (n, 2)) * image_size
    intensities = np.random.uniform(0, 1, n)
    
    chunks = []
    chunk_size = n // num_workers
    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_workers - 1 else n
        chunks.append((ps_coords[start:end], intensities[start:end], image_size))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_chunk_contributions, chunks))
    
    background = np.sum(results, axis=0)
    kernel_center = sinc_kernel(image_size, offset_x=0.0, offset_y=0.0)[0]
    return correlate(background, kernel_center, mode='same')


# ---------------- MAIN ----------------
if __name__ == "__main__":
    """
    Example script to generate a synthetic dirty image.

    Randomly distributes point sources across an image, applies sinc
    interpolation for subpixel positioning, and convolves the result
    with a PSF to create a dirty image saved as a FITS file.
    """
    ps_n = 50
    image_size = 128

    start_time = time.perf_counter()
    sinc_sources = generate_ps_image(ps_n, image_size, num_workers=4)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    psf_path = "./wsclean-psf.fits"
    psf = fits.getdata(psf_path, ext=0)[-1][-1].astype("f8")

    ps_image = correlate(sinc_sources, psf, mode='same')

    # Reshape to match WSClean format (Stokes, Freq, Y, X)
    ps_image_4d = ps_image[np.newaxis, np.newaxis, :, :]

    # Save the result as FITS
    fits.writeto("ps.fits", ps_image_4d.astype("f8"), overwrite=True)

    plt.imshow(ps_image, vmin=ps_image.min(), vmax=ps_image.max())
    plt.title("Generated Synthetic Dirty Image")
    plt.show()

    print("Saved ps_image_dense.fits with shape (1, 1, 128, 128)")
