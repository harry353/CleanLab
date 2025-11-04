import numpy as np
from scipy.signal import correlate
from astropy.io import fits
import matplotlib.pyplot as plt
import concurrent.futures
import time


def gaussian_2d(size, x0, y0, sigma_x, sigma_y, amplitude=1.0):
    """
    Generate a 2D Gaussian distribution.

    Creates a 2D Gaussian array centered at (x0, y0) with specified
    standard deviations along each axis and peak amplitude.

    Parameters
    ----------
    size : int
        Size of the square output array in pixels.
    x0, y0 : float
        Center coordinates of the Gaussian (in pixels).
    sigma_x, sigma_y : float
        Standard deviations of the Gaussian along x and y axes, respectively.
    amplitude : float, optional
        Peak amplitude of the Gaussian (default: 1.0).

    Returns
    -------
    g : np.ndarray
        2D array of shape (size, size) representing the Gaussian intensity profile.

    Notes
    -----
    This function forms the basis for simulating diffuse emission components
    in synthetic sky models used for CLEAN testing.
    """
    y, x = np.indices((size, size))
    g = amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                             ((y - y0) ** 2) / (2 * sigma_y ** 2)))
    return g


def compute_gaussian_chunk(chunk_args):
    """
    Generate a partial image of Gaussian components for one cube channel.

    This helper function computes the summed contribution of several
    Gaussian sources, used for parallel generation of large model cubes.

    Parameters
    ----------
    chunk_args : tuple
        Tuple containing:
        - coords_chunk : np.ndarray
            Array of (x, y) positions of sources.
        - amps_chunk : np.ndarray
            Corresponding amplitudes of sources.
        - sigmas_chunk : np.ndarray
            Standard deviations for each Gaussian [(σx, σy) per source].
        - image_size : int
            Size of the output image in pixels.

    Returns
    -------
    contrib : np.ndarray
        2D array representing the summed contribution from all sources in the chunk.
    """
    coords_chunk, amps_chunk, sigmas_chunk, image_size = chunk_args
    contrib = np.zeros((image_size, image_size))
    for (x0, y0), amp, (sx, sy) in zip(coords_chunk, amps_chunk, sigmas_chunk):
        contrib += gaussian_2d(image_size, x0, y0, sx, sy, amplitude=amp)
    return contrib


def generate_dynamic_diffuse_cube(n_sources, image_size, n_channels=10, num_workers=4):
    """
    Generate a synthetic 3D spectral cube of diffuse Gaussian emission.

    Creates a dynamic cube of shape (nchan, ny, nx) containing multiple
    diffuse Gaussian sources whose brightness, width, and position vary
    with frequency. This can be used as a realistic test dataset for
    multi-channel CLEAN algorithms and deconvolution benchmarking.

    Parameters
    ----------
    n_sources : int
        Number of Gaussian sources to generate in each channel.
    image_size : int
        Size of each image plane (in pixels).
    n_channels : int, optional
        Number of spectral channels to simulate (default: 10).
    num_workers : int, optional
        Number of parallel processes used for generating Gaussian chunks (default: 4).

    Returns
    -------
    cube : np.ndarray
        Synthetic 3D image cube of shape (n_channels, image_size, image_size).

    Notes
    -----
    - Each channel introduces slight variations in amplitude, position,
      and width to emulate realistic diffuse spectral structure.
    - Parallel processing is used to speed up Gaussian summation.
    - The resulting cube can be convolved with a PSF to produce
      a corresponding dirty cube.
    """
    base_coords = np.random.uniform(0, image_size, (n_sources, 2))
    base_amps = np.random.uniform(0.5, 1.0, n_sources)
    base_sigmas = np.random.uniform(4, 12, (n_sources, 2))

    cube = np.zeros((n_channels, image_size, image_size))
    freqs = np.linspace(-1, 1, n_channels)

    for ci, f in enumerate(freqs):
        # Vary amplitude and width with frequency
        amps = base_amps * (1 + 0.8 * np.sin(2 * np.pi * f + np.random.uniform(0, np.pi)))
        sigmas = base_sigmas * (1 + 0.3 * np.random.randn(*base_sigmas.shape))

        # Add small random shifts per channel
        coords = base_coords + np.random.uniform(-1.5, 1.5, base_coords.shape)
        coords = np.clip(coords, 0, image_size - 1)

        # Split sources into parallel chunks
        chunks = []
        chunk_size = n_sources // num_workers
        for i in range(num_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_workers - 1 else n_sources
            chunks.append((coords[start:end], amps[start:end],
                           sigmas[start:end], image_size))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(compute_gaussian_chunk, chunks))

        cube[ci] = np.sum(results, axis=0)

    return cube


if __name__ == "__main__":
    """
    Example script for generating a synthetic diffuse spectral cube.

    Produces a dynamic model cube of Gaussian emission and its corresponding
    dirty cube by convolution with a PSF. Both cubes are saved as FITS files
    and displayed for visual inspection.
    """
    ps_n = 10
    image_size = 256
    n_channels = 16

    start_time = time.perf_counter()
    cube = -1 * generate_dynamic_diffuse_cube(ps_n, image_size, n_channels=n_channels, num_workers=4)
    end_time = time.perf_counter()
    print(f"Synthetic dynamic diffuse cube generated in {end_time - start_time:.2f} s")
    print(f"Cube shape: {cube.shape}")

    # --- Save model cube ---
    model_path = "images/synthetic_model_cube.fits"
    dirty_path = "images/synthetic_dirty_cube.fits"

    hdu_model = fits.PrimaryHDU(cube.astype("float32"))
    hdr = hdu_model.header
    hdr["CTYPE3"] = "FREQ"
    hdr["CUNIT3"] = "GHz"
    hdr["CRVAL3"] = 100.0
    hdr["CDELT3"] = 0.05
    hdr["CRPIX3"] = 1
    hdu_model.writeto(model_path, overwrite=True)
    print(f"Saved dynamic model cube to {model_path}")

    # --- Load PSF ---
    psf = fits.getdata("./wsclean-psf.fits", ext=0)
    if psf.ndim == 4:
        psf = psf[-1, -1]
    elif psf.ndim == 3:
        psf = psf[-1]
    psf = psf.astype("float64")
    psf /= np.sum(psf)
    print(f"Loaded PSF with shape {psf.shape}")

    # --- Convolve cube with PSF to create dirty cube ---
    dirty_cube = np.zeros_like(cube)
    for i in range(cube.shape[0]):
        dirty_cube[i] = correlate(cube[i], psf, mode="same")

    # --- Save dirty cube ---
    hdu_dirty = fits.PrimaryHDU(dirty_cube.astype("float32"), header=hdr)
    hdu_dirty.writeto(dirty_path, overwrite=True)
    print(f"Saved dynamic dirty cube to {dirty_path}")

    # --- Quick visual check ---
    mid = cube.shape[0] // 2
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cube[mid], origin="lower", cmap="inferno")
    plt.title("Model (Dynamic Diffuse Sky)")
    plt.subplot(1, 2, 2)
    plt.imshow(dirty_cube[mid], origin="lower", cmap="inferno")
    plt.title("Dirty (Convolved with PSF)")
    plt.tight_layout()
    plt.show()
