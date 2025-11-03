import numpy as np
from scipy.signal import correlate
from astropy.io import fits
import matplotlib.pyplot as plt
import concurrent.futures
import time


def gaussian_2d(size, x0, y0, sigma_x, sigma_y, amplitude=1.0):
    """Return a 2D Gaussian image."""
    y, x = np.indices((size, size))
    g = amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                             ((y - y0) ** 2) / (2 * sigma_y ** 2)))
    return g


def compute_gaussian_chunk(chunk_args):
    """Generate a chunk of diffuse Gaussian components for one channel."""
    coords_chunk, amps_chunk, sigmas_chunk, image_size = chunk_args
    contrib = np.zeros((image_size, image_size))
    for (x0, y0), amp, (sx, sy) in zip(coords_chunk, amps_chunk, sigmas_chunk):
        contrib += gaussian_2d(image_size, x0, y0, sx, sy, amplitude=amp)
    return contrib


def generate_dynamic_diffuse_cube(n_sources, image_size, n_channels=10, num_workers=4):
    """
    Generate a 3D spectral cube (nchan × ny × nx) of diffuse Gaussians
    that vary in amplitude, size, and position across channels.
    """
    # Base Gaussian properties
    base_coords = np.random.uniform(0, image_size, (n_sources, 2))
    base_amps = np.random.uniform(0.5, 1.0, n_sources)
    base_sigmas = np.random.uniform(4, 12, (n_sources, 2))

    cube = np.zeros((n_channels, image_size, image_size))
    freqs = np.linspace(-1, 1, n_channels)

    for ci, f in enumerate(freqs):
        # Vary amplitude and width with frequency
        amps = base_amps * (1 + 0.8 * np.sin(2 * np.pi * f + np.random.uniform(0, np.pi)))
        sigmas = base_sigmas * (1 + 0.3 * np.random.randn(*base_sigmas.shape))

        # Add small random shifts in position per channel
        coords = base_coords + np.random.uniform(-1.5, 1.5, base_coords.shape)
        coords = np.clip(coords, 0, image_size - 1)

        # Split sources into chunks for parallel processing
        chunks = []
        chunk_size = n_sources // num_workers
        for i in range(num_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_workers - 1 else n_sources
            chunks.append((coords[start:end], amps[start:end],
                           sigmas[start:end], image_size))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(compute_gaussian_chunk, chunks))

        channel_image = np.sum(results, axis=0)
        cube[ci] = channel_image

    return cube


# ---------------- MAIN ----------------
if __name__ == "__main__":
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

    # --- Load PSF from parent directory ---
    psf = fits.getdata("./wsclean-psf.fits", ext=0)
    if psf.ndim == 4:
        psf = psf[-1, -1]
    elif psf.ndim == 3:
        psf = psf[-1]
    psf = psf.astype("float64")
    psf /= np.sum(psf)
    print(f"Loaded PSF from ../wsclean-psf.fits with shape {psf.shape}")

    # --- Convolve each channel with PSF ---
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
