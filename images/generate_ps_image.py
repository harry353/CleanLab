import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from astropy.io import fits
import concurrent.futures
import time

def sinc_kernel(size, offset_x=0, offset_y=0):
    f = np.sinc
    total_offset_x = (size + 1) // 2 + np.atleast_1d(offset_x)[:, None]
    total_offset_y = (size + 1) // 2 + np.atleast_1d(offset_y)[:, None]
    x_coords = np.arange(size)[None, :] - total_offset_y
    y_coords = np.arange(size)[None, :] - total_offset_x
    kernel = f(x_coords)[:, :, None] * f(y_coords)[:, None, :]
    return kernel

def compute_ps_contribution(ps_coord, intensity, image_size):
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
    ps_coords_chunk, intensities_chunk, image_size = chunk_args
    contributions = np.zeros((image_size, image_size))
    for ps_coord, intensity in zip(ps_coords_chunk, intensities_chunk):
        contributions += compute_ps_contribution(ps_coord, intensity, image_size)
    return contributions

def generate_ps_image(n, image_size, num_workers=4):
    ps_coords = np.random.uniform(0, 1, (n, 2)) * image_size
    intensities = np.random.uniform(0, 1, n)
    
    # Split indices into chunks
    chunks = []
    chunk_size = n // num_workers
    for i in range(num_workers):
        start = i * chunk_size
        end = (i+1) * chunk_size if i < num_workers - 1 else n
        chunks.append((ps_coords[start:end], intensities[start:end], image_size))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_chunk_contributions, chunks))
    background = np.sum(results, axis=0)
    kernel_center = sinc_kernel(image_size, offset_x=0.0, offset_y=0.0)[0]
    return correlate(background, kernel_center, mode='same')

ps_n = 50
image_size = 128

start_time = time.perf_counter()
sinc_sources = generate_ps_image(ps_n, image_size, num_workers=4)
end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.4f} seconds")

psf_path = "./wsclean-psf.fits"
psf = fits.getdata(psf_path, ext=0)[-1][-1]
psf = psf.astype("f8")

ps_image = correlate(sinc_sources, psf, mode='same')
plt.imshow(ps_image, vmin=ps_image.min(), vmax=ps_image.max())
plt.show()
np.save("ps_image_dense.npy", ps_image)
