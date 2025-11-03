import numpy as np
import time

def crop_and_add(comps, patch, y_min, y_max, x_min, x_max):
    comp_region = comps[y_min:y_max, x_min:x_max]
    h = min(comp_region.shape[0], patch.shape[0])
    w = min(comp_region.shape[1], patch.shape[1])
    comps[y_min:y_min+h, x_min:x_min+w] += patch[:h, :w]
    return comps


def pad_and_add(comps, patch, y_min, y_max, x_min, x_max):
    target_shape = (y_max - y_min, x_max - x_min)
    pad_y = target_shape[0] - patch.shape[0]
    pad_x = target_shape[1] - patch.shape[1]
    if pad_y > 0 or pad_x > 0:
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode='constant')
    comps[y_min:y_max, x_min:x_max] += patch
    return comps

# Setup
np.random.seed(0)
image_shape = (256, 256)
comps_crop = np.zeros(image_shape)
comps_pad = np.zeros(image_shape)

patch_shape = (59, 53)
patch = np.random.rand(*patch_shape)

# Simulate offset region
y_min, x_min = 100, 100
y_max, x_max = y_min + 59, x_min + 54  # intentional mismatch

N = 10000

# Time cropping
start_crop = time.time()
for _ in range(N):
    crop_and_add(np.copy(comps_crop), patch, y_min, y_max, x_min, x_max)
crop_time = time.time() - start_crop

# Time padding
start_pad = time.time()
for _ in range(N):
    pad_and_add(np.copy(comps_pad), patch, y_min, y_max, x_min, x_max)
pad_time = time.time() - start_pad

print(f"Cropping time for {N} iterations: {crop_time:.4f} sec")
print(f"Padding time for {N} iterations: {pad_time:.4f} sec")
