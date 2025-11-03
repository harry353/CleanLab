import numpy as np
import matplotlib.pyplot as plt

ps_image = np.load("ps_image.npy")

x1, x2 = 32, 64
y1, y2 = 32, 64
empty_region_average = np.mean(ps_image[x1:x2, y1:y2])
mask = ps_image >= 5 * empty_region_average
masked_image = np.where(mask, ps_image, 0)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(ps_image, vmin=ps_image.min(), vmax=ps_image.max())
axes[0].set_title("Full PS Image")
axes[0].axis('off')
axes[1].imshow(masked_image, vmin=ps_image.min(), vmax=ps_image.max())
axes[1].set_title("Masked Image")
axes[1].axis('off')
plt.tight_layout()
plt.show()

np.save("auto_mask.npy", mask)
