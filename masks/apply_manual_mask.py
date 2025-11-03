import numpy as np
import matplotlib.pyplot as plt

ps_image = np.load("ps_image.npy")

circles = []  # List to store circles as (center, radius)
current_center = None
current_circle_artist = None

fig, ax = plt.subplots()
ax.imshow(ps_image, vmin=ps_image.min(), vmax=ps_image.max())
ax.set_title("Select circles to KEEP (Press 'q' to finish)")

def on_press(event):
    global current_center, current_circle_artist
    if event.inaxes != ax:
        return
    current_center = (event.xdata, event.ydata)
    current_circle_artist = plt.Circle(current_center, 0, color='red', fill=False, linewidth=2)
    ax.add_patch(current_circle_artist)
    fig.canvas.draw()

def on_motion(event):
    global current_center, current_circle_artist
    if current_center is None or event.inaxes != ax:
        return
    radius = np.hypot(event.xdata - current_center[0], event.ydata - current_center[1])
    current_circle_artist.set_radius(radius)
    fig.canvas.draw()

def on_release(event):
    global current_center, current_circle_artist, circles
    if current_center is None or event.inaxes != ax:
        return
    radius = np.hypot(event.xdata - current_center[0], event.ydata - current_center[1])
    circles.append((current_center, radius))
    current_center = None
    current_circle_artist = None
    fig.canvas.draw()

def on_key(event):
    if event.key.lower() == 'q':
        plt.close(fig)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

ny, nx = ps_image.shape[:2]
y_grid, x_grid = np.indices((ny, nx))
mask = np.zeros((ny, nx), dtype=bool)

for center, radius in circles:
    cx, cy = center  # x, y coordinates from the mouse events
    mask |= ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= radius ** 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(ps_image, vmin=ps_image.min(), vmax=ps_image.max())
axes[0].set_title("Full PS Image")
axes[0].axis("off")
axes[1].imshow(mask * ps_image, vmin=ps_image.min(), vmax=ps_image.max())
axes[1].set_title("Masked Image")
axes[1].axis("off")
plt.tight_layout()
plt.show()

np.save("manual_mask.npy", mask)
