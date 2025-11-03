import numpy as np
import matplotlib.pyplot as plt

def bg_subtr(image, output_mask_path="auto_mask.npy"):
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=image.min(), vmax=image.max())
    ax.set_title("Draw a rectangle over background region (press 'q' to finish)")

    rect_artist = None
    start_point = None
    coords = None

    def on_press(event):
        nonlocal start_point, rect_artist
        if event.inaxes != ax:
            return
        start_point = (event.xdata, event.ydata)
        rect_artist = plt.Rectangle(start_point, 0, 0, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect_artist)
        fig.canvas.draw()

    def on_motion(event):
        nonlocal rect_artist
        if start_point is None or event.inaxes != ax:
            return
        dx = event.xdata - start_point[0]
        dy = event.ydata - start_point[1]
        rect_artist.set_width(dx)
        rect_artist.set_height(dy)
        fig.canvas.draw()

    def on_release(event):
        nonlocal coords
        if start_point is None or event.inaxes != ax:
            return
        x0, y0 = start_point
        x1, y1 = event.xdata, event.ydata
        coords = (int(min(y0, y1)), int(max(y0, y1)), int(min(x0, x1)), int(max(x0, x1)))
        plt.close(fig)

    def on_key(event):
        if event.key.lower() == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if coords is None:
        raise RuntimeError("No background region selected.")

    y1, y2, x1, x2 = coords
    background_avg = np.mean(image[y1:y2, x1:x2])
    mask = image >= 5 * background_avg
    masked_image = np.where(mask, image, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, vmin=image.min(), vmax=image.max())
    axes[0].set_title("Full PS Image")
    axes[0].axis('off')
    axes[1].imshow(masked_image, vmin=image.min(), vmax=image.max())
    axes[1].set_title("Masked Image")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    np.save(output_mask_path, mask)


def manual(image, output_mask_path="manual_mask.npy"):
    circles = []
    current_center = None
    current_circle_artist = None

    fig, ax = plt.subplots()
    ax.imshow(image, vmin=image.min(), vmax=image.max())
    ax.set_title("Select circles to KEEP (Press 'q' to finish)")

    def on_press(event):
        nonlocal current_center, current_circle_artist
        if event.inaxes != ax:
            return
        current_center = (event.xdata, event.ydata)
        current_circle_artist = plt.Circle(current_center, 0, color='red', fill=False, linewidth=2)
        ax.add_patch(current_circle_artist)
        fig.canvas.draw()

    def on_motion(event):
        nonlocal current_center, current_circle_artist
        if current_center is None or event.inaxes != ax:
            return
        radius = np.hypot(event.xdata - current_center[0], event.ydata - current_center[1])
        current_circle_artist.set_radius(radius)
        fig.canvas.draw()

    def on_release(event):
        nonlocal current_center, current_circle_artist
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

    ny, nx = image.shape[:2]
    y_grid, x_grid = np.indices((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)

    for center, radius in circles:
        cx, cy = center
        mask |= ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= radius ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, vmin=image.min(), vmax=image.max())
    axes[0].set_title("Full PS Image")
    axes[0].axis("off")
    axes[1].imshow(mask * image, vmin=image.min(), vmax=image.max())
    axes[1].set_title("Masked Image")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    np.save(output_mask_path, mask)
