import numpy as np
import matplotlib.pyplot as plt
import time
from astropy.modeling import models, fitting
from tabulate import tabulate
from matplotlib.patches import Ellipse


def gauss(residual, my, mx, fit_image_size=21, true_params=None, debug=False):
    """Fit a 2D Gaussian (no background term) to a local patch around (my, mx)."""
    start_time = time.time()

    # --- extract local patch ---
    y_min = max(0, my - fit_image_size)
    y_max = min(residual.shape[0], my + fit_image_size + 1)
    x_min = max(0, mx - fit_image_size)
    x_max = min(residual.shape[1], mx + fit_image_size + 1)
    ps_patch = residual[y_min:y_max, x_min:x_max]

    # --- coordinate grid ---
    y, x = np.mgrid[0:ps_patch.shape[0], 0:ps_patch.shape[1]]

    # --- initial guess ---
    amplitude = np.max(ps_patch)
    x0 = ps_patch.shape[1] / 2
    y0 = ps_patch.shape[0] / 2
    sigma_x = sigma_y = max(1, fit_image_size / 5)
    theta = 0.0

    # --- initialize Gaussian2D model (no offset/background) ---
    g_init = models.Gaussian2D(
        amplitude=amplitude,
        x_mean=x0,
        y_mean=y0,
        x_stddev=sigma_x,
        y_stddev=sigma_y,
        theta=theta,
    )

    # --- fit the model ---
    fitter = fitting.LevMarLSQFitter()
    g_fit = fitter(g_init, x, y, ps_patch)
    elapsed = time.time() - start_time

    # --- fitted parameters ---
    popt = (
        g_fit.amplitude.value,
        g_fit.x_mean.value,
        g_fit.y_mean.value,
        g_fit.x_stddev.value,
        g_fit.y_stddev.value,
        g_fit.theta.value,
    )

    # --- debug visualization ---
    if debug:
        fitted_ps_patch = g_fit(x, y)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(ps_patch, origin="lower", cmap="viridis")
        ax.set_title("Astropy Gaussian2D fit (no background)")

        x0, y0 = g_fit.x_mean.value, g_fit.y_mean.value
        sigma_x, sigma_y = g_fit.x_stddev.value, g_fit.y_stddev.value
        theta = g_fit.theta.value

        for nsig, color in zip([1, 2, 3], ["red", "orange", "cyan"]):
            ell = Ellipse(
                xy=(x0, y0),
                width=2 * nsig * sigma_x,
                height=2 * nsig * sigma_y,
                angle=np.degrees(theta),
                edgecolor=color,
                facecolor="none",
                linewidth=2,
                label=f"{nsig}σ",
            )
            ax.add_patch(ell)

        ax.plot(x0, y0, "r+", markersize=12, label="Fit center")
        local_max_y, local_max_x = np.unravel_index(np.argmax(ps_patch), ps_patch.shape)
        ax.plot(local_max_x, local_max_y, "c+", markersize=10, label="Brightest pixel")
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"Fitting time: {1000 * elapsed:.3f} ms")

        if true_params is not None:
            tp = np.array(true_params[:6], dtype=float)
            popt_arr = np.array(popt)
            param_names = ["Amplitude", "X0", "Y0", "Sigma_x", "Sigma_y", "Theta (rad)"]
            errors = np.abs((popt_arr - tp) / tp) * 100
            table = [
                [name, f"{t:.3f}", f"{p:.3f}", f"{e:.2f}"]
                for name, t, p, e in zip(param_names, tp, popt_arr, errors)
            ]
            print("\nFit Results:")
            print(tabulate(table, headers=["Parameter", "True", "Fitted", "Error (%)"], tablefmt="github"))

    return popt


def generate_test_data(size=51):
    """Generate synthetic 2D Gaussian test data without background offset."""
    rng = np.random.default_rng()
    amplitude = rng.uniform(3, 10)
    X0 = rng.uniform(size * 0.3, size * 0.7)
    Y0 = rng.uniform(size * 0.3, size * 0.7)
    sigma_x = rng.uniform(2, 10)
    sigma_y = rng.uniform(2, 10)
    theta = rng.uniform(0, np.pi)

    y, x = np.mgrid[0:size, 0:size]
    g_model = models.Gaussian2D(
        amplitude=amplitude,
        x_mean=X0,
        y_mean=Y0,
        x_stddev=sigma_x,
        y_stddev=sigma_y,
        theta=theta,
    )

    data = g_model(x, y) + rng.normal(0, 0.1, (size, size))
    return data, (amplitude, X0, Y0, sigma_x, sigma_y, theta)


if __name__ == "__main__":
    data, true_params = generate_test_data()
    peak_y, peak_x = np.unravel_index(np.argmax(data), data.shape)
    gauss(data, peak_y, peak_x, fit_image_size=21, true_params=true_params, debug=True)
