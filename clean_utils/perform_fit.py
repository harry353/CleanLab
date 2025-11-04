import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate
from matplotlib.patches import Ellipse


def twoD_gaussian(xy, amplitude, X0, Y0, sigma_x, sigma_y, theta, offset):
    """
    Two-dimensional Gaussian model function.

    Computes the value of a rotated 2D Gaussian surface given coordinate
    grids and parameter values. This analytic form is used for fitting
    compact sources in residual images during CLEAN deconvolution.

    Parameters
    ----------
    xy : tuple of np.ndarray
        Tuple (x, y) of meshgrid arrays defining pixel coordinates.
    amplitude : float
        Peak amplitude of the Gaussian.
    X0, Y0 : float
        Center coordinates of the Gaussian.
    sigma_x, sigma_y : float
        Standard deviations of the Gaussian along the x and y axes.
    theta : float
        Rotation angle of the Gaussian in radians.
    offset : float
        Constant background offset.

    Returns
    -------
    g : np.ndarray
        Flattened array of model values matching the shape of `x` and `y`.

    Notes
    -----
    The rotated Gaussian is defined as:

    .. math::

        G(x, y) = A \exp\left(-\left[a(x-X_0)^2 + 2b(x-X_0)(y-Y_0)
        + c(y-Y_0)^2\right]\right) + O

    where :math:`a`, :math:`b`, and :math:`c` depend on the rotation angle.
    """
    x, y = xy
    X0 = float(X0)
    Y0 = float(Y0)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * (x - X0) ** 2 + 2 * b * (x - X0) * (y - Y0) + c * (y - Y0) ** 2))
    return g.ravel()


def gauss(residual, my, mx, fit_image_size=21, true_params=None, debug=False):
    """
    Fit a 2D Gaussian to a residual image patch around a detected peak.

    Extracts a square region centered on a detected peak and performs a
    non-linear least-squares fit to a rotated 2D Gaussian model. The
    fitted parameters can then be used for subpixel centroid refinement
    of point sources during CLEAN.

    Parameters
    ----------
    residual : np.ndarray
        2D residual image containing the source peak.
    my, mx : int
        Pixel coordinates (y, x) of the detected peak.
    fit_image_size : int, optional
        Half-size of the fitting window in pixels (default: 21).
    true_params : tuple, optional
        True Gaussian parameters for comparison when debugging.
        Expected order: (amplitude, X0, Y0, sigma_x, sigma_y, theta, offset).
    debug : bool, optional
        If True, displays contour plots, fit diagnostics, and parameter
        comparison tables (default: False).

    Returns
    -------
    popt : np.ndarray
        Array of best-fit Gaussian parameters:
        [amplitude, X0, Y0, sigma_x, sigma_y, theta, offset].

    Notes
    -----
    - Fitting is performed using SciPy’s :func:`curve_fit`.
    - Parameter bounds enforce positive amplitudes and valid rotations.
    - If `debug=True`, the function plots 1σ–3σ ellipses and reports
      fit accuracy relative to provided `true_params`.
    """
    start_time = time.time()
    
    y_min = max(0, my - fit_image_size)
    y_max = min(residual.shape[0], my + fit_image_size + 1)
    x_min = max(0, mx - fit_image_size)
    x_max = min(residual.shape[1], mx + fit_image_size + 1)

    ps_patch = residual[y_min:y_max, x_min:x_max]
    x = np.arange(ps_patch.shape[1])
    y = np.arange(ps_patch.shape[0])
    x, y = np.meshgrid(x, y)
    
    initial_guess = (np.max(ps_patch), ps_patch.shape[1] // 2, ps_patch.shape[0] // 2, 1, 1, 0, np.min(ps_patch))
    popt, _ = curve_fit(
        twoD_gaussian, (x, y), ps_patch.ravel(), p0=initial_guess, maxfev=10000,
        bounds=([0, 0, 0, 0, 0, -np.pi, -np.inf],
                [np.inf, ps_patch.shape[1], ps_patch.shape[0], np.inf, np.inf, np.pi, np.inf])
    )

    popt[5] = popt[5] % np.pi
    fitted_ps_patch = twoD_gaussian((x, y), *popt).reshape(ps_patch.shape)
    elapsed = time.time() - start_time

    x0, y0 = popt[1], popt[2]
    sigma_x, sigma_y = popt[3], popt[4]
    theta = popt[5] + np.pi / 2

    if debug:
        # Plot 1σ, 2σ, 3σ ellipses
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(residual, origin='lower', cmap='viridis')
        ax.set_title("Fitted Gaussian Contours")
        for nsig, color in zip([1, 2, 3], ['red', 'orange', 'cyan']):
            ell = Ellipse(
                xy=(x0, y0),
                width=2 * nsig * sigma_x,
                height=2 * nsig * sigma_y,
                angle=np.degrees(theta),
                edgecolor=color,
                facecolor='none',
                linewidth=2,
                label=f'{nsig}σ'
            )
            ax.add_patch(ell)

        ax.plot(x0, y0, 'r+', markersize=12, label='Fit center')
        ax.plot(mx, my, 'c+', markersize=10, label='Peak')
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"Fitting time: {1000*elapsed:.6f} ms")
        if true_params is not None:
            tp = list(true_params)
            tp[5] = tp[5] % np.pi
            if tp[3] < tp[4]:
                tp[3], tp[4] = tp[4], tp[3]
                tp[5] = (tp[5] + np.pi / 2) % np.pi

            errors = np.abs((np.array(popt) - np.array(tp)) / np.array(tp)) * 100
            param_names = ["Amplitude", "X0", "Y0", "Sigma_x", "Sigma_y", "Theta (deg)", "Offset"]
            tp[5] = np.degrees(tp[5]) % 180
            popt[5] = np.degrees(popt[5]) % 180
            errors[5] = np.abs((popt[5] - tp[5]) / tp[5]) * 100
            table = [
                [name, f"{orig:.3f}", f"{fit:.3f}", f"{err:.2f}"]
                for name, orig, fit, err in zip(param_names, tp, popt, errors)
            ]
            print("\nFit Results:")
            print(tabulate(table, headers=["Parameter", "Original", "Fitted", "Error (%)"], tablefmt="github"))

    return popt


def generate_test_data(size=51):
    """
    Generate synthetic 2D Gaussian test data.

    Creates a noisy 2D Gaussian field with random parameters for testing
    the fitting performance of :func:`gauss`.

    Parameters
    ----------
    size : int, optional
        Size of the generated image in pixels (default: 51).

    Returns
    -------
    gaussian : np.ndarray
        Simulated 2D Gaussian image with added Gaussian noise.
    true_params : tuple
        True Gaussian parameters used to generate the data:
        (amplitude, X0, Y0, sigma_x, sigma_y, theta, offset).

    Notes
    -----
    - The orientation angle `theta` is set to −π/4 by default for reproducibility.
    - Random noise with σ = 0.1 is added to the final image.
    """
    amplitude = np.random.uniform(3, 10)
    X0 = np.random.uniform(size * 0.3, size * 0.7)
    Y0 = np.random.uniform(size * 0.3, size * 0.7)
    sigma_x = np.random.uniform(2, 10)
    sigma_y = np.random.uniform(2, 10)
    theta = -np.pi / 4
    offset = np.random.uniform(0, 2)

    x = np.arange(size)
    y = np.arange(size)
    x, y = np.meshgrid(x, y)

    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)

    gaussian = offset + amplitude * np.exp(-(a * (x - X0) ** 2 + 2 * b * (x - X0) * (y - Y0) + c * (y - Y0) ** 2))
    noise = np.random.normal(0, 0.1, gaussian.shape)
    return gaussian + noise, (amplitude, X0, Y0, sigma_x, sigma_y, theta, offset)


if __name__ == "__main__":
    data, true_params = generate_test_data()
    peak = np.max(data)
    my, mx = np.where(data == peak)
    my, mx = my[0], mx[0]
    gauss(residual=data, my=my, mx=mx, fit_image_size=101, true_params=true_params, debug=True)
