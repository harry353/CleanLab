import numpy as np


def step(x):
    """
    Piecewise step gain function for CLEAN iterations.

    Returns a discrete gain value based on the current signal strength.
    This simple heuristic can be used to adjust the CLEAN gain dynamically,
    increasing convergence speed for strong signals and stabilizing it for
    weaker ones.

    Parameters
    ----------
    x : float
        Signal strength metric, typically derived from the ratio of peak flux
        to residual noise.

    Returns
    -------
    gain : float
        Gain value to be applied during the CLEAN iteration.

    Notes
    -----
    The returned gain follows three regimes:
    - `gain = 0.8` for strong signals (`x > 10`)
    - `gain = 0.5` for moderate signals (`x > 5`)
    - `gain = 0.1` for weak signals (`x ≤ 5`)
    """
    if x > 10:
        gain = 0.8
    elif x > 5:
        gain = 0.5
    else:
        gain = 0.1
    return gain


def scaled_logistic(x, shift=5, min_val=0.1, max_val=0.6):
    """
    Scaled logistic gain function for adaptive CLEAN convergence.

    Implements a smooth logistic curve to continuously scale the gain
    between `min_val` and `max_val` depending on signal strength.
    This formulation provides a more gradual adjustment compared to
    the discrete `step()` function, reducing oscillations during
    the CLEAN iteration process.

    Parameters
    ----------
    x : float
        Signal strength metric (e.g., peak-to-noise ratio).
    shift : float, optional
        Midpoint of the logistic curve (default: 5).
    min_val : float, optional
        Minimum gain value at low signal levels (default: 0.1).
    max_val : float, optional
        Maximum gain value at high signal levels (default: 0.6).

    Returns
    -------
    gain : float
        Computed gain value between `min_val` and `max_val`.

    Notes
    -----
    The logistic function is defined as:

    .. math::

        g(x) = v_{min} + (v_{max} - v_{min}) \times
               \frac{1}{1 + e^{-(x - s)}}

    where :math:`s` is the shift, controlling where the curve transitions.
    """
    base = 1 / (1 + np.exp(-(x - shift)))
    return min_val + (max_val - min_val) * base
