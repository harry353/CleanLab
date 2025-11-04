import numpy as np


def noise(res, thresh, peak):
    """
    Check whether the CLEAN process has reached the noise threshold.

    Computes the current signal-to-noise ratio (SNR) of the brightest
    residual peak relative to the image RMS and determines whether
    it falls below the user-defined threshold. If so, the function
    signals that CLEAN should stop.

    Parameters
    ----------
    res : np.ndarray
        Current residual image.
    thresh : float
        Noise stopping threshold. If the peak SNR drops below this
        value, the function returns True.
    peak : float or array-like
        Detected peak flux (or array of peaks in multi-peak mode).

    Returns
    -------
    bool
        True if the residual image has reached the noise stopping
        criterion, False otherwise.

    Notes
    -----
    - In multi-peak modes, the function automatically uses the
      strongest detected peak.
    - The function prints the current SNR when the threshold is reached.
    """
    res_std = np.std(res)

    # Handle multi-peak input (take the strongest one)
    if isinstance(peak, (list, tuple, np.ndarray)):
        peak = np.max(peak)

    noise_sigma = peak / res_std
    if noise_sigma < thresh:
        print(f"Reached noise stopping threshold: {noise_sigma:.4f} < {thresh}")
        return True


def iterations(res, mask, max_iter):
    """
    Report when the maximum number of CLEAN iterations has been reached.

    Computes the signal-to-noise ratio (SNR) at the last iteration
    and prints a summary message including the final noise level.

    Parameters
    ----------
    res : np.ndarray
        Final residual image.
    mask : np.ndarray or bool array
        Boolean mask defining valid regions in the residual image.
    max_iter : int
        Maximum number of iterations allowed for CLEAN.

    Notes
    -----
    This function is purely informative and does not influence control
    flow — it prints a summary of the achieved noise level after the
    iteration limit is reached.
    """
    peak_flux = np.max(res[mask])
    res_std = np.std(res)
    noise_sigma = peak_flux / res_std
    print(f"Reached iterations stopping threshold: {max_iter} at noise level {noise_sigma:.4f}")
