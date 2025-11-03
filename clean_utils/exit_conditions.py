import numpy as np

def noise(res, thresh, peak):
    res_std = np.std(res)

    # Handle multi-peak input (take the strongest one)
    if isinstance(peak, (list, tuple, np.ndarray)):
        peak = np.max(peak)

    noise_sigma = peak / res_std
    if noise_sigma < thresh:
        print(f"Reached noise stopping threshold: {noise_sigma:.4f} < {thresh}")
        return True

    
def iterations(res, mask, max_iter):
    peak_flux = np.max(res[mask])
    res_std = np.std(res)
    noise_sigma = peak_flux / res_std        
    print(f"Reached iterations stopping threshold: {max_iter} at noise level {noise_sigma:.4f}")