import numpy as np

def step(x):
    if x > 10:
        gain = 0.8
    elif x > 5:
        gain = 0.5
    else:
        gain = 0.1
    return gain

def scaled_logistic(x, shift=5, min_val=0.1, max_val=0.6):
    base = 1 / (1 + np.exp(-(x - shift)))
    return min_val + (max_val - min_val) * base