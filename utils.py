import numpy as np

def rotation_matrix(alpha):
    return np.array([
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])