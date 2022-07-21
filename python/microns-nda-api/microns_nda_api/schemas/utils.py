import numpy as np
def pcorr(x, y):
    """ Compute the cross correlation matrix of two 2D arrays
    Args：
        x: 2D numpy array， (n_features, n_samples)
        y: 2D numpy array， (n_features, n_samples)
    Returns:
        cross_correlation_matrix: 2D numpy array， (n_features, n_features)
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    assert x.shape[1] == y.shape[1] and x.ndim == y.ndim == 2
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    x = x / np.sqrt((x**2).sum(axis=1, keepdims=True))
    y = y / np.sqrt((y**2).sum(axis=1, keepdims=True))
    return (x * y).sum(axis=1)
    