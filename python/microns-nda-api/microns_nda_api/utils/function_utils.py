import numpy as np


def cdiff(alpha, beta, period=np.pi):
    """
    Returns the cirvular difference between two orientations given the period
    """
    return (alpha - beta + period / 2) % period - period / 2


def cdist(alpha, beta, period=np.pi):
    """
    Returns the cirvular distance between two orientations given the period

    Example:
        import numpy as np
        from matplotlib import pyplot as plt
        ori_scale = np.linspace(0, np.pi, 100)
        ori_x, ori_y = np.meshgrid(ori_scale, ori_scale)
        delta_ori = cdist(ori_x, ori_y)
        plt.scatter(ori_x.ravel(), ori_y.ravel(), c=delta_ori.ravel())
        plt.colorbar()
    """
    return np.abs(cdiff(alpha, beta, period))

def pcorr(x, y):
    """ Compute the pairwise correlation of two 2D arrays
    Args：
        x: 2D numpy array， (n_features, n_samples)
        y: 2D numpy array， (n_features, n_samples)
    Returns:
        pairwise_correlation: 1D numpy array， (n_samples,)
    """
    x = np.atleast_2d(np.array(x, dtype=np.float64))
    y = np.atleast_2d(np.array(y, dtype=np.float64))
    assert x.shape[1] == y.shape[1] and x.ndim == y.ndim == 2
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    x = x / np.sqrt((x**2).sum(axis=1, keepdims=True))
    y = y / np.sqrt((y**2).sum(axis=1, keepdims=True))
    return (x * y).sum(axis=1)

def xcorr(x):
    """ Compute the cross correlation matrix of two 2D arrays
    Args：
        x: 2D numpy array， (n_features, n_samples)
        y: 2D numpy array， (n_features, n_samples)
    Returns:
        cross_correlation_matrix: 2D numpy array， (n_features, n_features)
    """
    x = np.atleast_2d(np.array(x, dtype=np.float64))
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.sqrt((x**2).sum(axis=1, keepdims=True))
    return x @ x.T