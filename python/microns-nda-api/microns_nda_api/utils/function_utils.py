import numpy as np
from tqdm import tqdm


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
        pairwise_correlation: 1D numpy array， (n_features,)
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

def xcorr_p(x, n_perm=1000, rng=np.random.default_rng(0), replace=True, pbar=True):
    """Compute the cross correlation matrix of X and its permutation p value

    Args:
        x (2D numpy array): (n_features, n_samples)
        n_perm (int, optional): Number of permutations. Defaults to 1000.
        rng (np.random.Generator, optional): Random number generator. Defaults to np.random.default_rng(0).
        replace (bool, optional): Whether to sample with replacement. Defaults to True.
    """
    x = np.atleast_2d(np.array(x, dtype=np.float64))
    rho = np.corrcoef(x)
    perm_small = np.zeros((x.shape[0], x.shape[0]))
    perm_large = np.zeros((x.shape[0], x.shape[0]))

    for _ in tqdm(range(n_perm), disable=not pbar):
        perm_mat = np.stack([rng.choice(v, v.shape, replace=replace) for v in x], axis=0)
        perm_small += np.corrcoef(perm_mat) < rho
        perm_large += np.corrcoef(perm_mat) > rho
    p = (np.min(np.stack([perm_small, perm_large]), axis=0) * 2 + 1) / n_perm
    return rho, p

def pcorr_p(x, y, n_perm=1000, rng=np.random.default_rng(0), replace=True, pbar=True):
    """Compute the pairwise correlation of X and Y and its permutation p value

    Args:
        x (2D numpy array): (n_features, n_samples)
        y (2D numpy array): (n_features, n_samples)
        n_perm (int, optional): Number of permutations. Defaults to 1000.
        rng (np.random.Generator, optional): Random number generator. Defaults to np.random.default_rng(0).
        replace (bool, optional): Whether to sample with replacement. Defaults to True.
    """
    x = np.atleast_2d(np.array(x, dtype=np.float64))
    y = np.atleast_2d(np.array(y, dtype=np.float64))
    rho = pcorr(x, y)
    perm_small = np.zeros(x.shape[0])
    perm_large = np.zeros(x.shape[0])

    for _ in tqdm(range(n_perm), disable=not pbar):
        perm_x = np.stack([rng.choice(v, v.shape, replace=replace) for v in x], axis=0)
        perm_y = np.stack([rng.choice(v, v.shape, replace=replace) for v in x], axis=0)
        perm_small += pcorr(perm_x, perm_y) < rho
        perm_large += pcorr(perm_x, perm_y) > rho
    p = (np.min(np.stack([perm_small, perm_large]), axis=0) * 2 + 1) / n_perm
    return rho, p