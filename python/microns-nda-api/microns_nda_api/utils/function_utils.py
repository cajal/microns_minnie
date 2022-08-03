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
