"""VMDMS decomposition step (per-channel VMD) from the full forecasting pipeline."""

import numpy as np

from VMD_pers import VMD_pers


def decompose_vmdms(data, alpha, tau, k_modes, dc, init, tol, omega0=None):
    """
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)

    Returns
    -------
    modes : ndarray, shape (n_channels, k_modes, n_samples)
    omegas : ndarray, shape (n_channels, k_modes)
    """
    modes = []
    omegas = []
    for channel in data:
        u, _, omega = VMD_pers(channel, alpha, tau, k_modes, dc, init, tol, omega0)
        modes.append(u)
        omegas.append(omega[-1])
    return np.asarray(modes), np.asarray(omegas)
