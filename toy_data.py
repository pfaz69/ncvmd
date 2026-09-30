"""Synthetic toy datasets (no external files, no electric-load data)."""

import numpy as np


def test_1d(seed=0):
    """Univariate toy signal from the repository's Test_1D setup."""
    rng = np.random.default_rng(seed)
    t_len = 1000
    t = np.linspace(1 / t_len, 1, t_len)
    v1 = np.cos(2 * np.pi * 2 * t)
    v2 = 0.25 * np.cos(2 * np.pi * 24 * t)
    v3 = (1 / 16) * np.cos(2 * np.pi * 288 * t)
    signal = v1 + v2 + v3 + 0.1 * rng.standard_normal(t_len)
    return signal.astype(np.float64)


def artificial_2d_1():
    """
    Bivariate toy signal from the repository's Artificial2D_1 setup.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
    """
    t = np.arange(1, 1000 + 1) / 1000
    ch1 = (
        np.cos(2 * np.pi * 2 * t)
        + (1 / 16) * np.cos(2 * np.pi * 36 * t)
        + (1 / 8) * np.cos(2 * np.pi * 64 * t)
    )
    ch2 = (
        (1 / 4) * np.cos(2 * np.pi * 24 * t)
        + (1 / 16) * np.cos(2 * np.pi * 44 * t)
        + (1 / 12) * np.cos(2 * np.pi * 88 * t)
    )
    return np.array([ch1, ch2], dtype=np.float64)
