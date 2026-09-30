#!/usr/bin/env python3
"""Run VMD, MVMD, NCVMD, and VMDMS on the built-in toy example."""

import argparse
import sys

import numpy as np

from mvmd import MVMD
from ncvmd import NCVMD
from toy_data import artificial_2d_1, test_1d
from VMD_pers import VMD_pers
from vmdms import decompose_vmdms


def _rmse(original, reconstructed):
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def _run_vmd(data, params):
    rmses = []
    for channel in data:
        modes, _, _ = VMD_pers(channel, **params)
        rmses.append(_rmse(channel, modes.sum(axis=0)))
    return rmses


def _run_vmdms(data, params):
    modes, _ = decompose_vmdms(
        data,
        params["alpha"],
        params["tau"],
        params["K"],
        params["DC"],
        params["init"],
        params["tol"],
        params.get("omega0"),
    )
    rmses = []
    for i, channel in enumerate(data):
        rmses.append(_rmse(channel, modes[i].sum(axis=0)))
    return rmses


def _run_mvmd(data, params):
    u, _, _ = MVMD(
        data,
        params["alpha"],
        params["tau"],
        params["K"],
        params["DC"],
        params["init"],
        params["tol"],
        None,
        500,
    )
    # u: (K, T, C)
    recon = u.sum(axis=0).T
    return [_rmse(data[i], recon[i]) for i in range(data.shape[0])]


def _run_ncvmd(data, params):
    betas = (0,) + (0,) * (data.shape[0] - 1)
    modes_list, _, _ = NCVMD(data, params["K"], 0, params["tau"], betas, 500)
    modes = np.asarray(modes_list)
    recon = modes.sum(axis=1)
    return [_rmse(data[i], recon[i]) for i in range(data.shape[0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("Artificial2D_1", "Test_1D"),
        default="Artificial2D_1",
        help="Toy dataset (default: bivariate Artificial2D_1)",
    )
    args = parser.parse_args()

    if args.dataset == "Test_1D":
        signal = test_1d()
        data = signal.reshape(1, -1)
    else:
        data = artificial_2d_1()

    if data.shape[1] % 2:
        data = data[:, :-1]

    vmd_params = dict(
        alpha=2000,
        tau=1,
        K=6,
        DC=0,
        init=1,
        tol=1e-7,
        omega0=None,
    )

    print(f"Dataset: {args.dataset}  shape={data.shape}")
    print("Per-channel reconstruction RMSE:")
    for name, runner in (
        ("VMD", _run_vmd),
        ("VMDMS", _run_vmdms),
        ("MVMD", _run_mvmd),
        ("NCVMD", _run_ncvmd),
    ):
        try:
            rmses = runner(data, vmd_params)
            print(f"  {name}: " + ", ".join(f"ch{i + 1}={r:.6e}" for i, r in enumerate(rmses)))
        except Exception as exc:
            print(f"  {name}: FAILED ({exc})", file=sys.stderr)
            return 1

    print("OK — all decompositions completed on toy data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
