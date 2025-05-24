
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable
from scipy.optimize import minimize

import torch  # only needed for RNG parity with the L2O script

from L2O_hydrogen import * 

# ----------------------------------------------------------------------------
# Hyper‑parameters ------------------------------------------------------------
# ----------------------------------------------------------------------------
seed = 42
num_samples= 50
T_max = 100
quality_def = 2
E0_def = 0.06
def bfgs_scipy_run(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = T_max,
) -> np.ndarray:
    """Run SciPy BFGS, capture *fₖ / f₀* at each iteration (max_iter ≤ 100)."""
    f0 = f(x0)
    g0 = grad(x0)
    H0 = np.diag(1.0 / (np.abs(g0) + 1e-16) )

    ratios = np.full(max_iter, np.nan, dtype=np.float64)
    ratios[0] = 1.0

    def cb(xk: np.ndarray):
        idx = cb.k
        if idx < max_iter:
            ratios[idx] = f(xk) / f0
        cb.k += 1

    cb.k = 1  # we already stored k = 0

    res = minimize(
        f,
        x0,
        method="BFGS",
        jac=grad,
        callback=cb,
        options={
            "maxiter": max_iter,
            "gtol": 0.0,      
            "disp": False,
            "hess_inv0": H0,  
        },
    )

    # If SciPy exits early (line‑search issues, etc.), pad with last ratio.
    if np.isnan(ratios[-1]):
        last_valid = np.nanmax(np.where(~np.isnan(ratios))[0])
        ratios[last_valid + 1 :] = ratios[last_valid]

    return ratios


def generate_dataset(time_points: np.ndarray, quality: int = quality_def) -> np.ndarray:
    data = np.empty((len(time_points), T_max), dtype=np.float64)
    for i, t in enumerate(time_points):
        f, g, x0 = make_error_and_gradient_functions(E0_def, quality, t)
        data[i, :] = bfgs_scipy_run(f, g, x0)
        print(i)
    return data

np.random.seed(seed)
if torch is not None:
    torch.manual_seed(seed)

_cfg_narrow = {"tmin": 180, "tmax": 200, "tmin_test": 200, "tmax_test": 210}
_cfg_wide   = {"tmin": 100, "tmax": 200, "tmin_test": 210, "tmax_test": 330}

_grid = lambda lo, hi: np.arange(lo, hi + 1, 0.2)

train_narrow = np.random.choice(_grid(_cfg_narrow["tmin"], _cfg_narrow["tmax"]), num_samples, replace=False)
test_narrow  = np.random.choice(_grid(_cfg_narrow["tmin_test"], _cfg_narrow["tmax_test"]), num_samples, replace=False)
val_narrow   = np.random.choice(_grid(210, 250), num_samples, replace=False)
train_wide   = np.random.choice(_grid(_cfg_wide["tmin"], _cfg_wide["tmax"]), num_samples, replace=False)
test_wide    = np.random.choice(_grid(_cfg_wide["tmin_test"], _cfg_wide["tmax_test"]), num_samples, replace=False)
output_dir = "results_L2O"

narrow_file = "%s/BFGS_narrow.npz"% output_dir
if not file_exists(narrow_file):
    train_data= generate_dataset(train_narrow)
    print("Done 1")
    test_data = generate_dataset(test_narrow)
    print("Done 2")
    val_data  = generate_dataset(val_narrow)
    print("Done 3")
    np.savez(
        narrow_file,
        train=train_data,
        test=test_data,
        val=val_data,
    )
else:
    print("Skipping narrow data generation")

# Alt‑validation (quality=3)
alt_file =   "%s/BFGS_narrow2.npz"% output_dir
if not file_exists(alt_file):
    
    np.savez(alt_file, val2=generate_dataset(train_narrow, quality=3))
    print("Done 4")
else:
    print("Skipping narrow data2 generation")
# Broad / wide split
broad_file = "%s/BFGS_broad.npz"% output_dir
if not file_exists(broad_file):
    train_data= generate_dataset(train_wide)
    print("Done 5")
    test_data = generate_dataset(test_wide)
    print("Done 6")
    np.savez(
        broad_file,
        train=train_data,
        test=test_data,
    )

else:
    print("Skipping wide generation")

print("Done.")
