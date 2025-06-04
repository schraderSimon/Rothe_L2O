
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable
from scipy.optimize import minimize

import torch  # only needed for RNG parity with the L2O script

from L2O_hydrogen import * 

num_samples= 50
T_max = 100
def bfgs_scipy_run(f,grad,x0,max_iter = T_max):
    """Run SciPy BFGS, capture *fₖ / f₀* at each iteration (max_iter ≤ 100)."""
    f0 = f(x0)
    g0 = grad(x0)
    H0 = np.diag(1.0 / (np.abs(g0) + 1e-16) )

    ratios = np.full(max_iter, np.nan, dtype=np.float64)
    ratios[0] = 1.0

    def cb(xk): #Callback function to capture the ratio fₖ / f₀
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
            "gtol": 0.0,      # Run as long as possible
            "hess_inv0": H0,  
        },
    )

    # If SciPy exits early (line‑search issues, etc.), pad with last ratio.
    if np.isnan(ratios[-1]):
        last_valid = np.nanmax(np.where(~np.isnan(ratios))[0])
        ratios[last_valid + 1 :] = ratios[last_valid]

    return ratios


def generate_dataset(time_points, quality,E0) -> np.ndarray:
    data = np.empty((len(time_points), T_max), dtype=np.float64)
    for i, t in enumerate(time_points):
        if E0 == "all":
            E0_val = np.random.choice([0.03, 0.06, 0.12])
        else:
            E0_val = E0
        f, g, x0 = make_error_and_gradient_functions(E0_val, quality, t)
        data[i, :] = bfgs_scipy_run(f, g, x0)
        print(E0_val,i)
    return data

np.random.seed(seed)
torch.manual_seed(seed)

_cfg_narrow = {"tmin": 180, "tmax": 200, "tmin_test": 200, "tmax_test": 210}
_cfg_wide   = {"tmin": 180, "tmax": 260, "tmin_test": 260, "tmax_test": 280}

_grid = lambda lo, hi: np.arange(lo, hi + 1, 0.2)

train_narrow = np.random.choice(_grid(_cfg_narrow["tmin"], _cfg_narrow["tmax"]), num_samples, replace=False)
test_narrow  = np.random.choice(_grid(_cfg_narrow["tmin_test"], _cfg_narrow["tmax_test"]), num_samples, replace=False)
val_narrow   = np.random.choice(_grid(210, 250), num_samples, replace=False)
train_wide   = np.random.choice(_grid(_cfg_wide["tmin"], _cfg_wide["tmax"]), num_samples, replace=False)
test_wide    = np.random.choice(_grid(_cfg_wide["tmin_test"], _cfg_wide["tmax_test"]), num_samples, replace=False)
val_wide     = np.random.choice(_grid(280, 330), num_samples, replace=False)
output_dir = "results_L2O"

narrow_file = "%s/BFGS_narrow.npz"% output_dir
if not file_exists(narrow_file):
    train_data= generate_dataset(train_narrow,quality=2,E0=0.06)
    print("Done 1")
    test_data = generate_dataset(test_narrow,quality=2,E0=0.06)
    print("Done 2")
    val_data  = generate_dataset(val_narrow,quality=2,E0=0.06)
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
    
    np.savez(alt_file, val2=generate_dataset(train_narrow, quality=3, E0=0.06))
    print("Done 4")
else:
    print("Skipping narrow data2 generation")


wide_file = "%s/BFGS_wide.npz"% output_dir
if not file_exists(wide_file):
    train_data = generate_dataset(train_wide, quality=1, E0="all")
    print("Done 5")
    test_data  = generate_dataset(test_wide, quality=1, E0="all")
    print("Done 6")
    val_data   = generate_dataset(val_wide, quality=1, E0="all")
    print("Done 7")
    np.savez(
        wide_file,
        train=train_data,
        test=test_data,
        val=val_data,
    )
else:
    print("Skipping wide data generation")
# Alt‑validation (quality=2)
alt_wide_file = "%s/BFGS_wide2.npz"% output_dir
if not file_exists(alt_wide_file):
    np.savez(alt_wide_file, val2=generate_dataset(train_wide, quality=2, E0="all"))
    print("Done 8")
else:
    print("Skipping wide data2 generation")
print("All datasets generated successfully.")