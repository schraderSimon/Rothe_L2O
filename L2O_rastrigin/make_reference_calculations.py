import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys
# If you want SciPy’s BFGS:
from functools import partial
from scipy.optimize import minimize

from L2O_quadratic_singleVariable import generate_random_quadratic, quadratic
def generate_random_quadratic_problem(n):
    M = torch.randn(n, n)
    A = np.array(generate_random_quadratic(n,1)[0,:,:])  # positive semidefinite
    mu = np.random.randn(n)
    x0 = np.zeros(n)

    return A, mu, x0
def quadratic(x, A, mu):
    xminmu = x - mu
    return xminmu.T@A@xminmu  # Shape: (batch_size,)

def quadratic_grad(x, A, mu):
    """
    gradient of f(x).
    grad f(x) = 2 A (x - mu)  (since A is symmetric).
    """
    xminmu = x - mu
    # We assume A is symmetric. If not, we can do: (A + A.T)/2
    return 2.0 * (A @ xminmu)

def run_adam_on_quadratic(A, mu, x0, lr=1e-3, max_iters=100):
    """
    Run Adam to minimize f(x) = (x - mu)^T A (x - mu).
    We record f(x_k)/f(x0) at each iteration k.
    Returns a 1D numpy array of length max_iters with the relative errors.
    """
    # Convert A, mu, x0 to Torch. We'll do a direct definition of the objective using
    # PyTorch's autograd. 
    device = torch.device("cpu")
    A_t = torch.tensor(A, dtype=torch.float32, device=device, requires_grad=False)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device, requires_grad=False)

    # x will be a torch parameter
    x = torch.tensor(x0, dtype=torch.float32, device=device, requires_grad=True)

    # Set up Adam
    optimizer = optim.Adam([x], lr=lr)

    # initial error
    f0 = quadratic(x0, A, mu)
    rel_errors = np.zeros(max_iters, dtype=np.float32)

    for i in range(max_iters):
        optimizer.zero_grad()
        # compute f(x)
        diff = x - mu_t
        f_val = diff @ A_t @ diff
        f_val.backward()  # autograd for gradient
        optimizer.step()

        # store relative error
        current_f = f_val.detach().cpu().item()
        rel_errors[i] = current_f / f0 if f0 != 0.0 else 0.0

    return rel_errors

def run_bfgs_on_quadratic(A, mu, x0, max_iters=100):
    f0 = quadratic(x0, A, mu)
    if f0 == 0.0:
        return np.zeros(max_iters, dtype=np.float32)

    # compute initial grad
    g0 = quadratic_grad(x0, A, mu)
    initial_inv_hessian = np.diag(1/(abs(g0)+1e-14))

    iter_xs = []

    def callback(xk):
        iter_xs.append(xk.copy())

    def f_scipy(x):
        return quadratic(x, A, mu)

    def grad_scipy(x):
        return quadratic_grad(x, A, mu)

    res = minimize(
        fun=f_scipy,
        x0=x0,
        method='BFGS',
        jac=grad_scipy,
        callback=callback,
        options={
            'gtol': 1e-16,
            'maxiter': max_iters,
            'hess_inv0': initial_inv_hessian  
        }
    )

    iter_xs.insert(0, x0.copy())

    rel_errors = []
    for xk in iter_xs:
        rel_errors.append(quadratic(xk, A, mu)/f0)

    if len(rel_errors) < max_iters:
        rel_errors += [rel_errors[-1]] * (max_iters - len(rel_errors))
    else:
        rel_errors = rel_errors[:max_iters]

    return np.array(rel_errors, dtype=np.float32)


def main_experiment(
    num_problems = 1000,
    dims_min = 1,
    dims_max = 20,
    max_iters = 100,
    adam_learning_rates = [1e-4, 1e-3, 1e-2, 1e-1,1e0,1e1]
):

    np.random.seed(42)
    torch.manual_seed(42)
    adam_data = {}
    for lr in adam_learning_rates:
        adam_data[lr] = np.zeros((num_problems, max_iters), dtype=np.float32)

    # BFGS data
    bfgs_data = np.zeros((num_problems, max_iters), dtype=np.float32)

    # Generate problems and solve
    for i in tqdm(range(num_problems), desc="Solving problems"):
        # pick random dimension n
        n = np.random.randint(dims_min, dims_max+1)
        A, mu, x0 = generate_random_quadratic_problem(n)

        for lr in adam_learning_rates:
            rel_errs_adam = run_adam_on_quadratic(A, mu, x0, lr=lr, max_iters=max_iters)
            adam_data[lr][i, :] = rel_errs_adam

        rel_errs_bfgs = run_bfgs_on_quadratic(A, mu, x0, max_iters=max_iters)
        bfgs_data[i, :] = rel_errs_bfgs

    # -------------- Compute average and std across problems --------------
    iters_axis = np.arange(1, max_iters+1)

    adam_quartiles = {}
    for lr in adam_learning_rates:
        q25, q50, q75 = np.percentile(adam_data[lr], [25, 50, 75], axis=0)
        adam_quartiles[lr] = (q25, q50, q75)

    bfgs_q25, bfgs_q50, bfgs_q75 = np.percentile(bfgs_data, [25, 50, 75], axis=0)

    df_dict = {
        "iteration": iters_axis
    }
    for lr in adam_learning_rates:
        q25, q50, q75 = adam_quartiles[lr]
        df_dict[f"Adam(lr={lr})_q25"] = q25
        df_dict[f"Adam(lr={lr})_q50"] = q50
        df_dict[f"Adam(lr={lr})_q75"] = q75

    df_dict["BFGS_q25"] = bfgs_q25
    df_dict["BFGS_q50"] = bfgs_q50
    df_dict["BFGS_q75"] = bfgs_q75

    df = pd.DataFrame(df_dict)
    df.to_csv("reference_%s.csv"%sys.argv[1], index=False)
    print("Saved CSV: reference_%s.csv"%sys.argv[1])

if __name__ == "__main__":
    type_ref=sys.argv[1]
    if type_ref=="training":
        dims=[1,20]
    elif type_ref=="testing":
        dims=[21,50]
    main_experiment(dims_min=dims[0], dims_max=dims[1], num_problems=1000, max_iters=100)