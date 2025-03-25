import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# If you want SciPy’s BFGS:
from functools import partial
from scipy.optimize import minimize

from L2O_quadratic_singleVariable import generate_random_quadratic, quadratic
def generate_random_quadratic_problem(n):
    M = torch.randn(n, n)
    A = np.array(generate_random_quadratic(n,1)[0,:,:])  # positive semidefinite
    # random shift mu
    mu = np.random.randn(n)
    # random initial guess
    x0 = np.zeros(n)

    return A, mu, x0
def quadratic(x, A, mu):
    # x: (batch_size, n), A: (batch_size, n, n), mu: (batch_size, n), s: (batch_size, )
    # Compute (x-mu)^T A (x-mu)+s for each batch element
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

# ----------------------------------------------------------------------------------------
# 2. Adam optimization (PyTorch) for each problem
# ----------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------
# 3. BFGS optimization (SciPy), with custom initial_inv_hessian
# ----------------------------------------------------------------------------------------

def run_bfgs_on_quadratic(A, mu, x0, max_iters=100):
    """
    Minimize f(x) with BFGS, storing relative error each iteration.

    We'll use an initial inverse-Hessian ~ 1/||grad f(x0)|| * I 
    (i.e. scaled identity), so that the method is not sensitive to overall scale.

    This requires SciPy >= 1.11 for the `initial_inv_hessian` option.
    If you have an older SciPy version, you will need to manually implement or patch.
    """
    f0 = quadratic(x0, A, mu)
    if f0 == 0.0:
        # If by chance the initial guess is exactly the solution, just return zeros
        return np.zeros(max_iters, dtype=np.float32)

    # compute initial grad
    g0 = quadratic_grad(x0, A, mu)
    initial_inv_hessian = np.diag(1/(abs(g0)+1e-14))

    # We'll store x at each iteration in a list so we can track function values
    iter_xs = []

    def callback(xk):
        # Called once per iteration by SciPy, appends the current guess
        iter_xs.append(xk.copy())

    # Define function and gradient in a SciPy-friendly way
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
            'hess_inv0': initial_inv_hessian  # requires SciPy >= 1.11
        }
    )

    # SciPy calls callback() at each iteration, so iter_xs will have the guesses from iteration 1..end
    # We can also prepend the initial guess:
    iter_xs.insert(0, x0.copy())

    # If the solver stopped early, we might have fewer steps than max_iters. 
    # We'll artificially pad so that we return exactly max_iters entries:
    rel_errors = []
    for xk in iter_xs:
        rel_errors.append(quadratic(xk, A, mu)/f0)

    # If the solver used fewer steps than max_iters, pad with the last value
    if len(rel_errors) < max_iters:
        rel_errors += [rel_errors[-1]] * (max_iters - len(rel_errors))
    else:
        # or, if for some reason it used more, truncate
        rel_errors = rel_errors[:max_iters]

    return np.array(rel_errors, dtype=np.float32)

# ----------------------------------------------------------------------------------------
# 4. Main experiment
# ----------------------------------------------------------------------------------------

def main_experiment(
    num_problems = 500,
    dims_min = 1,
    dims_max = 20,
    max_iters = 100,
    adam_learning_rates = [1e-4, 1e-3, 1e-2, 1e-1,1e0,1e1]
):
    """
    1) Generate random dimension n in [dims_min..dims_max],
       random positive definite A, random shift mu, random x0.
    2) Optimize with Adam (4 LRs). Store relative errors (size [num_problems, max_iters]).
    3) Optimize with BFGS (1 setting). Store relative errors (size [num_problems, max_iters]).
    4) Plot average ± std for each method. 
    5) Save data to CSV for each method.
    """

    # We will store data in shape = (num_problems, max_iters)
    # Then we can average across the 0th axis (the problems)
    # for each LR in the Adam set:
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

        # run Adam for each LR
        for lr in adam_learning_rates:
            rel_errs_adam = run_adam_on_quadratic(A, mu, x0, lr=lr, max_iters=max_iters)
            adam_data[lr][i, :] = rel_errs_adam

        # run BFGS
        rel_errs_bfgs = run_bfgs_on_quadratic(A, mu, x0, max_iters=max_iters)
        bfgs_data[i, :] = rel_errs_bfgs

    # -------------- Compute average and std across problems --------------
    iters_axis = np.arange(1, max_iters+1)

    # For Adam, compute quartiles:
    adam_quartiles = {}
    for lr in adam_learning_rates:
        # percentiles returns shape (3, max_iters) for 25th, 50th, 75th
        q25, q50, q75 = np.percentile(adam_data[lr], [25, 50, 75], axis=0)
        adam_quartiles[lr] = (q25, q50, q75)

    # For BFGS, compute quartiles:
    bfgs_q25, bfgs_q50, bfgs_q75 = np.percentile(bfgs_data, [25, 50, 75], axis=0)

    # If you still want to save CSV with mean ± std, you can keep that block,
    # but if you want quartiles in the CSV, you might replace that code with:
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
    df.to_csv("quad_experiment_results_quartiles.csv", index=False)
    print("Saved CSV: quad_experiment_results_quartiles.csv")

    # -------------- Plot with median ± quartiles --------------

    # Plot Adam quartiles
    plt.figure()
    for lr in adam_learning_rates:
        q25, q50, q75 = adam_quartiles[lr]
        # q50 is the median line
        plt.plot(iters_axis, q50, label=f"Adam LR={lr}")
        # shade between the 25th and 75th quartiles
        plt.fill_between(iters_axis, q25, q75, alpha=0.2)
    plt.plot(iters_axis, bfgs_q50, label="BFGS (median)")
    plt.fill_between(iters_axis, bfgs_q25, bfgs_q75, alpha=0.2)
    plt.yscale("log")
    plt.ylim(1e-16, 10)
    plt.xlabel("Iteration")
    plt.ylabel("Relative error f(x)/f(x0)")
    plt.title("Performance on random Quadratic Problems (Median ± [25%, 75%])")
    plt.legend()
    plt.savefig("quadratic_results_quartiles.png", dpi=150)
    plt.show()


    print("Done. Plots saved as PNG files.")

if __name__ == "__main__":
    main_experiment()