import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import sys
from L2O_quadratic_singleVariable import *


def main_experiment_l2o(num_problems=500, dims_min=1, dims_max=20, T=100,T_training=10):
    model = load_l2o_model(config_filename="best_config_T=%d.json"%T_training, state_dict_filename="best_l2o_model_T=%d.pth"%T_training)

    data = np.zeros((num_problems, T), dtype=np.float32)

    for i in tqdm(range(num_problems), desc="Running L2O problems"):
        n = np.random.randint(dims_min, dims_max + 1)
        A = generate_random_quadratic(n=n, batch_size=1)
        mu = torch.randn(1, n) * 10

        _, losses = run_l2o_on_new_problem(model, A, mu, T=T, device="cpu")

        # Pad if fewer iterations
        if len(losses) < T:
            losses += [losses[-1]] * (T - len(losses))

        data[i, :] = losses[:T]

    # Compute quartiles
    q25, q50, q75 = np.percentile(data, [25, 50, 75], axis=0)

    df = pd.DataFrame({
        "iteration": np.arange(1, T + 1),
        "L2O_q25": q25,
        "L2O_q50": q50,
        "L2O_q75": q75,
    })

    df.to_csv("l2o_experiment_results_quartiles_T=%d.csv"%T_training, index=False)
    print("Saved CSV: l2o_experiment_results_quartiles.csv")


if __name__ == "__main__":
    main_experiment_l2o(T_training=int(sys.argv[1]))