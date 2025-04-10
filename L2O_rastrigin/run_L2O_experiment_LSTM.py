import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import sys
from L2O_quadratic_singleVariable import *


def main_experiment_l2o(num_problems=1000, dims=[1,20], T=100,T_training=10,wM=1.0,type="training"):
    np.random.seed(42)
    torch.manual_seed(42)
    model = load_l2o_model(config_filename="best_config_T=%d_wM=%.2f.json"%(T_training,wM), state_dict_filename="best_l2o_model_T=%d_wM=%.2f.pth"%(T_training,wM))

    data = np.zeros((num_problems, T), dtype=np.float32)
    dims_min,dims_max=dims
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
    if type== "training":
        df.to_csv("LSTM_L2O_training_T=%d_wM=%.2f.csv"%(T_training,wM), index=False)
    elif type=="testing":
        df.to_csv("LSTM_L2O_testing_T=%d_wM=%.2f.csv"%(T_training,wM), index=False)
    print("Saved CSV: l2o_experiment_results_quartiles.csv")


if __name__ == "__main__":
    type=sys.argv[3]
    T_training=int(sys.argv[1])
    wM=float(sys.argv[2])
    if type=="training":
        dims=[1,20]
        main_experiment_l2o(T_training=T_training,wM=wM,type="training",dims=dims)
    elif type=="testing":
        dims=[21,50]
        main_experiment_l2o(T_training=T_training,wM=wM,type="testing",dims=dims)
    else:
        print("Invalid type. Use 'training' or 'testing'.")
        sys.exit(1)
