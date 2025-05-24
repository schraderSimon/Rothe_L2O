import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import sys
from ssm_l2o import *

def load_l2o_model(config_filename: str, state_dict_filename: str):
    with open(config_filename, "r") as f:
        best_config = json.load(f)
    
    l2o_net = L2OOptimizer(
        state_size=best_config["state_size"],
        linear_size=best_config["linear_size"]
    )
    
    best_state = torch.load(state_dict_filename, map_location="cpu")
    l2o_net.load_state_dict(best_state)
    l2o_net.eval()
    
    return l2o_net

def main_experiment_l2o(num_problems=1000, dims=[1,20], T=100, T_training=10, wM=1.0, type="training"):
    np.random.seed(42)
    torch.manual_seed(42)
    model = load_l2o_model(
        config_filename="best_config_T=%d_wM=%.2f.json" % (T_training, wM),
        state_dict_filename="best_l2o_model_T=%d_wM=%.2f.pth" % (T_training, wM)
    )
    
    data = np.zeros((num_problems, T), dtype=np.float32)
    dims_min, dims_max = dims
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
    if type == "training":
        df.to_csv("SSM_L2O_training_T=%d_wM=%.2f.csv" % (T_training, wM), index=False)
    elif type == "testing":
        df.to_csv("SSM_L2O_testing_T=%d_wM=%.2f.csv" % (T_training, wM), index=False)
    print(f"Saved CSV: SSM_L2O_{type}_T=%d_wM=%.2f.csv" % (T_training, wM))

if __name__ == "__main__":
    type = sys.argv[3]
    T_training = int(sys.argv[1])
    wM = float(sys.argv[2])
    if type == "training":
        dims = [1, 20]
        main_experiment_l2o(T_training=T_training, wM=wM, type="training", dims=dims)
    elif type == "testing":
        dims = [21, 50]
        main_experiment_l2o(T_training=T_training, wM=wM, type="testing", dims=dims)
    else:
        print("Invalid type. Use 'training' or 'testing'.")
        sys.exit(1)