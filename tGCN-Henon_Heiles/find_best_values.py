import os
import re
import numpy as np
from own_tgcn import *
T=int(sys.argv[1])
run_GCN = sys.argv[2].lower() == 'true'  # Convert string to boolean

output_dir = "outputs"

# Build regex to match correct files
t_pattern = f"T={T}_"
run_gcn_pattern = f"run_GCN={run_GCN}_"
file_regex = re.compile(f".*{t_pattern}.*{run_gcn_pattern}.*\\.txt$")

best_test_loss = float("inf")
best_epoch = None
best_filename = None

for fname in os.listdir(output_dir):
    if not file_regex.match(fname):
        continue

    full_path = os.path.join(output_dir, fname)
    test_losses = []
    with open(full_path, "r") as f:
        next(f)  
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                epoch = int(parts[0])
                test_loss = float(parts[2])
                test_losses.append((epoch, test_loss))
            except Exception:
                continue

    if test_losses:
        # Find minimum test loss and corresponding epoch
        min_epoch, min_loss = min(test_losses, key=lambda x: x[1])
        if min_loss < best_test_loss:
            best_test_loss = min_loss
            best_epoch = min_epoch
            best_filename = fname

if best_filename is not None:
    print(f"Best file: {best_filename}")
    print(f"Lowest test loss: {best_test_loss:.8f} at epoch {best_epoch}")
else:
    print("No matching files found.")
if best_filename is not None:
    param_pattern = (
    r"T=(\d+)_"
    r"batch_size=(\d+)_"
    r"l2_penalty=([\d\.eE-]+)_"
    r"learning_rate=([\d\.eE-]+)_"
    r"num_epochs=(\d+)_"
    r"num_layers_GCN=(\d+)_"
    r"num_layers_LSTM=(\d+)_"
    r"num_nodes=(\d+)_"
    r"run_GCN=(True|False)_"
    r"size_GCN=(\d+)_"
    r"size_GCN_out=(\d+)_"
    r"size_LSTM=(\d+)\.txt"
)
    match = re.match(param_pattern, best_filename)
if match:
    t = int(match.group(1))
    batch_size = int(match.group(2))
    l2_penalty = float(match.group(3))
    learning_rate = float(match.group(4))
    num_epochs = int(match.group(5))
    num_layers_GCN = int(match.group(6))
    num_layers_LSTM = int(match.group(7))
    num_nodes = int(match.group(8))
    run_GCN = match.group(9) == "True"
    size_GCN = int(match.group(10))
    size_GCN_out = int(match.group(11))
    size_LSTM = int(match.group(12))
    cfg = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'T': t,
        'num_layers_GCN': num_layers_GCN,
        'num_layers_LSTM': num_layers_LSTM,
        'size_LSTM': size_LSTM,
        'size_GCN': size_GCN,
        'size_GCN_out': size_GCN_out,
        'num_nodes': num_nodes,
        'run_GCN': run_GCN,
        'l2_penalty': l2_penalty,
    }
    print(cfg)
else:
    print("No matching files found; cannot build cfg.")


dim=3
if dim==2:
    infile_2D = np.load("nonlinear_coefficients_dimension=2_ngauss_init=29.npz")
    L_data=L_data_2D = infile_2D['L']
    K_data=K_data_2D = infile_2D['K']
    mu_data=mu_data_2D = infile_2D['mu']
    p_data=p_data_2D = infile_2D['p']
elif dim==3:
    infile_3D= np.load("nonlinear_coefficients_dimension=3_ngauss_init=28.npz")
    L_data_3D = infile_3D['L']
    K_data_3D = infile_3D['K']
    mu_data_3D = infile_3D['mu']
    p_data_3D = infile_3D['p']
    L_data=L_data_3D
    K_data=K_data_3D
    mu_data=mu_data_3D
    p_data=p_data_3D
num_coefficients=L_data.shape[2]+K_data.shape[2] + mu_data.shape[2] + p_data.shape[2]
num_gaussians=L_data.shape[1]
num_nodes= num_gaussians * num_coefficients
adjaceny_matrix= build_adjacency_matrix(num_gaussians=num_gaussians, num_coefficients=num_coefficients)
train_timesteps = 1000  
params_train, params_test_and_valid, mean, std = preprocess_data(L_data, K_data, mu_data, p_data, train_timesteps)
params_test=params_test_and_valid[0:200,:,:]
torch.manual_seed(42)  
np.random.seed(42)  

model = train_model(params_train, cfg,adjaceny_matrix,params_test,save_output=False,save_model_state=best_epoch)
