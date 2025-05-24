from L2O_hydrogen import *
def test_l2o_model(l2o, test_times, quality=2, E0=0.06):

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    l2o.eval()
    T=100 #Test over 100 steps
    run_data=np.zeros((len(test_times), T), dtype=np.float64)
    for i,t in enumerate(test_times):
        optimisee, optimisee_grad, init_params = make_error_and_gradient_functions(
            E0, quality, t
        )

        parameters = torch.tensor(init_params, dtype=torch.float64,
                                  device=device).unsqueeze(0)
        init_loss  = optimisee(init_params)             # scalar float
        init_loss_t = torch.tensor([init_loss], dtype=torch.float64,
                                   device=device)

        hidden = l2o.get_init_states(1, parameters.size(1))
        hidden = (hidden[0].to(device), hidden[1].to(device))

        # ------------------ unroll the inner loop ----------------------
        for j in range(T):
            if j == 0:
                f_val = init_loss_t/(init_loss_t + 1e-16) # initial loss ratio
            else:
                curr_loss = optimisee(parameters[0].detach().cpu().numpy())
                curr_loss_t = torch.tensor([curr_loss], dtype=torch.float64,
                                           device=device)
                f_val = curr_loss_t / (init_loss_t + 1e-16)

            # take an L2O step except on final iteration
            if j < T - 1:
                grad_np  = optimisee_grad(parameters[0].detach().cpu().numpy())
                grads_t  = torch.tensor(grad_np, dtype=torch.float64,
                                        device=device).unsqueeze(0)
                parameters, hidden, _ = l2o(parameters, grads_t, hidden)

        # after T steps, f_val stores final-to-initial loss ratio
        # and parameters stores the final parameters
            print(i,j, f_val.item())
            run_data[i, j] = f_val.item()

    return run_data

def load_model_from_config(config, epoch=980,tag=None):
    if tag is None:
        tag = (
            f"size{config['linear_size']}"
            f"_layers{config['num_layers']}"
            f"_lr{config['lr']:.3f}"
            f"_bs{config['batch_size']}"
            f"_l1{config['l1']:d}"            
            f"_l2{config['l2']:.0e}"            
            f"_tmin{config['tmin']}"
            f"_tmax{config['tmax']}"
            f"_tmin_test{config['tmin_test']}"
            f"_tmax_test{config['tmax_test']}"
        )
    run_dir = f"runs/{tag}"
    ckpt_file = f"{run_dir}/ep{epoch:03d}.pt"
    return load_l2o_model(ckpt_file)
def load_l2o_model(checkpoint_path, device=None):
    """
    Load a trained L2OOptimizer model from a checkpoint and prepare it for evaluation.

    Args:
        checkpoint_path (str or Path): Path to the .pt checkpoint file.
        device (str, optional): Device to load the model on. Defaults to 'cuda' if available.

    Returns:
        l2o (L2OOptimizer): Loaded L2O optimizer model in eval mode.
        config (dict): Configuration dictionary used to train the model.
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    l2o = L2OOptimizer(
        hidden_size=config["hidden_size"],
        linear_size=config["linear_size"],
        num_layers=config["num_layers"]
    )
    l2o.load_state_dict(checkpoint["model_state"])
    l2o.to(device)
    l2o.eval()

    return l2o, config


# NARROW MODEL - best parameters
config_narrow=config= {
  "linear_size": 256,
  "hidden_size": 256,
  "num_layers": 3,
  "T": 10,
  "lr": 0.001,
  "batch_size": 2,
  "num_epochs": 1000, #1000 epochs to train the best model
  "w_multiplier": 1,
  "l2": 1e-05,
  "l1": 0,
  "tmin": 180,
  "tmax": 200,
  "tmin_test": 200,
  "tmax_test": 210
}
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
quality=2
E0=0.06
cfg=config
tag = (
f"size{cfg['linear_size']}"
f"_layers{cfg['num_layers']}"
f"_lr{cfg['lr']}"
f"_bs{cfg['batch_size']}"
f"_l1{cfg['l1']}"            
f"_l2{cfg['l2']}"            
f"_tmin{cfg['tmin']}"
f"_tmax{cfg['tmax']}"
f"_tmin_test{cfg['tmin_test']}"
f"_tmax_test{cfg['tmax_test']}"
)
mapname="runs/"+tag
if os.path.exists(mapname):
    if "ep980.pt" in os.listdir(mapname):
        print("Narrow model already trained")
    else:
        train_l2o_realistic(config,quality=quality, E0=E0,save_params=True)
    l2o_narrow, _ = load_model_from_config(config, epoch=980)
from L2O_hydrogen import *

###WIDE MODEL - best parameters
config_wide=config={
  "linear_size": 256,
  "hidden_size": 256,
  "num_layers": 3,
  "T": 10,
  "lr": 0.01,
  "batch_size": 2,
  "num_epochs": 1000,
  "w_multiplier": 1,
  "l2": 0.01,
  "l1": 0,
  "tmin": 100,
  "tmax": 200,
  "tmin_test": 210,
  "tmax_test": 330
}


seed=42
torch.manual_seed(seed)
np.random.seed(seed)

cfg=config
tag="size256_layers3_lr0.01_bs2_l10_l20.01_tmin100_tmax200_tmin_test210_tmax_test330"
mapname="runs/"+tag
if os.path.exists(mapname):
    if "ep980.pt" in os.listdir(mapname):
        print("wide model lready trained")
    else:
        train_l2o_realistic(config,quality=quality, E0=E0,save_params=True)
    l2o_broad, _ = load_model_from_config(config, epoch=980,tag=tag )
num_samples=50
training_range_narrow=np.random.choice(np.arange(config_narrow["tmin"], config_narrow["tmax"]+1,0.2),num_samples,replace=False)
test_range_narrow=np.random.choice(np.arange(config_narrow["tmin_test"], config_narrow["tmax_test"]+1,0.2),num_samples,replace=False)
validation_range_narrow=np.random.choice(np.arange(210, 250,0.2),num_samples,replace=False)
import os
file_exists = lambda path: os.path.isfile(path)
if not file_exists("results_L2O/narrow_data.npz"):
    narrow_train_data = test_l2o_model(l2o_narrow, training_range_narrow, quality=quality, E0=E0)
    narrow_test_data = test_l2o_model(l2o_narrow, test_range_narrow, quality=quality, E0=E0)
    narrow_val_data = test_l2o_model(l2o_narrow, validation_range_narrow, quality=quality, E0=E0)
    np.savez("results_L2O/narrow_data",train=narrow_train_data, test=narrow_test_data, val=narrow_val_data)
else:
    print("File already exists, skipping narrow model data generation")
    
if not file_exists("results_L2O/narrow_data2.npz"):
    narrov_val_data2=test_l2o_model(l2o_narrow, training_range_narrow, quality=3, E0=E0) #Use a different quality for the validation set
    np.savez("results_L2O/narrow_data2",val2=narrov_val_data2)
else:
    print("File already exists, skipping narrow model data2 generation")
training_range_wide=np.random.choice(np.arange(config_wide["tmin"], config_wide["tmax"]+1,0.2),num_samples,replace=False)

test_range_wide=np.random.choice(np.arange(config_wide["tmin_test"], config_wide["tmax_test"]+1,0.2),num_samples,replace=False)

wide_train_data = test_l2o_model(l2o_broad, training_range_wide, quality=quality, E0=E0)
wide_test_data = test_l2o_model(l2o_broad, test_range_wide, quality=quality, E0=E0)
if not file_exists("results_L2O/wide_data.npz"):
    np.savez("results_L2O/wide_data",train=wide_train_data, test=wide_test_data)
else:
    print("File already exists, skipping wide model data generation")