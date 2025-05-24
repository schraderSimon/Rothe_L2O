import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import json
import ray
from ray import tune
import sys
from ray.tune.schedulers import ASHAScheduler
# Import the new session and Checkpoint objects
from ray.tune import Checkpoint
#from ray.train import session, Checkpoint
import tempfile
import ray.cloudpickle as pickle
import numpy as np
# Define the shifted N-dimensional Rastrigin function.

def generate_random_quadratic(n, batch_size):
    # Random symmetric matrix + diagonal shift for positive definiteness
    A = torch.randn(batch_size, n, n) #Random matrix
    A =  torch.einsum('bij, bkj -> bik', A, A) 
    return A

def quadratic(x, A, mu):
    # x: (batch_size, n), A: (batch_size, n, n), mu: (batch_size, n), s: (batch_size, )
    # Compute (x-mu)^T A (x-mu)+s for each batch element
    xminmu = x - mu
    return torch.einsum('bi,bij,bj->b', xminmu, A, xminmu)  # Shape: (batch_size,)
def quadratic_grad(x,A,mu):
    xminmu=x-mu
    return 2.0 * torch.einsum('bij,bj->bi',A,xminmu)
# Define the L2O network that will learn the update rule.
# This version acts on one variable at a time.
class L2OOptimizer(nn.Module):
    def __init__(self, hidden_size, linear_size=10, num_layers=1):
        super(L2OOptimizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable hidden state
        self.c_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable cell state
        self.initial_transform = nn.Sequential(OrderedDict([ # Initial non-linear transformation
            ("linear1", nn.Linear(2, linear_size)),
            ("nonlin1", nn.ReLU()),
            ("linear2", nn.Linear(linear_size, linear_size)),
        ]))

        self.lstm_cells = nn.ModuleList() # List of LSTM cells
        self.lstm_cells.append(nn.LSTMCell(input_size=linear_size, hidden_size=hidden_size)) #First LSTM cell, input size is linear_size, hidden size is hidden_size

        for _ in range(1, num_layers):
            self.lstm_cells.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)) #Add num_layers-1 LSTM cells, input size is hidden_size, hidden size is hidden_size
        
        self.fc = nn.Linear(hidden_size, 1) #Final linear layer to output the perturbation
    def forward(self, x, grad, hidden_LSTM):
        batch_size, n = x.shape

        new_hidden_h_layers = torch.zeros(batch_size, n, self.num_layers, self.hidden_size, device=x.device) #Array containing the new hidden states after forward pass
        new_hidden_c_layers = torch.zeros(batch_size, n, self.num_layers, self.hidden_size, device=x.device) #Array containing the new cell states after forward pass

        x_new=torch.zeros(batch_size, n) #Array containing the new optimizee
        delta=torch.zeros(batch_size, n) #Update in x in the forward pass
        grad_expanded=grad[:,:,None] #Reshaping for numerical reasons
        gradient_signs = grad_expanded.sign()
        gradient_magnitudes=torch.log(torch.abs(grad_expanded)+1e-16)
        inp_all=torch.cat([gradient_signs, gradient_magnitudes], dim=2)  # (batch_size, n, 2)
        inp_transformed = self.initial_transform(inp_all)  # (batch_size, n, linear_size)

        inp_layer = inp_transformed # The first input to the LSTM is just the transformed input.
        for layer in range(self.num_layers):
            h_i = hidden_LSTM[0][:, :, layer, :]  
            c_i = hidden_LSTM[1][:, :, layer, :]  

            # Flatten batch_size and n into a single dimension for LSTM input
            # This essentially turns it into an "extra large batch" of size (batch_size * n)
            inp_reshaped=inp_layer.view(batch_size * n, -1) # view is more efficient than reshape
            h_i_reshaped=h_i.view(batch_size * n, -1)
            c_i_reshaped=c_i.view(batch_size * n, -1)
            h_new, c_new = self.lstm_cells[layer](inp_reshaped, (h_i_reshaped,  c_i_reshaped))

            # Reshape back to (batch_size, n, hidden_size)
            h_new = h_new.view(batch_size, n, -1)
            c_new = c_new.view(batch_size, n, -1)

            new_hidden_h_layers[:, :, layer, :] = h_new # Store the new hidden state
            new_hidden_c_layers[:, :, layer, :] = c_new # Store the new cell state
            inp_layer = h_new   # Update the input for the next layer

        delta = self.fc(h_new)[:, :, 0]
        x_new = x + delta
        return x_new, (new_hidden_h_layers, new_hidden_c_layers), delta


    def get_init_states(self, batch_size, n):
        h_0 = self.h_0[None,None,:,:].repeat(batch_size, n, 1, 1).contiguous()
        c_0 = self.c_0[None,None,:,:].repeat(batch_size, n, 1, 1).contiguous()
        return (h_0, c_0)
    
def train_l2o_tune(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    linear_size   = config["linear_size"]
    hidden_size   = config["hidden_size"]
    num_layers    = config["num_layers"]
    T             = config["T"]
    learning_rate = config["lr"]
    num_epochs    = config["num_epochs"]
    batch_size   = config["batch_size"]
    wm= config["w_multiplier"]
    nmin, nmax = 1, 20
    perturb = config["mu_std"]

    weights_T = torch.ones(T)
    weights_T[0] = 0
    for i in range(1, T - 1):
        weights_T[i+1] = weights_T[i]*wm
    weights_T = weights_T / weights_T.sum()

    l2o_net = L2OOptimizer(hidden_size, linear_size=linear_size, num_layers=num_layers)
    l2o_net.to(device)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)

    train_loss_buffer = []
    final_loss_buffer = []
    for epoch in range(num_epochs):
        n = int(torch.randint(nmin, nmax+1, (1,))[0])
        hidden = l2o_net.get_init_states(batch_size, n)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        A  = generate_random_quadratic(n, batch_size).to(device)
        mu = (torch.randn(batch_size, n)*perturb).to(device)

        x = torch.zeros(batch_size, n, device=device, requires_grad=True)

        init_loss = quadratic(x, A, mu)
        total_loss = 0.0
        final_loss_value = 0.0

        for j in range(T):
            f_val = quadratic(x, A, mu) / (init_loss + 1e-12)
            loss_step = f_val.mean()
            total_loss += loss_step * weights_T[j]

            if j == T-1:
                final_loss_value = loss_step.item()

            if j < T-1:
                grad_ = quadratic_grad(x, A, mu) / (init_loss.reshape(-1, 1) + 1e-12) / x.shape[0]
                x, hidden, _ = l2o_net(x, grad_, hidden)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(l2o_net.parameters(), 0.1)
        optimizer.step()

        current_train_loss = total_loss.item()
        current_final_loss = final_loss_value

        train_loss_buffer.append(current_train_loss)
        final_loss_buffer.append(current_final_loss)
        if len(train_loss_buffer) > 20:
            train_loss_buffer.pop(0)
            final_loss_buffer.pop(0)

        avg_train_loss_20 = sum(train_loss_buffer)/len(train_loss_buffer)
        avg_final_loss_20 = sum(final_loss_buffer)/len(final_loss_buffer)

        metrics = {
            "epoch": epoch,
            "train_loss": current_train_loss,
            "train_loss_20avg": avg_train_loss_20,
            "final_loss_20avg": avg_final_loss_20,
        }
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            with tempfile.TemporaryDirectory() as tmpdirname:
                ckpt_path = os.path.join(tmpdirname, "model_state_dict.pkl")
                with open(ckpt_path, "wb") as fp:
                    pickle.dump(l2o_net.state_dict(), fp)
                checkpoint = Checkpoint.from_directory(tmpdirname)
                tune.report(metrics, checkpoint=checkpoint)
        else:
            tune.report(metrics)
def train_l2o(config,create_output=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    linear_size   = config["linear_size"]
    hidden_size   = config["hidden_size"]
    num_layers    = config["num_layers"]
    T             = config["T"]
    learning_rate = config["lr"]
    num_epochs    = config["num_epochs"]
    batch_size   = config["batch_size"]
    wm= config["w_multiplier"]
    nmin, nmax = 1, 20
    perturb = config["mu_std"]

    weights_T = torch.ones(T)
    weights_T[0] = 0
    for i in range(1, T - 1):
        weights_T[i+1] = weights_T[i]*wm
    weights_T = weights_T / weights_T.sum()

    l2o_net = L2OOptimizer(hidden_size, linear_size=linear_size, num_layers=num_layers)
    l2o_net.to(device)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)

    train_loss_buffer = []
    final_loss_buffer = []
    if create_output:
        outfilename="training_trajectory_T=%d_wM=%.2f.npz"%(T,wm)
        epochs = []
        train_loss = []
    for epoch in range(num_epochs):
        n = int(torch.randint(nmin, nmax+1, (1,))[0])
        hidden = l2o_net.get_init_states(batch_size, n)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        A  = generate_random_quadratic(n, batch_size).to(device)
        mu = (torch.randn(batch_size, n)*perturb).to(device)

        x = torch.zeros(batch_size, n, device=device, requires_grad=True)

        init_loss = quadratic(x, A, mu)
        total_loss = 0.0
        final_loss_value = 0.0

        for j in range(T):
            f_val = quadratic(x, A, mu) / (init_loss + 1e-12)
            loss_step = f_val.mean()
            total_loss += loss_step * weights_T[j]

            if j == T-1:
                final_loss_value = loss_step.item()

            if j < T-1:
                grad_ = quadratic_grad(x, A, mu) / (init_loss.reshape(-1, 1) + 1e-12) / x.shape[0]
            x, hidden, _ = l2o_net(x, grad_, hidden)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(l2o_net.parameters(), 0.1)
        optimizer.step()

        current_train_loss = total_loss.item()
        current_final_loss = final_loss_value

        train_loss_buffer.append(current_train_loss)
        final_loss_buffer.append(current_final_loss)
        if len(train_loss_buffer) > 20:
            train_loss_buffer.pop(0)
            final_loss_buffer.pop(0)

        avg_train_loss_20 = sum(train_loss_buffer)/len(train_loss_buffer)
        avg_final_loss_20 = sum(final_loss_buffer)/len(final_loss_buffer)

        if create_output:
            epochs.append(epoch)
            train_loss.append(current_train_loss)
        np.savez(outfilename,epochs=epochs,train_loss=train_loss)

def run_tuning(config, num_samples=10):
    ray.init()

    scheduler = ASHAScheduler(
        metric="train_loss_20avg",
        mode="min",
        max_t=config["num_epochs"],
        grace_period=50,
        reduction_factor=2
    )

    analysis = tune.run(
        train_l2o_tune,
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        verbose=1
    )

    best_trial = analysis.get_best_trial(metric="final_loss_20avg", mode="min")
    best_checkpoint = analysis.get_best_checkpoint(
        best_trial, 
        metric="final_loss_20avg", 
        mode="min"
    )
    best_config = best_trial.config
    with open("best_config_T=%d_wM=%.2f.json" % (config["T"], config["w_multiplier"]), "w") as f:
        json.dump(best_config, f, indent=2)
    if best_checkpoint is not None:
        best_checkpoint_dir = best_checkpoint.to_directory()
        with open(os.path.join(best_checkpoint_dir, "model_state_dict.pkl"), "rb") as f:
            best_state = pickle.load(f)
            torch.save(best_state, "best_l2o_model_T=%d_wM=%.2f.pth" % (config["T"], config["w_multiplier"]))

def load_l2o_model(config_filename: str, state_dict_filename: str):
   
    with open(config_filename, "r") as f:
        best_config = json.load(f)

    l2o_net = L2OOptimizer(
        hidden_size=best_config["hidden_size"],
        linear_size=best_config["linear_size"],
        num_layers=best_config["num_layers"],
    )

    best_state = torch.load(state_dict_filename, map_location="cpu")
    l2o_net.load_state_dict(best_state)
    l2o_net.eval()

    return l2o_net
def run_l2o_on_new_problem(l2o_net, A, mu, T=10, device="cpu"):
    l2o_net.to(device)
    A, mu = A.to(device), mu.to(device)
    
    batch_size, n, _ = A.shape
    x = torch.zeros(batch_size, n, device=device, requires_grad=True)
    
    hidden = l2o_net.get_init_states(batch_size, n)
    hidden = (hidden[0].to(device), hidden[1].to(device))
    
    init_loss = quadratic(x, A, mu)
    losses = []
    
    with torch.no_grad():
        for step in range(T):
            vals=quadratic(x, A, mu)
            loss = (vals / (init_loss + 1e-12)).mean()
            losses.append(loss.item())
            
            gradient = quadratic_grad(x, A, mu) 
            grad= gradient/ (init_loss.reshape(-1,1)  + 1e-12)
            
            x, hidden, _ = l2o_net(x, grad, hidden)
    return x, losses
if __name__ == "__main__":
    seed = 42 
    torch.manual_seed(seed)
    T= int(sys.argv[1])
    w_multiplier = float(sys.argv[2])
    config = {
        "linear_size": tune.choice([4,8,16,32]),
        "hidden_size": tune.choice([64,128,256]),
        "num_layers":  tune.choice([1,2,3]),
        "T":           T,
        "lr":          tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([8,16,32, 64]),
        "num_epochs":  200,
        "mu_std": 10,
        "w_multiplier": w_multiplier,
    }
    run_tuning(config,num_samples=128)
    config_filename="best_config_T=%d_wM=%.2f.json"%(T,w_multiplier)
    with open(config_filename, "r") as f:
        best_config = json.load(f)
    print("Done with tuning, loading best model to produce training trajectory")
    train_l2o(config=best_config,create_output=True)