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
from ray.tune import Checkpoint
import tempfile
import ray.cloudpickle as pickle
import numpy as np

# Quadratic function
def generate_random_quadratic(n, batch_size):
    A = torch.randn(batch_size, n, n)
    A = torch.einsum('bij,bkj->bik', A, A)
    return A

def quadratic(x, A, mu):
    xminmu = x - mu
    return torch.einsum('bi,bij,bj->b', xminmu, A, xminmu)

def quadratic_grad(x, A, mu):
    xminmu = x - mu
    return 2.0 * torch.einsum('bij,bj->bi', A, xminmu)

# SSM-based L2O Optimizer
class L2OOptimizer(nn.Module):
    def __init__(self, state_size, linear_size=10):
        super(L2OOptimizer, self).__init__()
        self.state_size = state_size
        
        # Initial transformation (same as LSTM)
        self.initial_transform = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(2, linear_size)),
            ("nonlin1", nn.ReLU()),
            ("linear2", nn.Linear(linear_size, linear_size)),
        ]))
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)  # State transition
        self.B = nn.Parameter(torch.randn(linear_size, state_size))  # Input projection
        self.C = nn.Parameter(torch.randn(state_size, 1))  # Output projection
        
        # Initial state
        self.h_0 = nn.Parameter(torch.zeros(state_size))
    
    def forward(self, x, grad, hidden):
        batch_size, n = x.shape
        grad_expanded = grad[:, :, None]  # (batch_size, n, 1)
        gradient_signs = grad_expanded.sign()
        gradient_magnitudes = torch.log(torch.abs(grad_expanded) + 1e-16)
        inp_all = torch.cat([gradient_signs, gradient_magnitudes], dim=2)  # (batch_size, n, 2)
        inp_transformed = self.initial_transform(inp_all)  # (batch_size, n, linear_size)
        
        # Flatten for SSM processing
        inp_flat = inp_transformed.view(batch_size * n, -1)  # (batch_size * n, linear_size)
        h_prev = hidden.view(batch_size * n, -1)  # (batch_size * n, state_size)
        
        # SSM update: h_t = A * h_{t-1} + B * u_t
        h_new = h_prev @ self.A + inp_flat @ self.B  # (batch_size * n, state_size)
        h_new = torch.tanh(h_new)  # Non-linearity
        
        # Output delta
        delta = (h_new @ self.C).view(batch_size, n)  # (batch_size, n)
        x_new = x + delta
        
        # Reshape hidden state
        h_new = h_new.view(batch_size, n, self.state_size)
        
        return x_new, h_new, delta
    
    def get_init_states(self, batch_size, n):
        h_0 = self.h_0[None, None, :].repeat(batch_size, n, 1).contiguous()
        return h_0

# Training function for Ray Tune
def train_l2o_tune(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    linear_size = config["linear_size"]
    state_size = config["state_size"]
    T = config["T"]
    learning_rate = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    wm = config["w_multiplier"]
    nmin, nmax = 1, 20
    perturb = config["mu_std"]
    
    weights_T = torch.ones(T)
    weights_T[0] = 0
    for i in range(1, T - 1):
        weights_T[i + 1] = weights_T[i] * wm
    weights_T = weights_T / weights_T.sum()
    
    l2o_net = L2OOptimizer(state_size, linear_size=linear_size)
    l2o_net.to(device)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)
    
    train_loss_buffer = []
    final_loss_buffer = []
    for epoch in range(num_epochs):
        n = int(torch.randint(nmin, nmax + 1, (1,))[0])
        hidden = l2o_net.get_init_states(batch_size, n).to(device)
        
        A = generate_random_quadratic(n, batch_size).to(device)
        mu = (torch.randn(batch_size, n) * perturb).to(device)
        
        x = torch.zeros(batch_size, n, device=device, requires_grad=True)
        
        init_loss = quadratic(x, A, mu)
        total_loss = 0.0
        final_loss_value = 0.0
        
        for j in range(T):
            f_val = quadratic(x, A, mu) / (init_loss + 1e-12)
            loss_step = f_val.mean()
            total_loss += loss_step * weights_T[j]
            
            if j == T - 1:
                final_loss_value = loss_step.item()
            
            if j < T - 1:
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
        
        avg_train_loss_20 = sum(train_loss_buffer) / len(train_loss_buffer)
        avg_final_loss_20 = sum(final_loss_buffer) / len(final_loss_buffer)
        
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

# Training function for single run
def train_l2o(config, create_output=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    linear_size = config["linear_size"]
    state_size = config["state_size"]
    T = config["T"]
    learning_rate = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    wm = config["w_multiplier"]
    nmin, nmax = 1, 20
    perturb = config["mu_std"]
    
    weights_T = torch.ones(T)
    weights_T[0] = 0
    for i in range(1, T - 1):
        weights_T[i + 1] = weights_T[i] * wm
    weights_T = weights_T / weights_T.sum()
    
    l2o_net = L2OOptimizer(state_size, linear_size=linear_size)
    l2o_net.to(device)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)
    
    train_loss_buffer = []
    final_loss_buffer = []
    if create_output:
        outfilename = "training_trajectory_T=%d_wM=%.2f.npz" % (T, wm)
        epochs = []
        train_loss = []
    
    for epoch in range(num_epochs):
        n = int(torch.randint(nmin, nmax + 1, (1,))[0])
        hidden = l2o_net.get_init_states(batch_size, n).to(device)
        
        A = generate_random_quadratic(n, batch_size).to(device)
        mu = (torch.randn(batch_size, n) * perturb).to(device)
        
        x = torch.zeros(batch_size, n, device=device, requires_grad=True)
        
        init_loss = quadratic(x, A, mu)
        total_loss = 0.0
        final_loss_value = 0.0
        
        for j in range(T):
            f_val = quadratic(x, A, mu) / (init_loss + 1e-12)
            loss_step = f_val.mean()
            total_loss += loss_step * weights_T[j]
            
            if j == T - 1:
                final_loss_value = loss_step.item()
            
            if j < T - 1:
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
        
        avg_train_loss_20 = sum(train_loss_buffer) / len(train_loss_buffer)
        avg_final_loss_20 = sum(final_loss_buffer) / len(final_loss_buffer)
        
        if create_output:
            epochs.append(epoch)
            train_loss.append(current_train_loss)
            np.savez(outfilename, epochs=epochs, train_loss=train_loss)

# Run L2O on a new problem
def run_l2o_on_new_problem(l2o_net, A, mu, T=10, device="cpu"):
    l2o_net.to(device)
    A, mu = A.to(device), mu.to(device)
    
    batch_size, n, _ = A.shape
    x = torch.zeros(batch_size, n, device=device, requires_grad=True)
    
    hidden = l2o_net.get_init_states(batch_size, n).to(device)
    
    init_loss = quadratic(x, A, mu)
    losses = []
    
    with torch.no_grad():
        for step in range(T):
            vals = quadratic(x, A, mu)
            loss = (vals / (init_loss + 1e-12)).mean()
            losses.append(loss.item())
            
            gradient = quadratic_grad(x, A, mu)
            grad = gradient / (init_loss.reshape(-1, 1) + 1e-12)
            
            x, hidden, _ = l2o_net(x, grad, hidden)
    return x, losses

# Main execution
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    T = int(sys.argv[1])
    w_multiplier = float(sys.argv[2])
    config = {
        "linear_size": tune.choice([4, 8, 16, 32]),
        "state_size": tune.choice([32, 65, 128, 256]),
        "T": T,
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "num_epochs": 200,
        "mu_std": 10,
        "w_multiplier": w_multiplier,
    }
    
    # Run tuning
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
        num_samples=128,
        scheduler=scheduler,
        verbose=1
    )
    
    best_trial = analysis.get_best_trial(metric="final_loss_20avg", mode="min")
    best_checkpoint = analysis.get_best_checkpoint(
        best_trial, metric="final_loss_20avg", mode="min"
    )
    best_config = best_trial.config
    with open("best_config_T=%d_wM=%.2f.json" % (T, w_multiplier), "w") as f:
        json.dump(best_config, f, indent=2)
    
    if best_checkpoint is not None:
        best_checkpoint_dir = best_checkpoint.to_directory()
        with open(os.path.join(best_checkpoint_dir, "model_state_dict.pkl"), "rb") as f:
            best_state = pickle.load(f)
            torch.save(best_state, "best_l2o_model_T=%d_wM=%.2f.pth" % (T, w_multiplier))
    
    # Run training with best config
    print("Done with tuning, loading best model to produce training trajectory")
    with open("best_config_T=%d_wM=%.2f.json" % (T, w_multiplier), "r") as f:
        best_config = json.load(f)
    train_l2o(config=best_config, create_output=True)