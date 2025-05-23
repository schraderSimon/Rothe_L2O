import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import minimize
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from calculate_RE_and_grad import make_error_and_gradient_functions
torch.set_default_dtype(torch.float64)  # Rothe's method NEEDS double precision
import sys
file_exists = lambda path: os.path.isfile(path)

# ──────────────────────────────────────────────────────────────────────────────
#  Check-/re-load helpers
# ──────────────────────────────────────────────────────────────────────────────
import json, pathlib, datetime, torch, os

def _run_directory(cfg: dict) -> pathlib.Path:
    """Create /runs/<tag>/ where <tag> now also includes L1 and L2."""

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
    run_dir = pathlib.Path("runs") / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))  
    return run_dir

def save_checkpoint(epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    avg20_train_error: float,
                    test_error: float,
                    cfg: dict,
                    root="runs",
                    save_params=False):
    """Called every 20 epochs."""
    run_dir = _run_directory(cfg)
    fname   = run_dir / f"ep{epoch:03d}.pt"
    if save_params:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "avg20_train_error": avg20_train_error,
            "test_error": test_error,
            "config": cfg,
        }, fname)

    else:
        torch.save({
            "epoch": epoch,
            #"model_state": model.state_dict(), #Too much memory need, we don't need the model anyways (can always rerun the best model)
            #"optim_state": optimizer.state_dict() #Too much memory need, we don't need the optimizer anyways
            "avg20_train_error": avg20_train_error,
            "test_error": test_error,
            "config": cfg,
        }, fname)

    # also append one-line CSV for quick plotting
    with open(run_dir / "log.csv", "a", encoding="utf-8") as fh:
        fh.write(f"{epoch},{avg20_train_error:.6e},{test_error:.6e}\n")

    return fname


class EvaluateFunction(torch.autograd.Function):
    """
    This workaround was 100% stolen from ChatGPT and I do not claim that I know how it works.
    """
    @staticmethod
    def forward(ctx, parameters, optimizee, optimizee_grad):
        """
        Forward pass:
          - parameters: Tensor of shape (batch_size, param_len)
          - optimizee: Callable that takes a 1D numpy array and returns a scalar loss.
          - optimizee_grad: Callable that takes a 1D numpy array and returns its gradient.
        """
        parameters_np = parameters.detach().cpu().numpy()
        losses=np.zeros(len(parameters_np))
        for i,sample in enumerate(parameters_np):
            losses[i] = optimizee(sample)  # Scale the loss to avoid numerical issues.
        # Save parameters and the gradient function for the backward pass.
        ctx.save_for_backward(parameters)
        ctx.optimizee_grad = optimizee_grad
        return torch.tensor(losses, dtype=parameters.dtype, device=parameters.device)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass:
          - Retrieves the saved parameters and calls the saved optimizee_grad 
            for each sample.
          - Multiplies the resulting gradient with grad_output.
        """
        parameters, = ctx.saved_tensors
        optimizee_grad = ctx.optimizee_grad
        parameters_np = parameters.detach().cpu().numpy()
        grads = []
        for sample in parameters_np:
            grad_sample = optimizee_grad(sample)  # Scale the gradient to avoid numerical issues.
            grads.append(grad_sample)
        grads = np.array(grads)
        grad_tensor = torch.tensor(grads, dtype=parameters.dtype, device=parameters.device)
        # Multiply elementwise by grad_output (shape adjustment via view).
        grad_tensor = grad_tensor * grad_output.view(-1, 1)
        # Return gradients for each input: first is for parameters,
        # then None for optimizee and optimizee_grad since they are not trainable.
        return grad_tensor, None, None
class BatchEvaluate(torch.autograd.Function):
    """Vectorised version of EvaluateFunction for heterogeneous optimisees."""
    @staticmethod
    def forward(ctx, parameters, optimisees, optimisee_grads):
        """
        parameters      : (B, P) tensor  (all problems share the same P)
        optimisees      : list of callables length B   (loss)
        optimisee_grads : list of callables length B   (∇loss)
        """
        parameters_np = parameters.detach().cpu().numpy()
        losses, grads = [], []
        # evaluate each problem *once*
        for p, f, g in zip(parameters_np, optimisees, optimisee_grads):
            loss = f(p)
            grad = g(p)
            losses.append(loss)
            grads.append(grad)
        losses_t = torch.tensor(losses, dtype=torch.float64, device=parameters.device)
        grads_t  = torch.tensor(grads,  dtype=torch.float64, device=parameters.device)
        ctx.save_for_backward(grads_t)
        return losses_t

    @staticmethod
    def backward(ctx, g_out):           # g_out has shape (B,)
        grads_t, = ctx.saved_tensors    # (B, P)
        g_out = g_out.view(-1, 1)       # (B, 1)
        # chain rule:   dL/dθ_b = g_out_b · stored_grad_b
        return grads_t * g_out, None, None

class L2OOptimizer(nn.Module):
    def __init__(self, hidden_size, linear_size=10, num_layers=1,scale=[-10.0,-10.0,-10.0,-10.0]):
        super(L2OOptimizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable hidden state
        self.c_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable cell state
        self.scale=((torch.tensor(scale)))  # Learnable scale parameter, initialized to 1e-5
        #self.scale=torch.tensor(scale)
        self.initial_transform = nn.Sequential(OrderedDict([ # Initial non-linear transformation
            ("linear1", nn.Linear(8, linear_size)),
            ("nonlin1", nn.ReLU()),
            ("linear2", nn.Linear(linear_size, linear_size)),
        ]))

        self.lstm_cells = nn.ModuleList() # List of LSTM cells
        self.lstm_cells.append(nn.LSTMCell(input_size=linear_size, hidden_size=hidden_size)) #First LSTM cell, input size is linear_size, hidden size is hidden_size

        for _ in range(1, num_layers):
            self.lstm_cells.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)) #Add num_layers-1 LSTM cells, input size is hidden_size, hidden size is hidden_size
        
        self.fc = nn.Linear(hidden_size, 4) #Final linear layer to output the perturbation
    def forward(self, x, grad, hidden_LSTM):
        batch_size, n = x.shape
        nd4=n//4
        new_hidden_h_layers = torch.zeros(batch_size, nd4, self.num_layers, self.hidden_size, device=x.device) #Array containing the new hidden states after forward pass
        new_hidden_c_layers = torch.zeros(batch_size, nd4, self.num_layers, self.hidden_size, device=x.device) #Array containing the new cell states after forward pass

        x_new=torch.zeros(batch_size, n) #Array containing the new optimizee
        delta=torch.zeros(batch_size, n) #Update in x in the forward pass

        grad_expanded_divided_by_gaussians=grad.reshape(batch_size,nd4,4)
        gradient_signs = grad_expanded_divided_by_gaussians.sign()
        gradient_magnitudes=torch.log(torch.abs(grad_expanded_divided_by_gaussians)+1e-16)
        inp_all=torch.cat([gradient_signs, gradient_magnitudes], dim=2)  # (batch_size, n//4,8)
        inp_transformed = self.initial_transform(inp_all)  # (batch_size, n//4, linear_size)
        inp_layer = inp_transformed # The first input to the LSTM is just the transformed input.
        for layer in range(self.num_layers):
            h_i = hidden_LSTM[0][:, :, layer, :]  
            c_i = hidden_LSTM[1][:, :, layer, :]  

            # Flatten batch_size and n into a single dimension for LSTM input
            # This essentially turns it into an "extra large batch" of size (batch_size * n)
            inp_reshaped=inp_layer.view(batch_size * nd4, -1) # view is more efficient than reshape
            h_i_reshaped=h_i.view(batch_size * nd4, -1)
            c_i_reshaped=c_i.view(batch_size * nd4, -1)
            #This line here causes issues
            h_new, c_new = self.lstm_cells[layer](inp_reshaped, (h_i_reshaped,  c_i_reshaped))
            h_new = h_new.view(batch_size, nd4, -1)
            c_new = c_new.view(batch_size, nd4, -1)

            new_hidden_h_layers[:, :, layer, :] = h_new # Store the new hidden state
            new_hidden_c_layers[:, :, layer, :] = c_new # Store the new cell state
            inp_layer = h_new   # Update the input for the next layer

        output=self.fc(h_new) # Final output of the LSTM
        
        delta = output.reshape(batch_size, n) # Reshape to match the original input shape
        ngauss=delta.shape[1]//4
        scale_exp = self.scale.repeat(ngauss)          # (n,) – still linked to the Parameter
        delta_scaled = delta *torch.exp(scale_exp)    # broadcasting keeps the same result
        x_new = x + delta_scaled #Factor to ensure that the output is small
        return x_new, (new_hidden_h_layers, new_hidden_c_layers), delta


    def get_init_states(self, batch_size, n):
        h_0 = self.h_0[None,None,:,:].repeat(batch_size, n//4, 1, 1).contiguous()
        c_0 = self.c_0[None,None,:,:].repeat(batch_size, n//4, 1, 1).contiguous()
        return (h_0, c_0)




# Convenience wrapper so the call site looks the same as before
def evaluate_batch(params, optimisees, optimisee_grads):
    return BatchEvaluate.apply(params, optimisees, optimisee_grads)


# ------------------------------------------------------------------------
# 2.  A helper that draws a *valid* batch of realistic problems
# ------------------------------------------------------------------------
def sample_realistic_batch(batch_size, t_pool, E0, quality):
    """
    Returns
        parameters0    : (B, P) float64 tensor on CPU
        optimisees     : list length B
        optimisee_grads: list length B
        init_losses    : (B,) float64 tensor on CPU
    """
    while True:
        # pick a starting time far enough from the upper bound
        t0 = np.random.choice(t_pool[: len(t_pool) - batch_size])
        times = t0 + 0.2 * np.arange(batch_size)

        triples = [make_error_and_gradient_functions(E0, quality, t) for t in times]
        lens    = [len(tr[2]) for tr in triples]

        if len(set(lens)) == 1:                 # all equal → we have a valid batch
            P = lens[0]
            params0    = np.stack([tr[2] for tr in triples])     # (B, P)
            optimisees = [tr[0] for tr in triples]
            opt_grads  = [tr[1] for tr in triples]
            init_losses = np.array([f(p) for f, p in zip(optimisees, params0)])

            return (times,torch.tensor(params0, dtype=torch.float64),
                    optimisees, opt_grads,
                    torch.tensor(init_losses, dtype=torch.float64))


# ------------------------------------------------------------------------
# 3.  The training loop – now *really* batched
# ------------------------------------------------------------------------
def train_l2o_realistic(config,quality=1, E0=0.06,save_params=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    linear_size   = config["linear_size"]
    hidden_size   = config["hidden_size"]
    num_layers    = config["num_layers"]
    T             = config["T"]
    lr            = config["lr"]
    num_epochs    = config["num_epochs"]
    batch_size    = config["batch_size"]
    wm            = config["w_multiplier"]
    tmin= config["tmin"]
    tmax= config["tmax"]
    tmin_test= config["tmin_test"]
    tmax_test= config["tmax_test"]
    L1L2          = (config["l1"], config["l2"])

    # geometric weights over the unrolled steps
    weights_T = torch.ones(T, device=device)
    weights_T[0] = 0.
    for i in range(1, T - 1):
        weights_T[i+1] = weights_T[i] * wm
    weights_T /= weights_T.sum()

    l2o = L2OOptimizer(hidden_size, linear_size=linear_size, num_layers=num_layers).to(device)
    opt = optim.Adam(l2o.parameters(), lr=lr)
    start_epoch = 0
    # pool of candidate times (100 ≤ t < 200 in 0.2‑s steps)
    t_pool = np.arange(tmin, tmax, 0.2)
    avg_20=[]
    total_losses=[]
    test_losses=[]
    
    for epoch in range(start_epoch, num_epochs):
        # ------------------------------------------------------------------
        # (a) realistic batch
        # ------------------------------------------------------------------
        (times,parameters, optimisees, opt_grads,
         init_losses) = sample_realistic_batch(batch_size, t_pool, E0, quality)

        parameters  = parameters.to(device)                          # (B, P)
        init_losses = init_losses.to(device)                         # (B,)

        hidden = l2o.get_init_states(batch_size, parameters.size(1))
        hidden = (hidden[0].to(device), hidden[1].to(device))

        total_loss, final_loss_value = 0., 0.

        # ------------------------------------------------------------------
        # (b) unroll T steps
        # ------------------------------------------------------------------
        for j in range(T):
            if j == 0:
                f_val = init_losses / (init_losses + 1e-16)
            else:
                curr_losses = evaluate_batch(parameters,
                                             optimisees, opt_grads)
                f_val = curr_losses / (init_losses + 1e-16)

            loss_step  = f_val.mean()
            total_loss = total_loss + (loss_step) * weights_T[j]

            if j == T - 1:
                final_loss_value = loss_step.item()

            # ----- meta‑gradient step (skip on the last unroll step) -------
            if j < T - 1:
                grads_t = torch.stack([
                    torch.tensor(g(parameters[i].detach().cpu().numpy()),
                                 dtype=torch.float64, device=device)
                    for i, g in enumerate(opt_grads)
                ])
                # normalise by the initial loss of *each* problem
                grad_norm = grads_t #/ (init_losses.view(-1, 1) + 1e-16)

                parameters, hidden, _ = l2o(parameters, grad_norm, hidden)
        lambda_l1,lambda_l2=L1L2
        l1_reg = 0.0
        l2_reg = 0.0
        for name, p in l2o.named_parameters():
            if "weight" in name:
                l1_reg += p.abs().sum()
                l2_reg += p.pow(2).sum()
        avg_20.append(final_loss_value)
        if len(avg_20)>20:
            avg_20.pop(0)
        avg_20_loss=np.mean(avg_20)
        print("Loss due to  L1/L2 regularization: %.3f/%.3f"%(lambda_l1*l1_reg.item(),lambda_l2*l2_reg.item()))
        total_losses.append(total_loss.item()) #The total loss without regularization
        total_loss = total_loss + lambda_l1 * l1_reg + lambda_l2 * l2_reg
        # ------------------------------------------------------------------
        # (c) optimise the L2O network
        # ------------------------------------------------------------------
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        print(f"[epoch {epoch:4d}] Time {times[0]:.1f}, nGauss: {parameters.shape[1]//4}, mean final‑to‑initial loss ratio = "
              f"{final_loss_value:6.4f}, average over 20: {avg_20_loss:6.4f}   (batch avg over {batch_size})")
        if epoch % 20 == 0:
            test_times=np.array(np.linspace(tmin_test,tmax_test,20),dtype=int)
            mean_ratio = evaluate_test_error(l2o, config,E0=E0, quality=quality,test_times=test_times, device=device)
            test_losses.append(mean_ratio)
            print(f"Mean final-to-initial loss ratio on test times: {mean_ratio:.6f}")

            # ------------------------------------------------------------------
            # NEW: save everything every 20th epoch
            # ------------------------------------------------------------------
            avg20 = avg_20_loss
            save_checkpoint(epoch, l2o, opt, avg20, mean_ratio, config,save_params=save_params)

@torch.no_grad()
def evaluate_test_error(
        l2o,
        config,
        test_times,   # 210, 220, …, 330
        E0=0.06,
        quality=1,
        device=None,
):
    """
    Evaluate the trained L2O network on a fixed grid of time points.
    Every problem is solved *individually* (no batching), so the varying
    number of Gaussians never causes shape clashes.
    Returns the mean final-to-initial loss ratio across all test times.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    l2o.eval()

    T       = config["T"]
    ratios  = []

    for t in test_times:
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
                f_val = init_loss_t / (init_loss_t + 1e-16)   # == 1
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
        ratios.append(f_val.item())

    l2o.train()      # restore training mode for continuing epochs
    return float(np.mean(ratios))
if __name__ == "__main__":
    T = 10
    w_multiplier = 1
    seed= 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    quality=2
    E0=0.06
    tmin=100
    tmax=200
    tmin_test=210
    tmax_test=330
    num_layers_considered=[3]
    lr_considered=[1e-3,3e-3,1e-2]
    l2_considered=[1e-3,1e-4,1e-5,1e-2]
    batchsize_considered=[1,2]
    sizes_considered=[256]
    # Define the configuration for hyperparameter tuning
    counter=0
    for i in range(len(lr_considered)):
        lr=lr_considered[i]
        for j in range(len(batchsize_considered)):
            batchsize=batchsize_considered[j]
            for k in range(len(sizes_considered)):
                size=sizes_considered[k]
                for m in range(len(l2_considered)):
                    counter+=1
                    l2=l2_considered[m]
                    config = {
                        "linear_size": size,
                        "hidden_size": size,
                        "num_layers":  3,
                        "T":           T,
                        "lr":          lr,
                        "batch_size":  batchsize,
                        "num_epochs":  500,
                        "w_multiplier": 1,
                        "l2":          l2,
                        "l1":          0,
                        "tmin":        tmin,
                        "tmax":        tmax,
                        "tmin_test": tmin_test,
                        "tmax_test": tmax_test,
                    }
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
                        if "ep480.pt" in os.listdir(mapname):
                            print("Configuration %d: size=%d, batchsize=%d, lr=%.3e, l2=%.3e already trained"%(counter,size,batchsize,lr,l2))
                            continue
                    print("--------------------------------------------------------------------------------")
                    print("Configuration %d: size=%d, batchsize=%d, lr=%.3e, l2=%.3e"%(counter,size,batchsize,lr,l2))
                    print("--------------------------------------------------------------------------------")
                    train_l2o_realistic(config,quality=quality, E0=E0)
    