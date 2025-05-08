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
def train_l2o(config, create_output=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    linear_size   = config["linear_size"]
    hidden_size   = config["hidden_size"]
    num_layers    = config["num_layers"]
    T             = config["T"]
    learning_rate = config["lr"]
    num_epochs    = config["num_epochs"]
    batch_size    = config["batch_size"]
    wm            = config["w_multiplier"]

    weights_T = torch.ones(T, device=device)
    weights_T[0] = 0
    for i in range(1, T - 1):
        weights_T[i+1] = weights_T[i] * wm
    weights_T = weights_T / weights_T.sum()

    l2o_net = L2OOptimizer(hidden_size, linear_size=linear_size, num_layers=num_layers)
    l2o_net.to(device)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)

    train_loss_buffer = []
    final_loss_buffer = []
    if create_output:
        epochs = []
        train_loss = []

    tvals = np.arange(100, 200, 0.2)
    E0      = 0.03
    for epoch in range(num_epochs):
        t = np.random.choice(tvals)
        t0      = t
        
        quality = 1 #Low quality

        # Get your black-box evaluate function, its gradient, and the initial parameters.
        optimizee, optimizee_grad, initial_parameters = make_error_and_gradient_functions(E0, quality, t0)
        loss0= optimizee(initial_parameters)
        grad0=optimizee_grad(initial_parameters)
        hess_inv0=np.diag(1/(abs(grad0)+1e-16))
        solution=minimize(optimizee, initial_parameters, method='BFGS', jac=optimizee_grad,options={"gtol":1e-14,"maxiter":T,"hess_inv0":hess_inv0})
        bfgs_solution_short=solution.fun
        solution= minimize(optimizee, initial_parameters, method='BFGS', jac=optimizee_grad,options={"gtol":1e-14,"maxiter":100,"hess_inv0":hess_inv0})
        bfgs_solution=solution.fun
        
        param_len = len(initial_parameters)
        bfgs_reduction= (bfgs_solution) / (loss0 + 1e-16)
        bfgs_short= (bfgs_solution_short) / (loss0 + 1e-16)
        hidden = l2o_net.get_init_states(batch_size, param_len)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        # Convert initial parameters to a tensor and duplicate them for the batch.
        initial_parameters = torch.tensor(initial_parameters, dtype=torch.float64, device=device)
        initial_parameters = initial_parameters.repeat(batch_size, 1)
        initial_parameters = initial_parameters + torch.randn(batch_size, param_len, device=device) * 1e-4 # Add noise
        total_loss = 0.0
        final_loss_value = 0.0

        # Compute the initial loss via the custom EvaluateFunction.
        initial_loss = EvaluateFunction.apply(initial_parameters, optimizee, optimizee_grad)
        print("Initial loss:", initial_loss)
        parameters = initial_parameters.clone()
        for j in range(T):
            if j == 0:
                f_val = initial_loss / (initial_loss + 1e-16)
            else:
                current_loss = EvaluateFunction.apply(parameters, optimizee, optimizee_grad)
                f_val = current_loss / (initial_loss + 1e-16)
            loss_step = f_val.mean()
            total_loss = total_loss + loss_step * weights_T[j]

            if j == T - 1:
                final_loss_value = loss_step.item()

            if j < T - 1:
                grads_list = []
                for i in range(batch_size):
                    grad_i = optimizee_grad(parameters[i, :].detach().cpu().numpy())
                    grads_list.append(grad_i)
                grads_tensor = torch.tensor(grads_list, dtype=torch.float64, device=device)
                #grad_ = grads_tensor / (initial_loss.view(-1, 1) + 1e-16) / parameters.shape[0]
                grad_ = grads_tensor / (initial_loss.view(-1, 1) + 1e-16) / parameters.shape[0]
                # The L2O network returns updated parameters, new hidden state, and (possibly) additional info.
                parameters, hidden, _ = l2o_net(parameters, grad_, hidden)

        print("Final loss value: at iteration %d: %.3f. BFGS: %.3f. BFGS (T iterations): %.3f"%(epoch, final_loss_value, bfgs_reduction,bfgs_short))
        lambda_l1=0
        lambda_l2=0
        l1_reg = 0.0
        l2_reg = 0.0
        for name, p in l2o_net.named_parameters():
            if "weight" in name:
                l1_reg += p.abs().sum()
                l2_reg += p.pow(2).sum()
            if "scale" in name:
                print("Scale parameter: %.3f"%(p.item()))
        print("Total loss contribution: %.3f. L1: %.3f. L2: %.3f"%(total_loss.item(), lambda_l1*l1_reg.item(), lambda_l2*l2_reg.item()))
        total_loss = total_loss + lambda_l1 * l1_reg+ lambda_l2 * l2_reg
        # -------------------------------------------------------------

        optimizer.zero_grad()
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(l2o_net.parameters(), 0.1)
        optimizer.step()




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
def train_l2o_realistic(config, create_output=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    linear_size   = config["linear_size"]
    hidden_size   = config["hidden_size"]
    num_layers    = config["num_layers"]
    T             = config["T"]
    lr            = config["lr"]
    num_epochs    = config["num_epochs"]
    batch_size    = config["batch_size"]
    wm            = config["w_multiplier"]

    # geometric weights over the unrolled steps
    weights_T = torch.ones(T, device=device)
    weights_T[0] = 0.
    for i in range(1, T - 1):
        weights_T[i+1] = weights_T[i] * wm
    weights_T /= weights_T.sum()

    l2o = L2OOptimizer(hidden_size, linear_size=linear_size, num_layers=num_layers).to(device)
    opt = optim.Adam(l2o.parameters(), lr=lr)

    # pool of candidate times (100 ≤ t < 200 in 0.2‑s steps)
    t_pool = np.arange(50, 150., 0.2)
    E0, quality = 0.06, 1

    for epoch in range(num_epochs):
        #E0=np.random.choice([0.03,0.06,0.12])
        #quality= np.random.choice([1,2,3])
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
        lambda_l1=0
        lambda_l2=1e-4
        l1_reg = 0.0
        l2_reg = 0.0
        for name, p in l2o.named_parameters():
            if "weight" in name:
                l1_reg += p.abs().sum()
                l2_reg += p.pow(2).sum()
                print(name)
            if "scale" in name:
                print(p)
        print("Loss due to L2 regularization: %.3f"%(lambda_l2*l2_reg.item()))
        total_loss = total_loss + lambda_l1 * l1_reg + lambda_l2 * l2_reg
        # ------------------------------------------------------------------
        # (c) optimise the L2O network
        # ------------------------------------------------------------------
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        print(f"[epoch {epoch:4d}] Time {times[0]:.1f}, mean final‑to‑initial loss ratio = "
              f"{final_loss_value:6.4f}   (batch avg over {batch_size})")
if __name__ == "__main__":
    # Example usage
    T = 5
    w_multiplier = 1
    seed= 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Define the configuration for hyperparameter tuning
    config = {
            "linear_size": 64,
            "hidden_size": 64,
            "num_layers":  3,
            "T":           T,
            "lr":          0.01,
            "batch_size":  8,
            "num_epochs":  500,
            "w_multiplier": w_multiplier,
        }

    train_l2o_realistic(config, create_output=True)