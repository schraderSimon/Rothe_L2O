import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import minimize
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import calculate_RE_and_grad
from calculate_RE_and_grad import make_error_and_gradient_functions
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
            losses[i] = optimizee(sample)
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
            grad_sample = optimizee_grad(sample)  # Should return an array with same shape as sample.
            grads.append(grad_sample)
        grads = np.array(grads)
        grad_tensor = torch.tensor(grads, dtype=parameters.dtype, device=parameters.device)
        # Multiply elementwise by grad_output (shape adjustment via view).
        grad_tensor = grad_tensor * grad_output.view(-1, 1)
        # Return gradients for each input: first is for parameters,
        # then None for optimizee and optimizee_grad since they are not trainable.
        return grad_tensor, None, None
class L2OOptimizer(nn.Module):
    def __init__(self, hidden_size, linear_size=10, num_layers=1):
        super(L2OOptimizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable hidden state
        self.c_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable cell state
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
        grad_expanded=grad[:,:,None] #Reshaping for numerical reasons
        #size: (batch_size, n, 1)

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
        x_new = x + delta*1e-5 #Factor of 1e-5 to ensure that the perturbation is small
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

    tvals = np.arange(50, 150, 0.2)
    E0      = 0.06
    for epoch in range(num_epochs):
        t = np.random.choice(tvals)
        t0      = t
        
        quality = 3

        # Get your black-box evaluate function, its gradient, and the initial parameters.
        optimizee, optimizee_grad, initial_parameters = make_error_and_gradient_functions(E0, quality, t0)
        param_len = len(initial_parameters)
        hidden = l2o_net.get_init_states(batch_size, param_len)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        # Convert initial parameters to a tensor and duplicate them for the batch.
        initial_parameters = torch.tensor(initial_parameters, dtype=torch.float32, device=device)
        initial_parameters = initial_parameters.repeat(batch_size, 1)
        initial_parameters = initial_parameters + torch.randn(batch_size, param_len, device=device) * 1e-5 # Add noise
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
                grads_tensor = torch.tensor(grads_list, dtype=torch.float32, device=device)
                grad_ = grads_tensor / (initial_loss.view(-1, 1) + 1e-16) / parameters.shape[0]
                # The L2O network returns updated parameters, new hidden state, and (possibly) additional info.
                parameters, hidden, _ = l2o_net(parameters, grad_, hidden)

        print("Final loss value: at iteration %d: "%epoch, final_loss_value)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(l2o_net.parameters(), 0.1)
        optimizer.step()
if __name__ == "__main__":
    # Example usage
    T = 10
    w_multiplier = 1

    # Define the configuration for hyperparameter tuning
    config = {
            "linear_size": 32,
            "hidden_size": 256,
            "num_layers":  3,
            "T":           T,
            "lr":          0.00154,
            "batch_size":  8,
            "num_epochs":  200,
            "w_multiplier": w_multiplier,
        }

    train_l2o(config, create_output=True)