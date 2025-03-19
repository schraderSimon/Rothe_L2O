import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import OrderedDict
import torch.jit
import sys
# Define the shifted N-dimensional Rastrigin function.
def rastrigin(x, shift, A):
    """
    Computes the shifted Rastrigin function:
      f(x) = A*n + sum((x - shift)^2 - A*cos(2*pi*(x - shift)))
    x: tensor of shape (batch_size, n)
    shift: tensor of shape (batch_size, n)
    Returns: tensor of shape (batch_size,) containing function values.
    """
    n = x.shape[1]
    z = x - shift
    return A * n + torch.sum(z ** 2 - A * (torch.cos(2 * math.pi * z)), dim=1)
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

# Define the L2O network that will learn the update rule.
# This version acts on one variable at a time.
class L2OOptimizer(nn.Module):
    def __init__(self, hidden_size, linear_size=10, num_layers=1,p=10):
        super(L2OOptimizer, self).__init__()
        self.p=p
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable hidden state
        self.c_0 = nn.Parameter(torch.randn(self.num_layers, hidden_size))  # Learnable cell state
        self.initial_transform = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(2, linear_size)),
            ("nonlin1", nn.ReLU()),
            ("linear2", nn.Linear(linear_size, linear_size)),
        ]))

        self.lstm_cells = nn.ModuleList()
        self.lstm_cells.append(nn.LSTMCell(input_size=linear_size, hidden_size=hidden_size))

        for _ in range(1, num_layers):
            self.lstm_cells.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))
        
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, grad, hidden_LSTM):
        batch_size, n = x.shape

        new_hidden_h_layers = torch.zeros(batch_size, n, self.num_layers, self.hidden_size, device=x.device)
        new_hidden_c_layers = torch.zeros(batch_size, n, self.num_layers, self.hidden_size, device=x.device)

        x_new=torch.zeros(batch_size, n)
        delta=torch.zeros(batch_size, n)
        grad_expanded=grad[:,:,None]
        p_exp=torch.exp(torch.tensor(self.p, dtype=grad_expanded.dtype, device=grad_expanded.device))
        #mask = grad_expanded.abs() > 1/p_exp
        #gradient_signs=torch.where(mask,grad_expanded.sign(),p_exp*grad_expanded)
        #gradient_magnitudes=torch.where(mask,torch.log(grad_expanded.abs())/self.p,-torch.ones_like(grad_expanded))
        gradient_signs = grad_expanded.sign()
        gradient_magnitudes=torch.log(torch.abs(grad_expanded)+1e-16)
        inp_all=torch.cat([gradient_signs, gradient_magnitudes], dim=2)  # (batch_size, n, 2)
        inp_transformed = self.initial_transform(inp_all)  # (batch_size, n, linear_size)
        inp_layer = inp_transformed # The first input to the LSTM is the transformed input.
        for layer in range(self.num_layers):
            h_i = hidden_LSTM[0][:, :, layer, :]  
            c_i = hidden_LSTM[1][:, :, layer, :]  

            # Flatten batch_size and n into a single dimension for LSTM input
            # This essentially turns it into an "extra large batch" of size (batch_size * n)
            inp_reshaped=inp_layer.view(batch_size * n, -1) 
            h_i_reshaped=h_i.view(batch_size * n, -1)
            c_i_reshaped=c_i.view(batch_size * n, -1)
            h_new, c_new = self.lstm_cells[layer](inp_reshaped, (h_i_reshaped,  c_i_reshaped))

            # Reshape back to (batch_size, n, hidden_size)
            h_new = h_new.view(batch_size, n, -1)
            c_new = c_new.view(batch_size, n, -1)

            new_hidden_h_layers[:, :, layer, :] = h_new
            new_hidden_c_layers[:, :, layer, :] = c_new
            inp_layer = h_new    

        delta = self.fc(h_new)[:, :, 0]
        x_new = x + delta
        return x_new, (new_hidden_h_layers, new_hidden_c_layers), delta


    def get_init_states(self, batch_size, n):
        h_0 = self.h_0[None,None,:,:].repeat(batch_size, n, 1, 1).contiguous()
        c_0 = self.c_0[None,None,:,:].repeat(batch_size, n, 1, 1).contiguous()
        return (h_0, c_0)
def main():
    # Hyperparameters.
    batch_size = 64   
    nmin = 1 # Minimal dimension of the function
    nmax =20 # Maximal dimension of the function
    lowest=nmax
    linear_size = 64
    hidden_size = 128  # Hidden dimension for each LSTM cell.
    num_layers =  3     # Number of stacked LSTM layers.
    T = 10              # Number of inner optimization steps per instance.
    num_epochs = 20000
    learning_rate = 1e-3
    perturb = 1      # Standard deviation of the random perturbation of the Rastrigin function.
    p=10
    # Create weights for the loss accumulated over T steps.
    weights_T = torch.ones(T)
    weights_T[0] = 0
    for i in range(1, T-1):
        weights_T[i+1] = weights_T[i]
    weights_T = weights_T / weights_T.sum()
    
    # Initialize the L2O network.
    l2o_net = L2OOptimizer(hidden_size, num_layers=num_layers,linear_size=linear_size,p=p)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        #lowest = torch.min(torch.tensor([int(epoch/10) + nmin + 1, nmax]))
        n = int(torch.randint(low=nmin, high=lowest, size=(1,))[0])
        #hidden_init_h = [torch.zeros(batch_size, n, hidden_size) for _ in range(num_layers)]
        #hidden_init_c = [torch.zeros(batch_size, n, hidden_size) for _ in range(num_layers)]
        #hidden = (hidden_init_h, hidden_init_c)
        hidden = l2o_net.get_init_states(batch_size, n)
        optimizer.zero_grad()
        
        # For each instance, sample a random shift vector s ~ N(0, perturb).
        A=generate_random_quadratic(n, batch_size)
        mu=torch.randn(batch_size, n)*perturb
        # The initial guess is x_0 = 0.
        x = torch.zeros(batch_size, n, requires_grad=True)
        
        total_loss = 0.0
        full_init_loss = quadratic(x, A, mu)
        initial_loss =full_init_loss.mean()

        final_loss = 0.0
        
        # Unroll the inner optimization for T steps.
        for j in range(T):
            f_val = quadratic(x, A, mu)/initial_loss # Normalize the function value by the initial loss.
            loss_step = f_val.mean() # Average over the batch dimension.
            total_loss = total_loss + loss_step * weights_T[j] 
            if j == T-1:
                final_loss = loss_step
            if j < T-1:
                grad_of_optim = torch.autograd.grad(loss_step, x, create_graph=True)[0]
                x, hidden, delta = l2o_net(x, grad_of_optim, hidden)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, n {n}, Loss (in percent): {(total_loss.mean()):.3e}, Initial Loss: 1, Final Loss (in percent): {(final_loss.mean()):.3e}")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(l2o_net.parameters(), max_norm=1) 
        optimizer.step()

if __name__ == '__main__':
    main()
