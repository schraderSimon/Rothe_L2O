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
    def __init__(self, hidden_size, linear_size=10, num_layers=1):
        """
        The network acts on one variable at a time.
        hidden_size: number of hidden units in each LSTM cell.
        num_layers: number of stacked LSTM layers.
        """
        super(L2OOptimizer, self).__init__()
        self.num_layers = num_layers
        # A simple transformation of the 2D input (variable and its gradient).
        self.initial_transform = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(2, linear_size)),
            ("nonlin1", nn.Tanh()),
            ("linear2", nn.Linear(linear_size, linear_size)),
        ]))
        # Create a list of LSTM cells. The module list makes sure that the LSTM cells are registered as submodules - a normal list would not do that.
        self.lstm_cells = nn.ModuleList()
        # First layer takes the transformed input.
        self.lstm_cells.append(nn.LSTMCell(input_size=linear_size, hidden_size=hidden_size))
        # Subsequent layers take the hidden state from the previous layer.
        for _ in range(1, num_layers):
            self.lstm_cells.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))
        # A linear layer that projects the final LSTM hidden state to a scalar update.
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, grad, hidden_LSTM):
        """
        x: tensor of shape (batch_size, n)
        grad: tensor of shape (batch_size, n)
        hidden_LSTM: tuple of lists, each list of length num_layers containing tensors of shape (batch_size, n, hidden_size)
                     (first element of the tuple is the hidden states, second is the cell states)
        """
        batch_size, n = x.shape
        # Prepare list of empty lists to store the new hidden and cell states for each layer.
        new_hidden_h_layers = [[] for k in range(self.num_layers)]
        new_hidden_c_layers = new_hidden_h_layers.copy()
        x_new_list = []    # List to store updated variables.
        delta_list = []    # List to store deltas.

        # Process each variable independently.
        for i in range(n):
            # Extract the i-th variable and its gradient.
            x_i = torch.reshape(x[:, i], (-1, 1))         # shape: (batch_size, 1)
            grad_i = torch.reshape(grad[:, i], (-1, 1))     # shape: (batch_size, 1)
            # Prepare the input for the first LSTM layer.
            gradient_sign = grad_i.sign()
            epsilon = 1e-14
            gradient_magnitude = torch.log(torch.abs(grad_i) + epsilon)
            inp = torch.cat([gradient_sign, gradient_magnitude], dim=1)  # shape: (batch_size, 2). "Cat" is pytorch's concatenation function.
            inp_transformed = self.initial_transform(inp)  # shape: (batch_size, linear_size)
            
            # Propagate through the stacked LSTM layers.
            inp_layer = inp_transformed
            new_h = None
            for layer in range(self.num_layers):
                # Get the corresponding hidden and cell state for this variable and layer.
                h_i = hidden_LSTM[0][layer][:, i, :]  # shape: (batch_size, hidden_size)
                c_i = hidden_LSTM[1][layer][:, i, :]  # shape: (batch_size, hidden_size)
                h_new, c_new = self.lstm_cells[layer](inp_layer, (h_i, c_i))
                new_hidden_h_layers[layer].append(h_new[:,None,:])  # shape: (batch_size, 1, hidden_size)
                new_hidden_c_layers[layer].append(c_new[:,None,:])  # shape: (batch_size, 1, hidden_size)
                # For next layer, input is the hidden state from current layer.
                inp_layer = h_new
                new_h = h_new  # Final layer's output.
            
            # Compute the update delta from the final layer's hidden state.
            delta = self.fc(new_h)  # shape: (batch_size, 1)
            x_new = x_i + delta     # shape: (batch_size, 1)
            
            x_new_list.append(x_new)
            delta_list.append(delta)
        
        # Concatenate along the variable dimension.
        new_hidden_h = [torch.cat(new_hidden_h_layers[layer], dim=1) for layer in range(self.num_layers)]
        new_hidden_c = [torch.cat(new_hidden_c_layers[layer], dim=1) for layer in range(self.num_layers)]
        new_hidden = (new_hidden_h, new_hidden_c)
        x_new = torch.cat(x_new_list, dim=1)
        delta = torch.cat(delta_list, dim=1)
        
        return x_new, new_hidden, delta

def main():
    # Hyperparameters.
    batch_size = 64   
    nmin = 1 # Minimal dimension of the function
    nmax =20 # Maximal dimension of the function
    linear_size = 256
    hidden_size = 128  # Hidden dimension for each LSTM cell.
    num_layers =  2     # Number of stacked LSTM layers.
    T = 10              # Number of inner optimization steps per instance.
    num_epochs = 20000
    learning_rate = 1e-4
    A = 10             # Amplitude of the Rastrigin function.
    perturb = 10      # Standard deviation of the random perturbation of the Rastrigin function.
    
    # Create weights for the loss accumulated over T steps.
    weights_T = torch.ones(T)
    weights_T[0] = 0
    for i in range(1, T-1):
        weights_T[i+1] = weights_T[i]*1.2
    weights_T = weights_T / weights_T.sum()
    
    # Initialize the L2O network.
    l2o_net = L2OOptimizer(hidden_size, num_layers=num_layers,linear_size=linear_size)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        lowest = torch.min(torch.tensor([int(epoch/10) + nmin + 1, nmax]))
        n = int(torch.randint(low=nmin, high=lowest, size=(1,))[0])
        # Initialize a separate hidden state per variable for each LSTM layer.
        hidden_init_h = [torch.zeros(batch_size, n, hidden_size) for _ in range(num_layers)]
        hidden_init_c = [torch.zeros(batch_size, n, hidden_size) for _ in range(num_layers)]
        hidden = (hidden_init_h, hidden_init_c)
        optimizer.zero_grad()
        
        # For each instance, sample a random shift vector s ~ N(0, perturb).
        A=generate_random_quadratic(n, batch_size)
        mu=torch.randn(batch_size, n)*perturb
        # The initial guess is x_0 = 0.
        x = torch.zeros(batch_size, n, requires_grad=True)
        
        total_loss = 0.0
        initial_loss = quadratic(x, A, mu).mean()
        final_loss = 0.0
        
        # Unroll the inner optimization for T steps.
        for j in range(T):
            f_val = quadratic(x, A, mu)
            loss_step = f_val.mean()
            total_loss = total_loss + loss_step * weights_T[j]
            if j == T-1:
                final_loss = loss_step
            if j < T-1:
                grad_of_optim = torch.autograd.grad(loss_step, x, create_graph=True)[0]
                x, hidden, delta = l2o_net(x, grad_of_optim, hidden)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, n {n}, Loss (in percent): {(total_loss.mean()/initial_loss.mean()):.3e}, Initial Loss: 1, Final Loss (in percent): {(final_loss.mean()/initial_loss.mean()):.3e}")
        total_loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
