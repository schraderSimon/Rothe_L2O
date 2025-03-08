import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import OrderedDict
import torch.jit
# Define the shifted N-dimensional Rastrigin function.
def rastrigin(x, shift,A):
    """
    Computes the shifted Rastrigin function:
      f(x) = 10*n + sum((x - shift)^2 - 10*cos(2*pi*(x - shift)))
    x: tensor of shape (batch_size, n)
    shift: tensor of shape (batch_size, n)
    Returns: tensor of shape (batch_size,) containing function values.
    """
    n = x.shape[1]
    z = x - shift
    return A * n + torch.sum(z ** 2 - A * (torch.cos(2 * math.pi * z)), dim=1)

# Define the L2O network that will learn the update rule.
# This version acts on one variable at a time.
class L2OOptimizer(nn.Module):
    def __init__(self, hidden_size):
        """
        The network acts on one variable at a time.
        hidden_size: number of hidden units in the RNN.
        """
        linear1_output=10
        linear2_output=10
        super(L2OOptimizer, self).__init__()
        # A simple transformation of the 2D input (variable and its gradient).
        self.initial_transform = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(2, linear1_output)),
            ("nonlin1", nn.Tanh()),
            ("linear2", nn.Linear(linear1_output, linear2_output)),
        ]))
        # The RNN cell now expects an input size of 2.
        self.rnn = nn.RNNCell(input_size=10, hidden_size=hidden_size)
        # A linear layer that projects the RNN hidden state to a scalar update.
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, grad, hidden_RNN):
        """
        x: tensor of shape (batch_size, n)
        grad: tensor of shape (batch_size, n)
        hidden_RNN: tensor of shape (batch_size, n, hidden_size)
        """
        batch_size, n = x.shape
        new_hidden_list = [] #A list of updated hidden states
        x_new_list = []      #A list of updated variables
        delta_list = []      #A list of deltas

        # Process each variable independently
        for i in range(n):
            # Extract the i-th variable and its gradient.
            x_i = x[:, i].unsqueeze(1)         # shape: (batch_size, 1)
            grad_i = grad[:, i].unsqueeze(1)     # shape: (batch_size, 1)
            h_i = hidden_RNN[:, i, :]            # shape: (batch_size, hidden_size)
            
            # Concatenate the variable with its gradient.
            gradient_sign=grad_i.sign() #Sign of the gradient
            epsilon=1e-14
            gradient_magnitude=torch.log(torch.abs(grad_i)+epsilon) #Magnitude of the gradient plus a small constant to avoid log(0) in log scale
            inp = torch.cat([gradient_sign, gradient_magnitude], dim=1)  # shape: (batch_size, 2) #Input: Sign and magnitude of the gradient
            # Transform the input.
            inp_transformed = self.initial_transform(inp) #Apply a linear transformation to the input
            # Update the hidden state using the RNN cell.
            h_new = self.rnn(inp_transformed, h_i)  # shape: (batch_size, hidden_size)
            # Compute the update delta.
            delta = self.fc(h_new)                  # shape: (batch_size, 1)
            # Update the variable.
            x_new = x_i + delta                     # shape: (batch_size, 1)
            
            new_hidden_list.append(h_new.unsqueeze(1))  # shape: (batch_size, 1, hidden_size)
            x_new_list.append(x_new)                      # shape: (batch_size, 1)
            delta_list.append(delta)                      # shape: (batch_size, 1)
        
        # Concatenate along the variable dimension.
        new_hidden = torch.cat(new_hidden_list, dim=1)  # shape: (batch_size, n, hidden_size)
        x_new = torch.cat(x_new_list, dim=1)              # shape: (batch_size, n)
        delta = torch.cat(delta_list, dim=1)              # shape: (batch_size, n)
        
        return x_new, new_hidden, delta
def main():
    # Hyperparameters.
    batch_size = 64   # Number of functions (shifted instances) per batch.
    nmin=2
    nmax=10
    hidden_size = 512 # Hidden dimension for the RNN.
    K = 5             # Number of inner optimization steps per instance.
    num_epochs = 20000
    learning_rate = 1e-4
    A=10 #Amplitude of the Rastrigin function
    perturb = 0.1 # Standard deviation of the random perturbation of the Rastrigin function. The initial guess is *always 0*.
    # Create weights for the loss accumulated over K steps.
    weights_K = torch.ones(K)
    weights_K[0] = 0
    for i in range(1, K-1):
        weights_K[i+1] = weights_K[i]
    weights_K = weights_K / weights_K.sum()
    
    # Initialize the L2O network.
    l2o_net = L2OOptimizer(hidden_size)
    optimizer = optim.Adam(l2o_net.parameters(), lr=learning_rate)
    # Initialize a separate hidden state per variable.
    
    
    for epoch in range(num_epochs):
        lowest=torch.min(torch.tensor([int(epoch/10)+nmin+1,nmax]))
        n=int(torch.randint(low=nmin,high=lowest,size=(1,))[0])
        hidden_init = torch.zeros(batch_size, n, hidden_size)
        optimizer.zero_grad()
        
        # For each instance, sample a random shift vector s ~ N(0, perturb).
        s = torch.randn(batch_size, n) * perturb
        # The initial guess is x_0 = 0.
        x = torch.zeros(batch_size, n, requires_grad=True)
        # Initialize the per-variable hidden state.
        hidden = hidden_init.clone()  # Shape: (batch_size, n, hidden_size)
        
        total_loss = 0.0
        initial_loss = rastrigin(x, s,A).mean()
        final_loss = 0.0
        
        # Unroll the inner optimization for K steps.
        for j in range(K):
            # Compute the loss for the current x.
            f_val = rastrigin(x, s,A)#/initial_loss  # shape: (batch_size,)
            loss_step = f_val.mean()
            total_loss = total_loss + loss_step * weights_K[j]
            if j == K-1:
                final_loss = loss_step
            if j < K-1:
                # Compute the gradient of the loss with respect to x.
                grad_of_optim = torch.autograd.grad(loss_step, x, create_graph=True)[0]
                # Update x by applying the L2O network per variable.
                x, hidden, delta = l2o_net(x, grad_of_optim, hidden)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, n {n}, Loss (in percent): {(total_loss.mean()/initial_loss.mean()):.3e}, Initial Loss: 1, Final Loss (in percent): {(final_loss.mean()/initial_loss.mean()):.3e}")
        total_loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
