import torch
import torch.nn as nn
import torch.optim as optim

# Define the quadratic function: f(x) = x^T A x, as in the paper
def quadratic(x, A):
    # x: (batch_size, n), A: (batch_size, n, n)
    # Compute x^T A x for each batch element
    return torch.einsum('bi,bij,bj->b', x, A, x)  # Shape: (batch_size,)

# Generate random positive definite A matrices
def generate_random_quadratic(n, batch_size):
    # Random symmetric matrix + diagonal shift for positive definiteness
    A = torch.randn(batch_size, n, n)
    A = 0.5 * (A + A.transpose(-1, -2))  # Make symmetric
    A = A @ A.transpose(-1, -2) + 0.1 * torch.eye(n).unsqueeze(0)  # Ensure positive definite
    return A

# Transform gradient into {log(grad), sign(grad)} or {-1, e^-p * grad}
def transform_gradient(grad, p=10):
    threshold = torch.exp(torch.tensor(-p, dtype=torch.float32))  # e^-p, e.g., e^-10
    abs_grad = grad.abs()  # Shape: (batch_size, n)
    mask = abs_grad > threshold  # Shape: (batch_size, n)
    
    # Initialize output: 2 channels per dimension
    batch_size, n = grad.shape
    input_t = torch.zeros(batch_size, 2 * n)  # Shape: (batch_size, 2n)
    
    # Compute log(|grad|) where mask is True, use 0 as a placeholder where False
    log_abs_grad = torch.log(abs_grad.clamp(min=threshold))  # Clamp to avoid log(0)
    log_abs_grad = torch.where(mask, log_abs_grad, torch.zeros_like(log_abs_grad))
    sign_grad = torch.sign(grad)  # -1, 0, or 1
    
    # Where |grad| <= e^-p: {-1, e^-p * grad}
    small_grad = threshold * grad
    
    # Fill first channel (log or -1)
    input_t[:, 0:n] = torch.where(mask, log_abs_grad, -1 * torch.ones_like(grad))
    
    # Fill second channel (sign or e^-p * grad)
    input_t[:, n:] = torch.where(mask, sign_grad, small_grad)
    
    return input_t  # Shape: (batch_size, 2n)

# Hyperparameters
n = 10              # Dimension of the input vector
hidden_size = 20   # LSTM hidden state size per layer (as per the paper)
num_layers = 2     # 2-layer LSTM as in the paper (as per the paper)
T = 15             # Number of unrolling steps (paper uses 20) -> the model seems to be sensitive to this parameter
batch_size = 32     # Number of starting points per batch (unsure about what the paper uses)
num_epochs = 5000  # Training epochs (unsure about what the paper uses)
learning_rate = 0.001
p = 10             # Scaling factor for log term (paper suggests 10)

# Define the model (input size is 2n due to two channels)
lstm = nn.LSTM(input_size=2 * n, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
linear = nn.Linear(hidden_size, n) # linear output
optimizer = optim.Adam(list(lstm.parameters()) + list(linear.parameters()), lr=learning_rate) # ADAM for training

# Training loop
for epoch in range(num_epochs):
    # Sample random starting points in [-5, 5]^n
    x_t = torch.rand(batch_size, n) * 10 - 5  # Shape: (batch_size, n)
    x_t = x_t.clone().requires_grad_(True)
    
    # Generate random quadratic function for this batch
    A = generate_random_quadratic(n, batch_size)  # Shape: (batch_size, n, n)
    
    # Initialize LSTM hidden states
    h = torch.ones(num_layers, batch_size, hidden_size)
    c = torch.ones(num_layers, batch_size, hidden_size)
    
    # Accumulate loss over trajectory
    total_loss = 0
    
    # Unroll optimization steps
    for t in range(T):
        # Compute function value and gradient
        f_x = quadratic(x_t, A)  # Shape: (batch_size,)
        print(f_x)
        grad = torch.autograd.grad(f_x.sum(), x_t, create_graph=True, retain_graph=True)[0]  # Shape: (batch_size, n)
        
        # Transform gradient into LSTM input
        input_t = transform_gradient(grad, p=p)  # Shape: (batch_size, 2n)
        input_t = input_t.unsqueeze(0)  # Shape: (1, batch_size, 2n)
        
        # LSTM forward pass
        output_t, (h, c) = lstm(input_t, (h, c))  # output_t: (1, batch_size, hidden_size)
        
        # Compute update
        delta_x_t = linear(output_t.squeeze(0))  # Shape: (batch_size, n)
        
        # Update position
        x_t = x_t + delta_x_t
        
        # Accumulate loss (unweighted sum as in paper)
        total_loss += f_x.mean()
    
    # Backpropagate and optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(lstm.parameters()) + list(linear.parameters()), 1.0)
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")

# Testing the trained LSTM
x_test = torch.tensor([[2.0, -2.0, -1.9, 1.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0]], requires_grad=True)  # Example starting point
A_test = generate_random_quadratic(n, 1)  # Single random A for testing
h = torch.zeros(num_layers, 1, hidden_size)
c = torch.zeros(num_layers, 1, hidden_size)
print(f"Initial f(x): {quadratic(x_test, A_test).item():.4f}")
for t in range(100):
    f_x = quadratic(x_test, A_test)
    grad = torch.autograd.grad(f_x.sum(), x_test, create_graph=True)[0]
    input_t = transform_gradient(grad, p=p).unsqueeze(0)
    output_t, (h, c) = lstm(input_t, (h, c))
    delta_x_t = linear(output_t.squeeze(0))
    x_test = (x_test + delta_x_t).detach().requires_grad_(True)
    if t == 0 or t == T-1:
        print(f"Step {t+1}, f(x): {quadratic(x_test, A_test).item():.4f}")
print(f"Final f(x) after 100 steps: {quadratic(x_test, A_test).item():.4f}")