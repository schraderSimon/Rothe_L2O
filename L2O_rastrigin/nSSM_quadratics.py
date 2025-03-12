import torch
import torch.nn as nn
import torch.optim as optim

# Define the quadratic function: f(x) = x^T A x
def quadratic(x, A):
    return torch.einsum('bi,bij,bj->b', x, A, x)

# Generate random positive definite A matrices
def generate_random_quadratic(n, batch_size):
    A = torch.randn(batch_size, n, n)
    A = 0.5 * (A + A.transpose(-1, -2))
    A = A @ A.transpose(-1, -2) + 0.1 * torch.eye(n).unsqueeze(0)
    return A

# Transform gradient as per paper
def transform_gradient(grad, rho=10):
    threshold = torch.exp(torch.tensor(-rho, dtype=torch.float32))  # e^-p, e.g., e^-10
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

# Stabilized nSSM (they tend to explode)
class NeuralSSM(nn.Module):
    def __init__(self, input_size, state_size, output_size):
        super(NeuralSSM, self).__init__()
        self.state_size = state_size
        self.input_size = input_size
        self.output_size = output_size
        
        self.A_net = nn.Sequential(
            nn.Linear(state_size + input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, state_size * state_size)
        )
        self.B_net = nn.Sequential(
            nn.Linear(state_size + input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, state_size * input_size)
        )
        self.C = nn.Linear(state_size, output_size)
        
    def forward(self, u, h_prev):
        batch_size = u.shape[0]
        combined = torch.cat([h_prev, u], dim=1)
        
        A_flat = self.A_net(combined)
        B_flat = self.B_net(combined)
        
        A = A_flat.view(batch_size, self.state_size, self.state_size)
        B = B_flat.view(batch_size, self.state_size, self.input_size)
        
        # Normalize A to prevent explosion (spectral norm < 1)
        A = A / (torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + 1e-5)
        
        h_t = torch.einsum('bsh,bh->bs', A, h_prev) + torch.einsum('bsi,bi->bs', B, u)
        h_t = h_t / (h_t.norm(dim=1, keepdim=True) + 1e-5)  # Normalize state
        
        y_t = self.C(h_t)
        y_t = torch.tanh(y_t)  # Bound updates to [-1, 1] (stabilzing factor)
        
        return y_t, h_t

# Hyperparameters
n = 10
state_size = 32
T = 20
batch_size = 32
num_epochs = 5000
learning_rate = 0.001  # Reduced learning rate
rho = 10

# Define the model
ssm = NeuralSSM(input_size=2 * n, state_size=state_size, output_size=n)
optimizer = optim.Adam(ssm.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    x_t = torch.rand(batch_size, n) * 10 - 5
    x_t = x_t.clone().requires_grad_(True)
    A = generate_random_quadratic(n, batch_size)
    h_t = torch.ones(batch_size, state_size)
    total_loss = 0
    
    for t in range(T):
        f_x = quadratic(x_t, A)
        grad = torch.autograd.grad(f_x.sum(), x_t, create_graph=True, retain_graph=True)[0]
        u_t = transform_gradient(grad, rho=rho)
        
        delta_x_t, h_t = ssm(u_t, h_t)
        x_t = x_t + delta_x_t
        total_loss += f_x.mean()
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")

# Testing with debugging
x_test = torch.tensor([[2.0, -3.0, -1.9, 1.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0]], requires_grad=True)
A_test = generate_random_quadratic(n, 1)
h_t = torch.zeros(1, state_size)
print(f"Initial f(x): {quadratic(x_test, A_test).item():.4f}")
for t in range(T):
    f_x = quadratic(x_test, A_test)
    grad = torch.autograd.grad(f_x.sum(), x_test, create_graph=True)[0]
    u_t = transform_gradient(grad, rho=rho)
    delta_x_t, h_t = ssm(u_t, h_t)
    x_test = (x_test + delta_x_t).detach().requires_grad_(True)
    print(f"Step {t+1}, f(x): {f_x.item():.4f}, |delta_x|: {delta_x_t.norm().item():.4f}, |h_t|: {h_t.norm().item():.4f}")
print(f"Final f(x) after {T} steps: {quadratic(x_test, A_test).item():.4f}")