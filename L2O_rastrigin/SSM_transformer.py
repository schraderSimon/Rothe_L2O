import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Maximum dimensionality for handling variable sizes
MAX_DIM = 20
HIDDEN_DIM = 256  # Hidden state dimension

# MLP for generating state-dependent matrices
class MatrixMLP(nn.Module):
    def __init__(self, input_dim, output_shape):
        """
        MLP to generate matrices A, B, C, D.
        - input_dim: Dimension of input (e.g., hidden state)
        - output_shape: Shape of output matrix (e.g., [d_h, d_h] for A)
        """
        super().__init__()
        self.output_dim = output_shape[0] * output_shape[1]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, x):
        """
        Generate matrix from input.
        Returns matrix of shape output_shape.
        """
        batch_size = x.shape[0]
        out = self.net(x)  # [batch_size, output_dim]
        return out.view(batch_size, self.output_dim)

# Neural State-Space Model
class SSMOptimizer(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, max_dim=MAX_DIM):
        """
        State-space model for optimization.
        - hidden_dim: Dimension of hidden state h_t
        - max_dim: Maximum dimensionality of parameters
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_dim = max_dim
        input_dim = 2 * max_dim + 1  # [grad, x, t]
        
        # MLPs for A, B, C, D
        self.A_mlp = MatrixMLP(hidden_dim, [hidden_dim, hidden_dim])
        self.B_mlp = MatrixMLP(hidden_dim, [hidden_dim, input_dim])
        self.C_mlp = MatrixMLP(hidden_dim, [max_dim, hidden_dim])
        self.D_mlp = MatrixMLP(hidden_dim, [max_dim, input_dim])
        
        # MLP for log-variance of Gaussian policy
        self.logvar_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_dim)
        )

    def forward(self, h_prev, x_input, dim):
        """
        Compute state transition and output.
        - h_prev: Previous hidden state [batch_size, hidden_dim]
        - x_input: Input [batch_size, 2*dim+1] (grad, x, t)
        - dim: Current dimensionality
        Returns new hidden state h_t and action mean/logvar.
        """
        # Pad input to max_dim
        padded_input = torch.zeros(x_input.shape[0], 2 * self.max_dim + 1, device=x_input.device)
        padded_input[:, :2*dim+1] = x_input
        
        # Compute matrices
        A = self.A_mlp(h_prev)  # [batch_size, hidden_dim, hidden_dim]
        B = self.B_mlp(h_prev)  # [batch_size, hidden_dim, 2*max_dim+1]
        h_t = A @ h_prev.unsqueeze(-1) + B @ padded_input.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
        h_t = h_t.squeeze(-1)  # [batch_size, hidden_dim]
        
        C = self.C_mlp(h_t)  # [batch_size, max_dim, hidden_dim]
        D = self.D_mlp(h_t)  # [batch_size, max_dim, 2*max_dim+1]
        mean = (C @ h_t.unsqueeze(-1) + D @ padded_input.unsqueeze(-1)).squeeze(-1)  # [batch_size, max_dim]
        mean = mean[:, :dim]  # Truncate to current dim
        
        logvar = self.logvar_mlp(h_t)[:, :dim]  # [batch_size, dim]
        
        return h_t, mean, logvar

# Quadratic function and gradient
def quadratic_function(x, Q, b):
    """
    Compute f(x) = 0.5 * x^T Q x + b^T x
    - x: [dim] or [batch_size, dim]
    - Q: [dim, dim]
    - b: [dim]
    Returns scalar.
    """
    x = x.view(-1, 1)
    return 0.5 * (x.T @ Q @ x).squeeze() + (b @ x).squeeze()

def quadratic_gradient(x, Q, b):
    """
    Compute grad = Q x + b
    - x: [dim] or [batch_size, dim]
    - Q: [dim, dim]
    - b: [dim]
    Returns [dim].
    """
    return (Q @ x.view(-1, 1)).squeeze() + b

# Generate random quadratic
def generate_quadratic(dim):
    """
    Generate random positive-definite Q and vector b.
    - dim: Dimensionality
    Returns Q, b.
    """
    Q = torch.randn(dim, dim)
    Q = Q @ Q.T + 0.1 * torch.eye(dim)  # Positive-definite
    b = torch.randn(dim)
    return Q, b

# Training function
def train_optimizer(num_episodes=1000, seq_len=20, lr=1e-4):
    """
    Train SSM using REINFORCE.
    - num_episodes: Number of episodes
    - seq_len: Optimization steps per episode
    - lr: Learning rate
    """
    model = SSMOptimizer()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        # Sample random dimension
        dim = np.random.randint(5, 21)
        Q, b = generate_quadratic(dim)
        
        # Initialize
        x = torch.randn(dim)
        h = torch.zeros(1, HIDDEN_DIM)  # Initial hidden state
        log_probs = []
        rewards = []
        
        # Optimization trajectory
        for t in range(seq_len):
            grad = quadratic_gradient(x, Q, b)
            x_input = torch.cat([
                grad,
                x,
                torch.tensor([t], dtype=torch.float32)
            ]).unsqueeze(0)  # [1, 2*dim+1]
            
            # Forward pass
            h, mean, logvar = model(h, x_input, dim)
            std = torch.exp(0.5 * logvar)
            
            # Sample action
            dist = torch.distributions.Normal(mean, std)
            delta_x = dist.sample()  # [1, dim]
            log_prob = dist.log_prob(delta_x).sum()
            log_probs.append(log_prob)
            
            # Update parameters
            x = x + delta_x.squeeze()
            
            # Compute reward
            reward = -quadratic_function(x, Q, b)
            rewards.append(reward)
        
        # Compute returns
        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Log
        if episode % 100 == 0:
            avg_reward = torch.mean(torch.tensor(rewards)).item()
            print(f"Episode {episode}, Dim: {dim}, Avg Reward: {avg_reward:.4f}, Loss: {loss.item():.4f}")

    return model

# Evaluation function
def evaluate_optimizer(model, num_tests=10, seq_len=20):
    """
    Evaluate on unseen quadratics.
    - model: Trained SSM
    - num_tests: Number of test functions
    - seq_len: Optimization steps
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(num_tests):
            dim = np.random.randint(5, 21)
            Q, b = generate_quadratic(dim)
            x = torch.randn(dim)
            h = torch.zeros(1, HIDDEN_DIM)
            
            for t in range(seq_len):
                grad = quadratic_gradient(x, Q, b)
                x_input = torch.cat([
                    grad,
                    x,
                    torch.tensor([t], dtype=torch.float32)
                ]).unsqueeze(0)
                h, mean, _ = model(h, x_input, dim)
                x = x + mean.squeeze()
            
            final_loss = quadratic_function(x, Q, b)
            total_loss += final_loss.item()
    
    avg_loss = total_loss / num_tests
    print(f"Average final function value on {num_tests} tests: {avg_loss:.4f}")

# Main execution
if __name__ == "__main__":
    trained_model = train_optimizer(num_episodes=2000, seq_len=20, lr=1e-4)
    evaluate_optimizer(trained_model, num_tests=50)