import numpy as np
import torch
import torch.nn as nn
from torch_geometric_temporal.nn import GConvGRU, TGCN,TGCN2
from torch_geometric.utils import dense_to_sparse

# Build Adjacency Matrix and Convert to Edge Index
def build_adjacency_matrix(num_gaussians, num_coefficients):
    num_nodes = num_gaussians * num_coefficients
    adj = torch.zeros(num_nodes, num_nodes)
    
    for j in range(num_gaussians):
        for i in range(num_coefficients):
            node_idx = j * num_coefficients + i
            
            # Same Gaussian (j = l)
            for k in range(num_coefficients):
                other_idx = j * num_coefficients + k
                adj[node_idx, other_idx] = 1
                adj[other_idx, node_idx] = 1
            
            # Same coefficient type (i = k)
            for l in range(num_gaussians):
                other_idx = l * num_coefficients + i
                adj[node_idx, other_idx] = 1
                adj[other_idx, node_idx] = 1
    
    adj.fill_diagonal_(0)  # Remove self-loops
    edge_index, _ = dense_to_sparse(adj)  # [2, num_edges]
    return edge_index

# T-GCN Model with GConvGRU
class TGCNmod(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super(TGCNmod, self).__init__()
        self.gconv_gru = TGCN(in_channels=in_channels, out_channels=hidden_channels)  # Chebyshev filter
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
    
    def forward(self, X, edge_index):
        # X: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = X.size()
        assert num_nodes == self.num_nodes, f"Expected {self.num_nodes} nodes, got {num_nodes}"
        outputs = []
        h = None  # Initial hidden state
        
        for t in range(seq_len):
            x_t = X[:, t, :, :]  # [batch_size, num_nodes, in_channels]
            x_t = x_t.contiguous().view(batch_size * num_nodes, in_channels)  # [batch_size * num_nodes, in_channels]
            h = self.gconv_gru(x_t, edge_index, H=h)  # [batch_size * num_nodes, hidden_channels]
            h_out = h.view(batch_size, num_nodes, self.hidden_channels)  # [batch_size, num_nodes, hidden_channels]
            outputs.append(h_out)
        
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, num_nodes, hidden_channels]
        out = self.fc(outputs)  # [batch_size, seq_len, num_nodes, out_channels]
        return out

# Data Preprocessing
def preprocess_data(L_data, K_data, mu_data, p_data, seq_len):
    num_timesteps, num_gaussians, num_L_params = L_data.shape
    num_K_params = K_data.shape[2]
    num_mu_params = mu_data.shape[2]
    num_p_params = p_data.shape[2]
    num_coefficients = num_L_params + num_K_params + num_mu_params + num_p_params
    
    # Validate shapes
    assert K_data.shape[:2] == mu_data.shape[:2] == p_data.shape[:2] == (num_timesteps, num_gaussians), \
        "Inconsistent shapes among L, K, mu, p"
    
    # Concatenate coefficients
    data = np.concatenate([L_data, K_data, mu_data, p_data], axis=2)  # [num_timesteps, num_gaussians, num_coefficients]
    data = data.reshape(num_timesteps, num_gaussians * num_coefficients)  # [num_timesteps, num_nodes]
    
    # Normalize
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-10
    data = (data - mean) / std
    
    # Create input-output pairs
    X, y = [], []
    for t in range(num_timesteps - seq_len):
        X.append(data[t:t+seq_len])
        y.append(data[t+seq_len])
    X = np.array(X, dtype=np.float32)  # [num_samples, seq_len, num_nodes]
    y = np.array(y, dtype=np.float32)  # [num_samples, num_nodes]
    X = torch.tensor(X)[:, :, :, None]  # [num_samples, seq_len, num_nodes, 1]
    y = torch.tensor(y)[:, :, None]     # [num_samples, num_nodes, 1]
    
    return X, y, mean, std, num_coefficients

# Training Function
def train_tgcn(L_data, K_data, mu_data, p_data, num_gaussians, seq_len=12, batch_size=32, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess data
    try:
        X, y, mean, std, num_coefficients = preprocess_data(L_data, K_data, mu_data, p_data, seq_len)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return
    
    print("Checkpoint 1")
    
    num_nodes = num_gaussians * num_coefficients
    X, y = X.to(device), y.to(device)

    print("Checkpoint 2")
    
    # Build edge index
    edge_index = build_adjacency_matrix(num_gaussians, num_coefficients).to(device)

    print("Checkpoint 3")
    
    # Initialize model
    in_channels = 1
    hidden_channels = 64
    out_channels = 1
    model = TGCNmod(in_channels, hidden_channels, out_channels, num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Checkpoint 4")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for i in range(0, len(X), batch_size):
            print(i)
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch_X, edge_index)
            loss = loss_fn(output[:, -1], batch_y)  # Predict last timestep
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        #if epoch % 1 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

# Example Usage
if __name__ == "__main__":
    # Load data
    try:
        infile_2D = np.load("nonlinear_coefficients_dimension=2_ngauss_init=29.npz")
        L_data_2D = infile_2D['L']
        K_data_2D = infile_2D['K']
        mu_data_2D = infile_2D['mu']
        p_data_2D = infile_2D['p']
        num_gaussians = 29
    except FileNotFoundError:
        print("Error: .npz file not found. Please check the file path.")
        exit(1)
    
    # Train model
    train_tgcn(L_data_2D, K_data_2D, mu_data_2D, p_data_2D, num_gaussians)