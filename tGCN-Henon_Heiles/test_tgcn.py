import numpy as np
import torch
import torch.nn as nn
from torch_geometric_temporal.nn import GConvGRU
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt

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
class TGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super(TGCN, self).__init__()
        self.gconv_gru = GConvGRU(in_channels=in_channels, out_channels=hidden_channels, K=2)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
    
    def forward(self, X, edge_index):
        batch_size, seq_len, num_nodes, in_channels = X.size()
        assert num_nodes == self.num_nodes, f"Expected {self.num_nodes} nodes, got {num_nodes}"
        outputs = []
        h = None
        
        for t in range(seq_len):
            x_t = X[:, t, :, :]  # [batch_size, num_nodes, in_channels]
            x_t = x_t.contiguous().view(batch_size * num_nodes, in_channels)
            h = self.gconv_gru(x_t, edge_index, H=h)  # [batch_size * num_nodes, hidden_channels]
            h_out = h.view(batch_size, num_nodes, self.hidden_channels)
            outputs.append(h_out)
        
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, num_nodes, hidden_channels]
        out = self.fc(outputs)  # [batch_size, seq_len, num_nodes, out_channels]
        return out

# Data Preprocessing
def preprocess_data(L_data, K_data, mu_data, p_data, seq_len, train_timesteps):
    num_timesteps, num_gaussians, num_L_params = L_data.shape
    num_K_params = K_data.shape[2]
    num_mu_params = mu_data.shape[2]
    num_p_params = p_data.shape[2]
    num_coefficients = num_L_params + num_K_params + num_mu_params + num_p_params
    
    assert K_data.shape[:2] == mu_data.shape[:2] == p_data.shape[:2] == (num_timesteps, num_gaussians), \
        "Inconsistent shapes among L, K, mu, p"
    assert train_timesteps <= num_timesteps, f"train_timesteps ({train_timesteps}) exceeds available timesteps ({num_timesteps})"
    assert num_coefficients == 10, f"Expected 10 coefficients per Gaussian, got {num_coefficients}" 
    
    # Limit to train_timesteps
    L_data = L_data[:train_timesteps]
    K_data = K_data[:train_timesteps]
    mu_data = mu_data[:train_timesteps]
    p_data = p_data[:train_timesteps]
    
    data = np.concatenate([L_data, K_data, mu_data, p_data], axis=2)  # [train_timesteps, num_gaussians, num_coefficients]
    data = data.reshape(train_timesteps, num_gaussians * num_coefficients)  # [train_timesteps, num_nodes]
    
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-10
    data = (data - mean) / std
    
    X, y = [], []
    for t in range(train_timesteps - seq_len):
        X.append(data[t:t+seq_len])
        y.append(data[t+seq_len])
    X = np.array(X, dtype=np.float32)  # [num_samples, seq_len, num_nodes]
    y = np.array(y, dtype=np.float32)  # [num_samples, num_nodes]
    X = torch.tensor(X)[:, :, :, None]  # [num_samples, seq_len, num_nodes, 1]
    y = torch.tensor(y)[:, :, None]     # [num_samples, num_nodes, 1]
    
    # Prepare input for future prediction
    data_normalized = torch.tensor(data, dtype=torch.float32)[:, :, None]  # [train_timesteps, num_nodes, 1]
    
    return X, y, data_normalized, mean, std, num_coefficients, (num_L_params, num_K_params, num_mu_params, num_p_params)

# Predict Over a Long Period Iteratively
def predict_long_term(model, data_normalized, edge_index, mean, std, num_gaussians, num_params_per_coeff, seq_len, forecast_horizon):
    device = data_normalized.device
    model.eval()
    
    # Start with the last seq_len timesteps
    input_seq = data_normalized[-seq_len:].clone()  # [seq_len, num_nodes, 1]
    predictions = []
    
    with torch.no_grad():
        for _ in range(forecast_horizon):
            input_seq = input_seq[None, :, :, :]  # [1, seq_len, num_nodes, 1]
            pred = model(input_seq, edge_index)[:, -1]  # [1, num_nodes, 1]
            
            # Append prediction to sequence
            input_seq = input_seq.squeeze(0)  # [seq_len, num_nodes, 1]
            input_seq = torch.cat([input_seq[1:], pred], dim=0)  # [seq_len, num_nodes, 1]
            predictions.append(pred.cpu().numpy())
    
    # Stack predictions
    predictions = np.stack(predictions, axis=0)  # [forecast_horizon, 1, num_nodes, 1]
    predictions = predictions.squeeze(1)  # [forecast_horizon, num_nodes, 1]
    
    # Denormalize
    mean = mean.reshape(1, -1, 1)  # [1, num_nodes, 1]
    std = std.reshape(1, -1, 1)    # [1, num_nodes, 1]
    predictions = predictions * std + mean  # [forecast_horizon, num_nodes, 1]
    predictions = predictions.squeeze(-1)   # [forecast_horizon, num_nodes]
    
    # Reshape to [forecast_horizon, num_gaussians, num_coefficients]
    num_coefficients = sum(num_params_per_coeff)
    predictions = predictions.reshape(forecast_horizon, num_gaussians, num_coefficients)  # [forecast_horizon, 29, 10]
    
    # Split into L, K, mu, p
    num_L_params, num_K_params, num_mu_params, num_p_params = num_params_per_coeff
    start = 0
    L_pred = predictions[:, :, start:start + num_L_params]
    start += num_L_params
    K_pred = predictions[:, :, start:start + num_K_params]
    start += num_K_params
    mu_pred = predictions[:, :, start:start + num_mu_params]
    start += num_mu_params
    p_pred = predictions[:, :, start:start + num_p_params]
    
    return L_pred, K_pred, mu_pred, p_pred

# Training and Long-Term Prediction Function
def train_and_predict_long_term(L_data, K_data, mu_data, p_data, num_gaussians, seq_len=20, train_timesteps=1000, forecast_horizon=100, batch_size=32, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess data
    try:
        X, y, data_normalized, mean, std, num_coefficients, num_params_per_coeff = preprocess_data(
            L_data, K_data, mu_data, p_data, seq_len, train_timesteps
        )
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return
    
    num_nodes = num_gaussians * num_coefficients
    X, y, data_normalized = X.to(device), y.to(device), data_normalized.to(device)
    
    # Build edge index
    edge_index = build_adjacency_matrix(num_gaussians, num_coefficients).to(device)
    
    # Initialize model
    in_channels = 1
    hidden_channels = 64
    out_channels = 1
    model = TGCN(in_channels, hidden_channels, out_channels, num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch_X, edge_index)
            loss = loss_fn(output[:, -1], batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        if epoch % 10 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # Predict long term
    L_pred, K_pred, mu_pred, p_pred = predict_long_term(
        model, data_normalized, edge_index, mean, std, num_gaussians, num_params_per_coeff, seq_len, forecast_horizon
    )
    
    # Get true values for the forecast period (timesteps 1001-1100)
    start_idx = train_timesteps
    end_idx = start_idx + forecast_horizon
    L_true = L_data[start_idx:end_idx]  # [forecast_horizon, 29, 3]
    K_true = K_data[start_idx:end_idx]  # [forecast_horizon, 29, 3]
    mu_true = mu_data[start_idx:end_idx]  # [forecast_horizon, 29, 2]
    p_true = p_data[start_idx:end_idx]  # [forecast_horizon, 29, 2]
    
    # Save predictions
    np.savez(
        "predicted_coefficients_2D_long_term.npz",
        L_pred=L_pred, K_pred=K_pred, mu_pred=mu_pred, p_pred=p_pred,
        L_true=L_true, K_true=K_true, mu_true=mu_true, p_true=p_true
    )
    
    # Visualize first Gaussian's first L coefficient
    plt.figure(figsize=(10, 5))
    timesteps = range(start_idx, end_idx)
    plt.plot(timesteps, L_pred[:, 0, 0], label='Predicted L (Gaussian 1, param 1)')
    plt.plot(timesteps, L_true[:, 0, 0], label='True L (Gaussian 1, param 1)')
    plt.xlabel('Timestep')
    plt.ylabel('L Coefficient')
    plt.title('Long-Term Prediction (Timesteps 1001-1100)')
    plt.legend()
    plt.savefig("prediction_L_gaussian1_long_term.png")
    plt.close()
    
    return L_pred, K_pred, mu_pred, p_pred

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
    
    # Train and predict long term
    L_pred, K_pred, mu_pred, p_pred = train_and_predict_long_term(
        L_data_2D, K_data_2D, mu_data_2D, p_data_2D, num_gaussians, 
        seq_len=20, train_timesteps=1000, forecast_horizon=100
    )