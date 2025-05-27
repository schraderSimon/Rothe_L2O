import numpy as np
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)
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
            #for l in range(num_gaussians):
            #    other_idx = l * num_coefficients + i
            #    adj[node_idx, other_idx] = 1
            #    adj[other_idx, node_idx] = 1
    
    adj.fill_diagonal_(0)  # Remove self-loops
    #edge_index, _ = dense_to_sparse(adj)  # [2, num_edges]
    return adj

# T-GCN Model with GConvGRU
class own_TGCN(nn.Module):
    def __init__(self,num_layers_GCN,num_layers_LSTM,size_LSTM,size_GCN,size_GCN_out,adjacency_matrix,num_nodes,run_GCN=True,T=5):
        super(own_TGCN, self).__init__()
        self.num_layers_GCN = num_layers_GCN
        self.num_layers_LSTM = num_layers_LSTM
        self.size_LSTM = size_LSTM
        self.size_GCN = size_GCN
        self.size_GCN_out = size_GCN_out       # store
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = num_nodes
        self.run_GCN = run_GCN
        self.T=T
        self.softmax = nn.Softmax(dim=-1)

        if not run_GCN:
            print("Warning: run_GCN is set to False. The GCN layers will not be used.")
            self.size_GCN_out = 1  # Set to 1 if GCN is not used, so LSTM can still work
            size_GCN_out = 1  # Set GCN size to 1 if not used
        # ----- build convolution matrix (unchanged) -----
        A = torch.tensor(adjacency_matrix) if not torch.is_tensor(adjacency_matrix) else adjacency_matrix
        assert A.size(0) == A.size(1), "Adjacency matrix must be square"
        assert A.size(0) == num_nodes, f"Adjacency matrix size {A.size(0)} does not match num_nodes {num_nodes}"
        Atilde = A + torch.eye(A.size(0))
        Dtilde = torch.sum(Atilde, dim=1)
        Dtilde_inv_sqrt = torch.diag(1 / torch.sqrt(Dtilde))
        convolution_matrix = torch.matmul(Dtilde_inv_sqrt, torch.matmul(Atilde, Dtilde_inv_sqrt))
        convolution_matrix_sparse = convolution_matrix.to_sparse()

        self.register_buffer('convolution_matrix', convolution_matrix)
        self.register_buffer('convolution_matrix_sparse', convolution_matrix_sparse)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(nn.Linear(1, size_GCN))           
        for _ in range(num_layers_GCN - 1):
            self.gcn_layers.append(nn.Linear(size_GCN, size_GCN))
        self.gcn_layers.append(nn.Linear(size_GCN, size_GCN_out)) 

        # ----- LSTM now sees N * size_GCN_out features ----------
        self.lstm = nn.LSTM(input_size=num_nodes * size_GCN_out, 
                            hidden_size=size_LSTM,
                            num_layers=num_layers_LSTM,
                            batch_first=True)

        self.fc = nn.Linear(size_LSTM, num_nodes)  # unchanged
        self.fc_alpha = nn.Linear(size_LSTM, T)   #
    # ------------------------------------------------------------------
    def forward_gcn(self, x):
        # x is of shape [batch_size, num_nodes, 1]
        for layer in self.gcn_layers:               
            x = torch.matmul(self.convolution_matrix, x) 
            x = torch.relu(layer(x))                            
        return x                                        

    # ------------------------------------------------------------------
    def forward(self, x):
        # x: [batch, T, num_nodes, 1]
        batch_size, seq_len, num_nodes, _ = x.shape
        assert seq_len == self.T, "input window must equal T"

        # ------------ run (optional) GCN -----------------
        x_flat = x.reshape(batch_size * seq_len, num_nodes, 1) 
        if self.run_GCN:
            x_flat = self.forward_gcn(x_flat)
        x_flat = x_flat.reshape(batch_size, seq_len,
                                num_nodes * self.size_GCN_out)         

        _, (h_n, _) = self.lstm(x_flat)
        alpha = self.fc_alpha(h_n[-1]) 
        pred = torch.einsum('bt,btni->bni', alpha, x)
        return pred, alpha       
def preprocess_data(L_data, K_data, mu_data, p_data,train_timesteps):
    L = L_data.reshape(L_data.shape[0], L_data.shape[1] * L_data.shape[2], 1)
    K=K_data.reshape((K_data.shape[0], K_data.shape[1]*K_data.shape[2], 1))  
    mu=mu_data.reshape((mu_data.shape[0], mu_data.shape[1]*mu_data.shape[2], 1))  
    p=p_data.reshape((p_data.shape[0], p_data.shape[1]*p_data.shape[2], 1)) 
    params=np.concatenate([L, K, mu, p], axis=1)  
    dparams=params[1:,:,:] - params[:-1,:,:]  
    params_train=dparams[:train_timesteps,:,:] 
    params_test=dparams[train_timesteps:,:,:]
    #params_train=params[:train_timesteps,:,:]
    #params_test=params[train_timesteps:,:,:]
    mean = params_train.mean(axis=0, keepdims=True)
    std = params_train.std(axis=0, keepdims=True) 
    params_train_normalized = (params_train - mean) / (std + 1e-14)
    params_test_normalized = (params_test - mean) / (std + 1e-14)
    return params_train_normalized, params_test_normalized, mean, std

def train_model(train_data, cfg,adjaceny_matrix,test_data):
    num_layers_GCN = cfg['num_layers_GCN']
    num_layers_LSTM = cfg['num_layers_LSTM']
    size_LSTM = cfg['size_LSTM']
    size_GCN = cfg['size_GCN']
    size_GCN_out = cfg['size_GCN_out']
    num_nodes = cfg['num_nodes']
    run_GCN = cfg['run_GCN']
    model= own_TGCN(
        num_layers_GCN=num_layers_GCN,
        num_layers_LSTM=num_layers_LSTM,
        size_LSTM=size_LSTM,
        size_GCN=size_GCN,
        size_GCN_out=size_GCN_out, 
        adjacency_matrix=adjaceny_matrix,
        num_nodes=num_nodes,
        run_GCN=run_GCN,
        T=cfg['T']
    )
    train_data=torch.tensor(train_data)  # Convert to tensor
    test_data=torch.tensor(test_data)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()
    model.train()
    T=cfg["T"] #Unrolling parameter; i.e. on how many previous deltas the model is trained
    num_epochs= cfg['num_epochs']
    batch_size = cfg['batch_size']
    indices_testing=list(range(test_data.shape[0] - T))
    idiot_prediction=torch.stack([test_data[s + T-1, :, :] for s in indices_testing], dim=0) #We simply predict the previous change
    desired_output_test = torch.stack([test_data[s + T, :, :] for s in indices_testing], dim=0) #We simply predict 
    idiot_error=criterion(desired_output_test,idiot_prediction)
    for epoch in range(cfg['num_epochs']):
        start_idx=np.random.randint(0, train_data.shape[0] - T - 1,size=batch_size)
        train_data_batch = torch.stack([train_data[s : s + T, :, :] for s in start_idx], dim=0)
        desired_output = torch.stack([train_data[s + T, :, :] for s in start_idx], dim=0)
        output, _ = model(train_data_batch)          # unpack

        loss = criterion(output, desired_output)  
        
        if True:
            print(f'Epoch [{epoch + 1}/{cfg["num_epochs"]}], Loss: {loss.item():.4f}')
        for name, param in model.named_parameters():
            if 'weight' in name:
                loss += 1e-5* torch.sum(param ** 2)
        opt.zero_grad()  # clean the slate  
        loss.backward()  # compute gradients  
        opt.step()       # update parameters 
        if (epoch+1) % 5 == 0 or epoch == 0:  # Print every 5 epochs (adjust as you like)
            model.eval()
            losses = []
            with torch.no_grad():
                
                test_data_batch = torch.stack([test_data[s : s + T, :, :] for s in indices_testing], dim=0)
                
                output_test,_ = model(test_data_batch)
                loss = criterion(output_test, desired_output_test)
            print(f'Test Loss (MSE, average over test set): {loss.item():.4f}, Idiotic: {idiot_error.item():.4f}')
            model.train()
    return model
if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    dim=3
    if dim==2:
        infile_2D = np.load("nonlinear_coefficients_dimension=2_ngauss_init=29.npz")
        L_data=L_data_2D = infile_2D['L']
        K_data=K_data_2D = infile_2D['K']
        mu_data=mu_data_2D = infile_2D['mu']
        p_data=p_data_2D = infile_2D['p']
    elif dim==3:
        infile_3D= np.load("nonlinear_coefficients_dimension=3_ngauss_init=28.npz")
        L_data_3D = infile_3D['L']
        K_data_3D = infile_3D['K']
        mu_data_3D = infile_3D['mu']
        p_data_3D = infile_3D['p']
        L_data=L_data_3D
        K_data=K_data_3D
        mu_data=mu_data_3D
        p_data=p_data_3D
    num_coefficients=L_data.shape[2]+K_data.shape[2] + mu_data.shape[2] + p_data.shape[2]
    num_gaussians=L_data.shape[1]
    num_nodes= num_gaussians * num_coefficients
    adjaceny_matrix= build_adjacency_matrix(num_gaussians=num_gaussians, num_coefficients=num_coefficients)
    train_timesteps = 5000  # Number of timesteps to use for training
    params_train, params_test_and_valid, mean, std = preprocess_data(L_data, K_data, mu_data, p_data, train_timesteps)
    params_test=params_test_and_valid[:500,:,:]
    print(params_test.shape)
    cfg = {
        'num_epochs': 2000,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'T': 10,  # Unrolling parameter
        "num_layers_GCN": 2,
        "num_layers_LSTM": 3,
        "size_LSTM": 64,
        "size_GCN": 64,
        "size_GCN_out": 1,
        "num_nodes": num_nodes,
        "run_GCN": True,
    }
    model = train_model(params_train, cfg,adjaceny_matrix,params_test)
