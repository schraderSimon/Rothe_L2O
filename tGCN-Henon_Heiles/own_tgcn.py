import numpy as np
import torch
import torch.nn as nn
import copy
import os
import sys
torch.set_default_tensor_type(torch.DoubleTensor)
import matplotlib.pyplot as plt
def _build_filename(cfg: dict, ext = ".pt", epoch = None, base_dir: str = "outputs/") -> str:
    cfg_local = copy.deepcopy(cfg)
    if epoch is not None:
        cfg_local["num_epochs"] = epoch

    parts = [f"{k}={cfg_local[k]}" for k in sorted(cfg_local.keys())]
    fname = "_".join(parts) + ext
    return os.path.join(base_dir, fname)


def save_model(model: torch.nn.Module, cfg: dict, epoch: int, base_dir: str = "outputs/") -> str:  # noqa: D401
    os.makedirs(base_dir, exist_ok=True)
    path = _build_filename(cfg, ext=".pt", epoch=epoch, base_dir=base_dir)
    torch.save(model.state_dict(), path)
    return path

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
    return adj

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

        self.lstm = nn.LSTM(input_size=num_nodes * size_GCN_out, 
                            hidden_size=size_LSTM,
                            num_layers=num_layers_LSTM,
                            batch_first=True)

        self.fc = nn.Linear(size_LSTM, num_nodes)  # unchanged
        self.fc_alpha = nn.Linear(size_LSTM, T)   #
    def forward_gcn(self, x):
        for layer in self.gcn_layers:               
            x = torch.matmul(self.convolution_matrix, x) 
            x = torch.relu(layer(x))                            
        return x                                        

    def forward(self, x):
        batch_size, seq_len, num_nodes, _ = x.shape
        assert seq_len == self.T, "input window must equal T"

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
    mean = params_train.mean(axis=0, keepdims=True)
    std = params_train.std(axis=0, keepdims=True) 
    params_train_normalized = (params_train - mean) / (std + 1e-14)
    params_test_normalized = (params_test - mean) / (std + 1e-14)
    return params_train_normalized, params_test_normalized, mean, std

def train_model(train_data, cfg,adjaceny_matrix,test_data,save_model_state=None,save_output=True):
    num_layers_GCN = cfg['num_layers_GCN']
    num_layers_LSTM = cfg['num_layers_LSTM']
    size_LSTM = cfg['size_LSTM']
    size_GCN = cfg['size_GCN']
    size_GCN_out = cfg['size_GCN_out']
    num_nodes = cfg['num_nodes']
    run_GCN = cfg['run_GCN']
    l2_penalty = cfg['l2_penalty']
    T=cfg["T"] #Unrolling parameter; i.e. on how many previous deltas the model is trained
    num_epochs= cfg['num_epochs']
    batch_size = cfg['batch_size']
    filename="outputs/"
    for key, value in sorted(cfg.items()):
        filename += f"{key}={value}_"
    filename = filename[:-1] + ".txt"  # Remove the last underscore and add .pt
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
    
    
    indices_testing=list(range(test_data.shape[0] - T))
    test_data_batch = torch.stack([test_data[s : s + T, :, :] for s in indices_testing], dim=0)
    idiot_prediction=torch.stack([test_data[s + T-1, :, :] for s in indices_testing], dim=0) #We simply predict the previous change
    desired_output_test = torch.stack([test_data[s + T, :, :] for s in indices_testing], dim=0) #We simply predict 
    idiot_error=criterion(desired_output_test,idiot_prediction)
    training_avg_20= []
    if save_output:
        #Make sure the file exists and is empty
        with open(filename, 'w') as f:
            f.write("Epoch, train Loss, test loss\n")
    test_losses= []
    for epoch in range(num_epochs):
        start_idx=np.random.randint(0, train_data.shape[0] - T - 1,size=batch_size)
        train_data_batch = torch.stack([train_data[s : s + T, :, :] for s in start_idx], dim=0)
        desired_output = torch.stack([train_data[s + T, :, :] for s in start_idx], dim=0)
        output, _ = model(train_data_batch)          # unpack

        loss = criterion(output, desired_output)  
        training_avg_20.append(loss.item())
        if len(training_avg_20) > 20:
            training_avg_20.pop(0)        
        for name, param in model.named_parameters():
            if 'weight' in name:
                loss += l2_penalty* torch.sum(param ** 2)
        opt.zero_grad()  # clean the slate  
        loss.backward()  # compute gradients  
        opt.step()       # update parameters
        if save_model_state is not None and save_model_state != False:
            if epoch==save_model_state: 
                save_model(model, cfg, epoch=save_model_state, base_dir="outputs/")
                print("Saved model state at epoch", save_model_state)
                sys.exit(0)  # Exit after saving the model state
        if (epoch+1) % 20 == 0 or epoch == 0:  # Print every 5 epochs (adjust as you like)
            print(f'Epoch [{epoch + 1}/{cfg["num_epochs"]}], Loss: {loss.item():.4f}')
            training_avg= np.mean(training_avg_20)
            model.eval()
            with torch.no_grad():
                output_test,_ = model(test_data_batch)
                loss = criterion(output_test, desired_output_test)
                testloss= loss.item()
                test_losses.append(testloss)
                idiotloss=idiot_error.item()
            print(f'Test Loss (MSE, average over test set): {testloss}, Idiotic: {idiotloss:.4f}')
            if save_output:
                with open(filename, 'a') as f:
                    f.write(f"{epoch + 1}, {np.mean(training_avg):.4f}, {loss.item():.4f}\n")
            biggerthanlast10= False  # If there are less than 10 test losses, we assume the model is still learning
            for i in range(1,12):  # Check if the last 10 test losses are bigger than the current one
                if len(test_losses) >= i and testloss<test_losses[-i]:
                    biggerthanlast10=False
            
            if (idiotloss< testloss and epoch > 500) or (3*idiotloss<testloss and epoch>200) or biggerthanlast10:  # If the idiot prediction is better than the model prediction even after a lot of epochs
                print("Warning: Idiot prediction is better than model prediction, or no more learning. Cancelling training.")
                break
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
    params_test=params_test_and_valid[0:100,:,:]
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
        "L2_penalty": 1e-5,  # L2 regularization term
    }
    model = train_model(params_train, cfg,adjaceny_matrix,params_test,save_output=False)
