from own_tgcn import *
import os
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
    train_timesteps = 1000  # Number of timesteps to use for training
    params_train, params_test_and_valid, mean, std = preprocess_data(L_data, K_data, mu_data, p_data, train_timesteps)
    params_test=params_test_and_valid[0:200,:,:]
    batch_sizes=[16, 32, 64]  # Different batch sizes to test
    T=[5,10, 20]  # Different unrolling parameters to test
    hidden_sizes=[64, 128, 256]  # Different hidden sizes to test
    run_GCNs=[False,True]  # Whether to run GCNs or not
    L2_penalties=[1e-4,1e-3,1e-2]  # Different L2 penalties to test
    num_layers_GCNs=[2, 3]  # Different number of GCN layers to test
    num_layers_LSTMs=[1,2, 3]  # Different number of LSTM layers to test
    #num_layers_LSTMs=[2]
    learning_rates= [1e-3]  # Different learning rates to test
    for run_GCN in run_GCNs:
            for learning_rate in learning_rates:
                for t in T:
                    for hidden_size in hidden_sizes:
                        for batch_size in batch_sizes:
                            for l2_penalty in L2_penalties:
                                for num_layers_LSTM in num_layers_LSTMs:
                                    for num_layers_GCN in num_layers_GCNs:
                                        if num_layers_GCN == 2 and run_GCN is False:
                                            continue # Skip this configuration, should only run a single time for pure LSTM
                                        torch.manual_seed(42)  # For reproducibility
                                        np.random.seed(42)  # For reproducibility
                                        cfg = {
                                            'num_epochs': 2000,
                                            'batch_size': batch_size,
                                            'learning_rate': learning_rate,
                                            'T': t,  # Unrolling parameter
                                            "num_layers_GCN": num_layers_GCN,
                                            "num_layers_LSTM": num_layers_LSTM,
                                            "size_LSTM": hidden_size,
                                            "size_GCN": hidden_size,
                                            "size_GCN_out": 1,
                                            "num_nodes": num_nodes,
                                            "run_GCN": run_GCN,
                                            "l2_penalty": l2_penalty,
                                        }
                                        filename="outputs/"
                                        for key, value in sorted(cfg.items()):
                                            filename += f"{key}={value}_"
                                        filename = filename[:-1] + ".txt"  # Remove the last underscore and add .pt
                                        if os.path.exists(filename):
                                            print(f"Skipping existing configuration: {filename}")
                                            continue
                                        print(f"Training with params: {cfg}")
                                        model = train_model(params_train, cfg,adjaceny_matrix,params_test,save_model_state=False,save_output=True)
