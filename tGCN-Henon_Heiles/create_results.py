
from __future__ import annotations

import os
import re
import copy
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
T=int(sys.argv[1])
use_gcn = sys.argv[2].lower() == 'true'  # Convert string to boolean
train_timesteps = 1000  
base_dir = "outputs"  


def _find_checkpoint(t, run_gcn, dir_) :
    """Return .pt path and *cfg* extracted from the filename.

    """
    t_pat = f"T={t}_"
    gcn_pat = f"run_GCN={run_gcn}_"
    regex = re.compile(rf".*{re.escape(t_pat)}.*{re.escape(gcn_pat)}.*\.pt$") #Regex that matches the filename pattern

    best = None  
    cfg = None

    for fname in os.listdir(dir_):
        if not regex.match(fname):
            continue
        m = re.search(r"num_epochs=(\d+)_", fname)
        if not m:
            continue
        epochs = int(m.group(1))
        if best is None or epochs > best[0]:
            best = (epochs, Path(dir_) / fname)
            cfg = _parse_cfg_from_filename(fname)
    if best is None or cfg is None:
        raise FileNotFoundError(f"No checkpoint for T={t}, run_GCN={run_gcn} in {dir_}")
    return best[1], cfg


def _parse_cfg_from_filename(fname):
    pat = (
        r"T=(\d+)_"
        r"batch_size=(\d+)_"
        r"l2_penalty=([\d\.eE-]+)_"
        r"learning_rate=([\d\.eE-]+)_"
        r"num_epochs=(\d+)_"
        r"num_layers_GCN=(\d+)_"
        r"num_layers_LSTM=(\d+)_"
        r"num_nodes=(\d+)_"
        r"run_GCN=(True|False)_"
        r"size_GCN=(\d+)_"
        r"size_GCN_out=(\d+)_"
        r"size_LSTM=(\d+)\.pt"
    )
    m = re.match(pat, fname)
    if m is None:
        raise ValueError(f"Filename does not match pattern: {fname}")
    (t, batch_size, l2_penalty, lr, num_epochs, nl_gcn, nl_lstm, nnodes,
     run_gcn, sz_gcn, sz_gcn_out, sz_lstm) = m.groups()
    return {
        'T': int(t),
        'batch_size': int(batch_size),
        'l2_penalty': float(l2_penalty),
        'learning_rate': float(lr),
        'num_epochs': int(num_epochs),
        'num_layers_GCN': int(nl_gcn),
        'num_layers_LSTM': int(nl_lstm),
        'num_nodes': int(nnodes),
        'run_GCN': run_gcn == "True",
        'size_GCN': int(sz_gcn),
        'size_GCN_out': int(sz_gcn_out),
        'size_LSTM': int(sz_lstm),
    }



def _load_raw_data(dim: int = 3):
    if dim == 2:
        infile = np.load("nonlinear_coefficients_dimension=2_ngauss_init=29.npz")
    elif dim == 3:
        infile = np.load("nonlinear_coefficients_dimension=3_ngauss_init=28.npz")
    else:
        raise ValueError("dim must be 2 or 3")
    return infile['L'], infile['K'], infile['mu'], infile['p']



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Locate checkpoint + cfg
    ckpt_path, cfg = _find_checkpoint(T, use_gcn, base_dir)
    print(f"Using checkpoint {ckpt_path}")
    from own_tgcn import own_TGCN, build_adjacency_matrix, preprocess_data  # noqa: E402

    num_nodes = cfg['num_nodes']

    L, K, mu, p = _load_raw_data(dim=3)
    num_gaussians= L.shape[1]
    params_train, params_rest, mean, std = preprocess_data(L, K, mu, p, train_timesteps)
    dparams = np.concatenate([params_train, params_rest], axis=0)  # (time, nodes, 1)
    dparams_torch = torch.tensor(dparams)
    num_coefficients = num_nodes // num_gaussians

    A = build_adjacency_matrix(num_gaussians, num_coefficients)
    model = own_TGCN(
        num_layers_GCN=cfg['num_layers_GCN'],
        num_layers_LSTM=cfg['num_layers_LSTM'],
        size_LSTM=cfg['size_LSTM'],
        size_GCN=cfg['size_GCN'],
        size_GCN_out=cfg['size_GCN_out'],
        adjacency_matrix=A,
        num_nodes=cfg['num_nodes'],
        run_GCN=cfg['run_GCN'],
        T=cfg['T'],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

   

    errors = []
    idiot_errors  = [] # The "previous delta"  baseline
    nochange = []  # for the "nothing" baseline
    with torch.no_grad():
        for idx in range(T, dparams_torch.shape[0]):
            window = dparams_torch[idx - T:idx, :, :] 
            truth = dparams_torch[idx, :, :]
            pred, _ = model(window.unsqueeze(0)) 
            mse = torch.mean((pred.squeeze(0) - truth) ** 2).item()
            errors.append(mse)
        for idx in range(1, dparams_torch.shape[0]):
            truth = dparams_torch[idx, :, :]
            idiot_pred = dparams_torch[idx - 1, :, :]
            nochange_mse = torch.mean((truth) ** 2).item()
            nochange.append(nochange_mse)
            idiot_mse = torch.mean((idiot_pred - truth) ** 2).item()
            idiot_errors.append(idiot_mse)

    errors = np.asarray(errors)
    idiot_errors = np.asarray(idiot_errors)

    # 5. Persist results
    out_model = f"MSE_T={T}_use_gcn={use_gcn}.npz"
    np.savez_compressed(out_model, mse=errors)

    out_idiot = "MSE_idiot.npz"
    np.savez_compressed(out_idiot, mse_idiot=idiot_errors)
    out_nochange = "MSE_nochange.npz"
    np.savez_compressed(out_nochange, mse_nochange=nochange)
