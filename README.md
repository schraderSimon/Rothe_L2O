# Rothe_L2O

In this project, we implement Learning to Optimize (L2O) to learn a LSTMs and n-SSMs, that aim to optimize
  1. Quadratic functions (as a test). This is done using LSTMS ans n-SSMs. See the directory [L2O_quadratic](L2O_quadratic).
  2. The Rothe error for the Hydrogen atom at different field strengths, time steps, and wave function qualities. This was done using LSTMs. See the directory [L2O_Hydrogen](L2O_Hydrogen).

Furthermore, we have implemented a T-GCN to predict the best starting guess for the nonlinear coefficients at the next time step for the Henon-Heiles model in 3D. See the directory [tGCN-Henon_Heiles](tGCN-Henon_Heiles).

## Usage
1. For the test problems, run the models (L2O_quadratic_singleVariable.py, ssm_l2o.py and s4_ssm.py), then the respective run_L2O_experiment_"model".py, before finally running
   plot_L2O_experiment_"model".py. Each "model".py takes command line arguments T (unrolling depth) and k (weight scaling). The experiement codes take three command lines
   T, k and type ('training' or 'testing'). The plotting codes, similar to the model codes, take T and k as command line arguments.
   ~~ SHTSHS ~~
