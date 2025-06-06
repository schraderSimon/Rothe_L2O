# Rothe_L2O

In this project, we implement Learning to Optimize (L2O) to learn a LSTMs and n-SSMs, that aim to optimize
  1. Quadratic functions (as a test). This is done using LSTMS ans n-SSMs. See the directory [L2O_quadratic](L2O_quadratic).
  2. The Rothe error for the Hydrogen atom at different field strengths, time steps, and wave function qualities. This was done using LSTMs. See the directory [L2O_Hydrogen](L2O_Hydrogen).

Furthermore, we have implemented a T-GCN to predict the best starting guess for the nonlinear coefficients at the next time step for the Henon-Heiles model in 3D. See the directory [tGCN-Henon_Heiles](tGCN-Henon_Heiles).

## Usage
1. For the test problems, run the models (L2O_quadratic_singleVariable.py, ssm_l2o.py and s4_ssm.py), then the respective run_L2O_experiment_"model".py, before finally running
   plot_L2O_experiment_"model".py. Each "model".py takes command line arguments T (unrolling depth) and k (weight scaling). The experiement codes take three command lines
   T, k and type ('training' or 'testing'). The plotting codes, similar to the model codes, take T and k as command line arguments.

   ~~~
   ssm_l2o.py 20 1
   run_L2O_experiment_SSM.py 20 1 'training'
   run_L2O_experiment_SSM.py 20 1 'testing'
   plot_L2O_experiment_SSM.py 20 1
   ~~~
2. For L2O_Hydrogen, run `L2O_hydrogen.py narrow` and `L2O_hydrogen.py wide` for the sweep over all metaparameters.
   To find the configuration for the best metaprameters, run `python3 analyze_runs.py narrow` and `python3 analyze_runs.py wide`.
   Run `train_and_evaluate_best_model.py` to re-train and re-evaluate the best models.
   Finally, run `make_plots.py narrow` and `make_plots.py wide` to reproduce the plots.
3. For the t-GCN, run `train_models.py` to do a sweep over the metaparameters.
   Run `find_best_values.py 5 True`, `find_best_values.py 5 False`, `find_best_values.py 10 True`, `find_best_values.py 10 False` to find the best metarparameters & epoch. This also re-trains the best model and produces an output.
   Run `create_results.py 5 True`, `create_results.py 10 True`, `create_results.py 5 False`, `create_results.py 10 False`, to create the MSE data that is visualized in the paper.
   Finally, run `plot_results True` and `plot_results False` to reproduce the plots.
