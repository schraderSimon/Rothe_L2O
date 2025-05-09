import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
infile_2D=np.load("nonlinear_coefficients_dimension=2_ngauss_init=29.npz")
infile_3D=np.load("nonlinear_coefficients_dimension=3_ngauss_init=28.npz")

times_2D=infile_2D['times']
times_3D=infile_3D['times']

print(times_2D[0],times_2D[-1],len(times_2D))
print(times_3D[0],times_3D[-1],len(times_3D))
num_params_2D=(infile_2D["L"].shape[2]+infile_2D["K"].shape[2]+infile_2D["mu"].shape[2]+infile_2D["p"].shape[2])*infile_2D["L"].shape[1]
num_params_3D=(infile_3D["L"].shape[2]+infile_3D["K"].shape[2]+infile_3D["mu"].shape[2]+infile_3D["p"].shape[2])*infile_3D["L"].shape[1]
print("Number of parameters in 2D: ",num_params_2D)
print("Number of parameters in 3D: ",num_params_3D)
L_data_2D=infile_2D['L']
L_data_3D=infile_3D['L']
K_data_2D=infile_2D['K']
K_data_3D=infile_3D['K']
q_data_2D=infile_2D['mu']
q_data_3D=infile_3D['mu']
p_data_2D=infile_2D['p']
p_data_3D=infile_3D['p']
plt.plot(times_3D, L_data_3D[:,0,:], label='L (first Gaussian)')
plt.show()