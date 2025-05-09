import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
infile_2D=np.load("nonlinear_coefficients_dimension=2_ngauss_init=29.npz")
infile_3D=np.load("nonlinear_coefficients_dimension=3_ngauss_init=28.npz")

times_2D=infile_2D['times']
times_3D=infile_3D['times']
L_data_2D=infile_2D['L']
L_data_3D=infile_3D['L']
K_data_2D=infile_2D['K']
K_data_3D=infile_3D['K']
mu_data_2D=infile_2D['mu']
mu_data_3D=infile_3D['mu']
p_data_2D=infile_2D['p']
p_data_3D=infile_3D['p']
print(L_data_2D.shape)
print(mu_data_2D.shape)
plt.plot(times_3D, L_data_3D[:,0,:], label='L (first Gaussian)')
plt.show()