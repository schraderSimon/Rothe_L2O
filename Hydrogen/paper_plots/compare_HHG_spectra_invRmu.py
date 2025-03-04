import numpy as np
import h5py
from matplotlib import pyplot as plt
from utils import compute_hhg_spectrum
import matplotlib
font = {'size'   : 14}

matplotlib.rc('font', **font)

E0 = 0.12
omega = 0.057
nc = 6
dt_list = [0.2]
r_max = 300.0
n_r = 1000
l_max = 10
potential_type = f"Coulomb"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
t_stop = 700


DVR=np.load("../DVR_reference_data/length_E0=0.12_omega=0.057_rmax=600_N=1200_lmax=70_dt=0.2.npz")
time_points_DVR = DVR["time_points"]
dipole_moment_DVR_invR = -DVR["expec_z"]


DVR=np.load("../DVR_reference_data/length_E0=0.12_omega=0.057_rmax=600_N=1200_lmax=70_dt=0.2_mu=100.npz")
dipole_moment_DVR_erfmu100 = -DVR["expec_z"]


DVR=np.load("../DVR_reference_data/length_E0=0.12_omega=0.057_rmax=600_N=1200_lmax=70_dt=0.2_mu=10.npz")
dipole_moment_DVR_erfmu10 = -DVR["expec_z"]

from os import listdir


import scipy
from scipy import signal
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
hann_window=True
print(time_points_DVR)
print(dipole_moment_DVR_invR)
dt=time_points_DVR[1]-time_points_DVR[0]
omega_s_D, Px_D_invr = compute_hhg_spectrum(time_points_DVR, -dipole_moment_DVR_invR, hann_window=hann_window)
omega_s_erfMu100, Px_D_erfMu100 = compute_hhg_spectrum(time_points_DVR, -dipole_moment_DVR_erfmu100, hann_window=hann_window)
omega_s_erfMu10, Px_D_erfMu10 = compute_hhg_spectrum(time_points_DVR, -dipole_moment_DVR_erfmu10, hann_window=hann_window)

fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,sharey=True)
ax2.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_invr, color="black",label=r"$\mu=\infty$")
ax2.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_erfMu100,"--", color="red",label=r"$\mu=100$")
ax1.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_invr, color="black",label=r"$\mu=\infty$")
ax1.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_erfMu10,"--", color="green",label=r"$\mu=10$")

Up = E0**2 / (4 * omega**2)
Ip = 0.5
Ecutoff = Ip + 3.17 * Up
print(Ecutoff / omega)
#ax1.axvline(Ecutoff / omega, linestyle="dotted", color="green", label=r"$E_{cutoff}$")
ax1.set_xlim(1, int(Ecutoff / omega) + 15)
ax1.set_xticks(np.arange(1, int(Ecutoff / omega) + 25, step=8))
ax1.set_ylim(1e-9,10**(5))
ax1.set_ylabel("Intensity")
ax2.set_xlabel("Harmonic order")
ax2.set_ylabel("Intensity")

ax1.legend()
ax2.legend()

plt.tight_layout()
plt.savefig("hhg_example.png",dpi=300)
plt.savefig("hhg_example.svg")

plt.savefig("hhg_example.pdf")
plt.show()
