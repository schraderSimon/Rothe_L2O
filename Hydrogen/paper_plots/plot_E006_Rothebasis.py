import numpy as np
import h5py
from matplotlib import pyplot as plt
from utils import compute_hhg_spectrum
import matplotlib
font = {'size'   : 18}
matplotlib.rcParams['axes.titlepad'] = 20

matplotlib.rc('font', **font)

E0 = 0.06
omega = 0.057
Up = E0**2 / (4 * omega**2)
Ip = 0.5
Ecutoff = Ip + 3.17 * Up

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
t_stop = 350
filename_rothe_01=filename="../hydrogen_simulations_SIMPLIFIED_shift/outputs/NEWNEW/data_E00.060_omega0.057_invRmu100_dt0.20_epsilon2.50e-02_frozenCore.h5"
with h5py.File(filename_rothe_01, "r") as data_file:
    time_points_rothe_full=np.array(data_file["times"])
    dipole_moment_rothe_01=np.array(data_file["dpm"])
    lens=np.zeros(len(time_points_rothe_full))
    for i,timep in enumerate(time_points_rothe_full):
        lens[i]=(len(np.array(data_file["coefficients_t=%.2f"%timep])))
    N_T_f_01=np.max(lens)
    err_t0=np.array(data_file["rothe_error"])[-1]
    print("%f %d"%(err_t0,N_T_f_01))

time_points_rothe = time_points_rothe_full[time_points_rothe_full <= t_stop]
dipole_moment_rothe_01 = dipole_moment_rothe_01[time_points_rothe_full <= t_stop]
DVR=np.load("../DVR_reference_data/length_E0=0.06_omega=0.057_rmax=600_N=1200_lmax=30_dt=0.2.npz")
time_points_DVR = DVR["time_points"]
dipole_moment_DVR_invR = -DVR["expec_z"]
from os import listdir
time_points_coccia=np.load("../build_efficient_basis/time_points_E=0.06_absorber=0.txt.npy")#[::2]
z_t_1=np.load("../build_efficient_basis/z_E=0.06_absorber=1_spacing=300,cutoff=1.0e-05_l=268.txt.npy")#[::2]
z_t_3=np.load("../build_efficient_basis/z_E=0.06_absorber=1_spacing=50,cutoff=1.0e-05_l=1067.txt.npy")#[::2]
z_t_coccia_Absorber=np.load("../tdse_pyscf/z_E=0.06_absorber=1.txt.npy")#[time_points_coccia <= t_stop]

time_points_coccia = time_points_coccia[time_points_coccia <= t_stop]

import scipy
from scipy import signal
from scipy.interpolate import splrep, BSpline
from scipy import interpolate

hann_window=True
omega_s_R, Px_R = compute_hhg_spectrum(time_points_rothe[:-1], (-dipole_moment_rothe_01)[:-1], hann_window=hann_window)
omega_s_Rb, Px_Rb1 = compute_hhg_spectrum(time_points_coccia, (-z_t_1), hann_window=hann_window)
omega_s_Rb, Px_Rb3 = compute_hhg_spectrum(time_points_coccia, (-z_t_3), hann_window=hann_window)
omega_s_Rc, Px_Rc = compute_hhg_spectrum(time_points_coccia, (-z_t_coccia_Absorber)[:-1], hann_window=hann_window)

omega_s_D, Px_D_invr = compute_hhg_spectrum(time_points_DVR, (-dipole_moment_DVR_invR), hann_window=hann_window)


fig, (ax1) = plt.subplots(1, 1,sharex=True,sharey=True,figsize=(9,5))

ax1.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_invr, color="black", label=r"DVR")
ax1.semilogy(omega_s_R / omega, omega_s_R**2*Px_R, color="red",linestyle="dashed", label=r"Rothe ($M(T_f)=%d$)"%N_T_f_01)
ax1.semilogy(omega_s_Rb / omega, omega_s_Rb**2*Px_Rb1, color="green",linestyle="dashed", label=r"N=300,$s_{\varepsilon}=10^{-5},M=268$")
ax1.semilogy(omega_s_Rb / omega, omega_s_Rb**2*Px_Rb3, color="blue",linestyle="dashed", label=r"N=50,$s_{\varepsilon}=10^{-5},M=1067$")
ax1.semilogy(omega_s_Rc / omega, omega_s_Rc**2*Px_Rc, color="purple",linestyle="dashed", label=r"6-aug-ccPVTZ+8K (HLM)")
ax1.axvline(Ecutoff / omega, linestyle="dotted",color="grey")

ax1.set_xlim(1, int(2*Ecutoff / omega) + 1)
ax1.set_xticks(np.arange(1, int(2*Ecutoff / omega) + 1, step=6))
ax1.set_ylim(1e-10,10**(2))
ax1.set_ylabel("Intensity")
ax1.set_xlabel("Harmonic order")
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lgd=fig.legend(lines, labels,loc='lower center', ncol=2,bbox_to_anchor=(0.5,-0.0))
plt.tight_layout()

plt.subplots_adjust(bottom=0.45)
plt.savefig("hhg_E0=0.06_RotheBasis.png",dpi=300)
plt.savefig("hhg_E0=0.06_RotheBasis.svg")

plt.savefig("hhg_E0=0.06_RotheBasis.pdf")
plt.show()
