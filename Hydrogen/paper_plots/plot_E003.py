import numpy as np
import h5py
from matplotlib import pyplot as plt
from utils import compute_hhg_spectrum
import matplotlib
import warnings
warnings.filterwarnings("ignore")

font = {'size'   : 14}
matplotlib.rcParams['axes.titlepad'] = 20

matplotlib.rc('font', **font)

E0 = 0.03
omega = 0.057

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    t_stop=float(sys.argv[1])+1e-4
except:
    t_stop =350

filename_rothe_01="../hydrogen_simulations_SIMPLIFIED_shift/outputs/NEWNEW/data_E00.030_omega0.057_invRmu100_dt0.20_epsilon3.00e-01_frozenCore.h5"

with h5py.File(filename_rothe_01, "r") as data_file:
    time_points_rothe_full=np.array(data_file["times"])
    dipole_moment_rothe_01=np.array(data_file["dpm"])
    lens=np.zeros(len(time_points_rothe_full))
    parameters_0=np.array(data_file["parameters_t=0.20"])

    for i,timep in enumerate(time_points_rothe_full):
        lens[i]=(len(np.array(data_file["coefficients_t=%.2f"%timep])))
        lens[i]=(len(np.array(data_file["parameters_t=%.2f"%timep]))//4)
    N_T_f_01=np.max(lens)
    err_t0=np.array(data_file["rothe_error"])[-1]
    print(N_T_f_01,err_t0)
    rothe_error=np.array(data_file["rothe_error"])

time_points_rothe = time_points_rothe_full[time_points_rothe_full <= t_stop]
dipole_moment_rothe_01 = dipole_moment_rothe_01[time_points_rothe_full <= t_stop]
filename_rothe_05="../hydrogen_simulations_SIMPLIFIED_shift/outputs/NEWNEW/data_E00.030_omega0.057_invRmu100_dt0.20_epsilon3.00e-02_frozenCore.h5"
with h5py.File(filename_rothe_05, "r") as data_file:
    time_points_rothe_full=np.array(data_file["times"])
    dipole_moment_rothe_05=np.array(data_file["dpm"])
    lens=np.zeros(len(time_points_rothe_full))
    for i,timep in enumerate(time_points_rothe_full):
        lens[i]=(len(np.array(data_file["coefficients_t=%.2f"%timep])))
    N_T_f_05=np.max(lens)
    err_t0=np.array(data_file["rothe_error"])[-1]
    rothe_error=np.array(data_file["rothe_error"])
    print(N_T_f_05,err_t0)

time_points_rothe = time_points_rothe_full[time_points_rothe_full <= t_stop]
dipole_moment_rothe_05 = dipole_moment_rothe_05[time_points_rothe_full <= t_stop]


filename_rothe_25="../hydrogen_simulations_SIMPLIFIED_shift/outputs/NEWNEW/data_E00.030_omega0.057_invRmu100_dt0.20_epsilon5.00e-03_frozenCore.h5"

with h5py.File(filename_rothe_25, "r") as data_file:
    time_points_rothe_full=np.array(data_file["times"])
    dipole_moment_rothe_25=np.array(data_file["dpm"])
    lens=np.zeros(len(time_points_rothe_full))
    for i,timep in enumerate(time_points_rothe_full):
        lens[i]=(len(np.array(data_file["coefficients_t=%.2f"%timep])))
    N_T_f_25=np.max(lens)
    err_t0=np.array(data_file["rothe_error"])[-1]
    print(N_T_f_25,err_t0)

time_points_rothe = time_points_rothe_full[time_points_rothe_full <= t_stop]
dipole_moment_rothe_25 = dipole_moment_rothe_25[time_points_rothe_full <= t_stop]




DVR=np.load("../DVR_reference_data/length_E0=0.03_omega=0.057_rmax=600_N=1200_lmax=20_dt=0.2.npz")
time_points_DVR = DVR["time_points"]
dipole_moment_DVR_invR = -DVR["expec_z"]
time_points_DVR = time_points_DVR[time_points_DVR <= t_stop]

from os import listdir
time_points_coccia=np.load("../tdse_pyscf/time_points_E=0.03.txt.npy")#[::2]
z_t_coccia_noAbsorber=np.load("../tdse_pyscf/z_E=0.03_absorber=0.txt.npy")#[::2]

z_t_coccia_Absorber=np.load("../tdse_pyscf/z_E=0.03_absorber=1.txt.npy")#[::2]

import scipy
from scipy import signal
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
Up = E0**2 / (4 * omega**2)
Ip = 0.5
Ecutoff = Ip + 3.17 * Up
print("Ecutoff: %f"%(Ecutoff/omega))
hann_window=True
omega_s_R, Px_R = compute_hhg_spectrum(time_points_rothe[:-1], (-dipole_moment_rothe_01)[:-1], hann_window=hann_window)
omega_s_R, Px_R2 = compute_hhg_spectrum(time_points_rothe[:-1], (-dipole_moment_rothe_05)[:-1], hann_window=hann_window)
omega_s_R25, Px_R3 = compute_hhg_spectrum(time_points_rothe[:-1], (-dipole_moment_rothe_25)[:-1], hann_window=hann_window)
omega_s_D, Px_D_invr = compute_hhg_spectrum(time_points_DVR, (-dipole_moment_DVR_invR), hann_window=hann_window)
omega_s_C, Px_C_Absorber = compute_hhg_spectrum(np.array(time_points_coccia)[:-1],-np.array(z_t_coccia_Absorber)[:-1], hann_window=hann_window)
omega_s_C, Px_C_NoAbsorber = compute_hhg_spectrum(np.array(time_points_coccia)[:-1],-np.array(z_t_coccia_noAbsorber)[:-1], hann_window=hann_window)

fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,sharey=True)
ax1.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_invr, color="black", label=r"DVR")
ax1.semilogy(omega_s_R / omega, omega_s_R**2*Px_R, color="blue",linestyle="dashed", label=r"Rothe ($M_{max}=%d$)"%N_T_f_01)
ax1.semilogy(omega_s_R / omega, omega_s_R**2*Px_R2, color="green",linestyle="dashed", label=r"Rothe ($M_{max}=%d$)"%N_T_f_05)
ax1.semilogy(omega_s_R / omega, omega_s_R**2*Px_R3, color="red",linestyle="dashed", label=r"Rothe ($M_{max}=%d$)"%N_T_f_25)
ax2.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_invr, color="black")
ax2.semilogy(omega_s_R / omega, omega_s_R**2*Px_R3, color="red",linestyle="dashed")

ax2.semilogy(omega_s_C / omega, omega_s_C**2*Px_C_Absorber, color="purple",linestyle="dashed",label="6-aug-ccPVTZ+8K (HLM)")
ax2.semilogy(omega_s_C / omega, omega_s_C**2*Px_C_NoAbsorber, color="orange",linestyle="dashed",label="6-aug-ccPVTZ+8K (no HLM)")
ax1.axvline(Ecutoff / omega, linestyle="dotted",color="grey")
ax2.axvline(Ecutoff / omega, linestyle="dotted",color="grey")


ax1.set_xlim(1, int(2*Ecutoff / omega) + 1)
ax1.set_xticks(np.arange(1, int(2*Ecutoff / omega) + 1, step=6))
ax1.set_ylim(1e-15,10**(1.2))
ax1.set_ylabel("Intensity")
ax2.set_xlabel("Harmonic order")
ax2.set_ylabel("Intensity")
#ax1.legend(ncol=1, handletextpad=0,labelspacing=0,loc="lower right",bbox_to_anchor=(0,0))
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lgd=fig.legend(lines, labels,loc='lower center', ncol=2,bbox_to_anchor=(0.5,-0))
#plt.suptitle(r" HHG spectrum ($E_0=0.03$, $N_c=3$, $\Delta t=0.2$, $\omega=0.057$)")

plt.tight_layout()

plt.subplots_adjust(bottom=0.35)
plt.savefig("hhg_E0=0.03.png",dpi=300)
plt.savefig("hhg_E0=0.03.svg")

plt.savefig("hhg_E0=0.03.pdf")
plt.show()
from calculate_HHG_Delta import *
print("%d Rothe functions"%N_T_f_01)
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R,omega,Ecutoff)

print("%d Rothe functions"%N_T_f_05)
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R2,omega,Ecutoff)


print("%d Rothe functions"%N_T_f_25)
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R3,omega,Ecutoff)


print("FGB (absorber)")
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_Absorber,omega,Ecutoff)

print("FGB (no absorber)")
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_NoAbsorber,omega,Ecutoff)


print("----------------")

print("%d Rothe functions"%N_T_f_01)
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R,omega,Ecutoff)

print("%d Rothe functions"%N_T_f_05)
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R2,omega,Ecutoff)


print("%d Rothe functions"%N_T_f_25)
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R3,omega,Ecutoff)


print("FGB (absorber)")
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_Absorber,omega,Ecutoff)

print("FGB (no absorber)")
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_NoAbsorber,omega,Ecutoff)


print("----------------")

print("%d Rothe functions"%N_T_f_01)
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R,omega,Ecutoff)

print("%d Rothe functions"%N_T_f_05)
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R2,omega,Ecutoff)


print("%d Rothe functions"%N_T_f_25)
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R3,omega,Ecutoff)


print("FGB (absorber)")
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_Absorber,omega,Ecutoff)

print("FGB (no absorber)")
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_NoAbsorber,omega,Ecutoff)
