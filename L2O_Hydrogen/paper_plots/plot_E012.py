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

E0 = 0.12
omega = 0.057
Up = E0**2 / (4 * omega**2)
Ip = 0.5
Ecutoff = Ip + 3.17 * Up

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    t_stop=float(sys.argv[1])
except:
    t_stop =350
filename_rothe_01="../hydrogen_simulations_SIMPLIFIED_shift/outputs/NEWNEW/data_E00.120_omega0.057_invRmu100_dt0.20_epsilon3.00e+00_frozenCore.h5"
with h5py.File(filename_rothe_01, "r") as data_file:
    time_points_rothe_full=np.array(data_file["times"])
    dipole_moment_rothe_01=np.array(data_file["dpm"])
    lens=np.zeros(len(time_points_rothe_full))
    for i,timep in enumerate(time_points_rothe_full):
        lens[i]=(len(np.array(data_file["coefficients_t=%.2f"%timep])))
    N_T_f_01=np.max(lens)
    errs=np.array(data_file["rothe_error"])
    err_t0=errs[-1]
    print("%f %d"%(err_t0,N_T_f_01))
time_points_rothe = time_points_rothe_full[time_points_rothe_full <= t_stop]
dipole_moment_rothe_01 = dipole_moment_rothe_01[time_points_rothe_full <= t_stop]
filename_rothe_05="../hydrogen_simulations_SIMPLIFIED_shift/outputs/NEWNEW/data_E00.120_omega0.057_invRmu100_dt0.20_epsilon6.00e+00_frozenCore.h5"
with h5py.File(filename_rothe_05, "r") as data_file:
    time_points_rothe_full=np.array(data_file["times"])
    dipole_moment_rothe_05=np.array(data_file["dpm"])
    lens=np.zeros(len(time_points_rothe_full))
    for i,timep in enumerate(time_points_rothe_full):
        lens[i]=(len(np.array(data_file["coefficients_t=%.2f"%timep])))
    N_T_f_05=np.max(lens)
    #N_T_f_01=103

    err_t0=np.array(data_file["rothe_error"])[-1]
    print("%f %d"%(err_t0,N_T_f_05))
dipole_moment_rothe_05 = dipole_moment_rothe_05[time_points_rothe_full <= t_stop]

filename_rothe_25="../hydrogen_simulations_SIMPLIFIED_shift/outputs/NEWNEW/data_E00.120_omega0.057_invRmu100_dt0.20_epsilon1.20e+01_frozenCore.h5"
with h5py.File(filename_rothe_25, "r") as data_file:
    time_points_rothe_25=np.array(data_file["times"])

    dipole_moment_rothe_25=np.array(data_file["dpm"])
    lens=np.zeros(len(time_points_rothe_25))
    for i,timep in enumerate(time_points_rothe_25):
        lens[i]=(len(np.array(data_file["coefficients_t=%.2f"%timep])))
    N_T_f_25=np.max(lens)
    #N_T_f_25=104

    err_t0=np.array(data_file["rothe_error"])[-1]
    print("%f %d"%(err_t0,N_T_f_25))
dipole_moment_rothe_25 = dipole_moment_rothe_25[time_points_rothe_25 <= t_stop]
time_points_rothe_25= time_points_rothe_25[time_points_rothe_25 <=t_stop]
DVR=np.load("../DVR_reference_data/length_E0=0.12_omega=0.057_rmax=600_N=1200_lmax=70_dt=0.2.npz")
time_points_DVR = DVR["time_points"]
dipole_moment_DVR_invR = -DVR["expec_z"]#print(time_points_DVR)
from os import listdir
time_points_coccia=np.load("../tdse_pyscf/time_points_E=0.12.txt.npy")#[::2]
z_t_coccia_noAbsorber=np.load("../tdse_pyscf/z_E=0.12_absorber=0.txt.npy")[time_points_coccia <= t_stop]
z_t_coccia_Absorber=np.load("../tdse_pyscf/z_E=0.12_absorber=1.txt.npy")[time_points_coccia <= t_stop]
time_points_coccia = time_points_coccia[time_points_coccia <= t_stop]

import scipy
from scipy import signal
from scipy.interpolate import splrep, BSpline
from scipy import interpolate

hann_window=True
omega_s_R, Px_R = compute_hhg_spectrum(time_points_rothe[:-1], (-dipole_moment_rothe_01)[:-1], hann_window=hann_window)
omega_s_R, Px_R2 = compute_hhg_spectrum(time_points_rothe[:-1], (-dipole_moment_rothe_05)[:-1], hann_window=hann_window)
omega_s_R_25, Px_R3 = compute_hhg_spectrum(time_points_rothe_25[:-1], (-dipole_moment_rothe_25)[:-1], hann_window=hann_window)
omega_s_D, Px_D_invr = compute_hhg_spectrum(time_points_DVR, (-dipole_moment_DVR_invR), hann_window=hann_window)
omega_s_C, Px_C_Absorber = compute_hhg_spectrum(np.array(time_points_coccia)[:-1],-np.array(z_t_coccia_Absorber)[:-1], hann_window=hann_window)
omega_s_C, Px_C_NoAbsorber = compute_hhg_spectrum(np.array(time_points_coccia)[:-1],-np.array(z_t_coccia_noAbsorber)[:-1], hann_window=hann_window)

fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,sharey=True)

ax1.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_invr, color="black", label=r"DVR")
ax1.semilogy(omega_s_R / omega, omega_s_R**2*Px_R, color="red",linestyle="dashed", label=r"Rothe ($M_{max}=%d$)"%N_T_f_01)
ax1.semilogy(omega_s_R / omega, omega_s_R**2*Px_R2, color="green",linestyle="dashed", label=r"Rothe ($M_{max}=%d$)"%N_T_f_05)
ax1.semilogy(omega_s_R_25 / omega, omega_s_R_25**2*Px_R3, color="blue",linestyle="dashed", label=r"Rothe ($M_{max}=%d$)"%N_T_f_25)
ax1.axvline(Ecutoff / omega, linestyle="dotted",color="grey")
ax2.axvline(Ecutoff / omega, linestyle="dotted",color="grey")
ax2.semilogy(omega_s_D / omega, omega_s_D**2*Px_D_invr, color="black")
ax2.semilogy(omega_s_R / omega, omega_s_R**2*Px_R, color="red",linestyle="dashed")

ax2.semilogy(omega_s_C / omega, omega_s_C**2*Px_C_Absorber, color="purple",linestyle="dashed",label="6-aug-ccPVTZ+8K (HLM)")
ax2.semilogy(omega_s_C / omega, omega_s_C**2*Px_C_NoAbsorber, color="orange",linestyle="dashed",label="6-aug-ccPVTZ+8K (no HLM)")


ax1.set_xlim(1, int(Ecutoff / omega) + 15)
ax1.set_xticks(np.arange(1, int(Ecutoff / omega) + 25, step=8))
ax1.set_ylim(1e-9,10**(5))
ax1.set_ylabel("Intensity")
ax2.set_xlabel("Harmonic order")
ax2.set_ylabel("Intensity")
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lgd=fig.legend(lines, labels,loc='lower center', ncol=2,bbox_to_anchor=(0.5,-0.0))
plt.tight_layout()

plt.subplots_adjust(bottom=0.35)
plt.savefig("hhg_E0=0.12.png",dpi=300)
plt.savefig("hhg_E0=0.12.svg")

plt.savefig("hhg_E0=0.12.pdf")
plt.show()


from calculate_HHG_Delta import *
print("%d Rothe functions"%N_T_f_25)
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R3,omega,Ecutoff)
print("%d Rothe functions"%N_T_f_05)
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R2,omega,Ecutoff)
print("%d Rothe functions"%N_T_f_01)
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R,omega,Ecutoff)





print("FGB (absorber)")
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_Absorber,omega,Ecutoff)

print("FGB (no absorber)")
calculate_HHG_Delta(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_NoAbsorber,omega,Ecutoff)


print("----------------")
print("%d Rothe functions"%N_T_f_25)
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R3,omega,Ecutoff)
print("%d Rothe functions"%N_T_f_05)
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R2,omega,Ecutoff)

print("%d Rothe functions"%N_T_f_01)
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R,omega,Ecutoff)





print("FGB (absorber)")
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_Absorber,omega,Ecutoff)

print("FGB (no absorber)")
calculate_HHG_Upsilon(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_NoAbsorber,omega,Ecutoff)


print("----------------")
print("%d Rothe functions"%N_T_f_25)
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R3,omega,Ecutoff)
print("%d Rothe functions"%N_T_f_05)
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R2,omega,Ecutoff)

print("%d Rothe functions"%N_T_f_01)
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_R**2*Px_R,omega,Ecutoff)





print("FGB (absorber)")
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_Absorber,omega,Ecutoff)

print("FGB (no absorber)")
calculate_HHG_D(omega_s_D,omega_s_R,omega_s_D**2*Px_D_invr,omega_s_C**2*Px_C_NoAbsorber,omega,Ecutoff)
