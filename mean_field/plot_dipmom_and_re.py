import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
import os
import scipy
import sys
import re

def compute_hhg_spectrum(time_points, dipole_moment, hann_window=True):
    dip=np.array(dipole_moment)-dipole_moment[0]
    #dip = scipy.signal.detrend(dipole_moment, type="constant")
    if hann_window:
        Px = (
            np.abs(
                scipy.fftpack.fftshift(
                    scipy.fftpack.fft(
                        dip * np.sin(np.pi * time_points / time_points[-1]) ** 2
                    )
                )
            )
            ** 2
        )
    else:
        Px = np.abs(scipy.fftpack.fftshift(scipy.fftpack.fft(dip))) ** 2

    dt = time_points[1] - time_points[0]
    omega = (
        scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(time_points)))
        * 2
        * np.pi
        / dt
    )

    return omega, Px

# Get command-line parameters
molecule = sys.argv[1]
fieldstrength_int= np.sqrt(int(sys.argv[2]))
fieldstrengh = 0.0534*fieldstrength_int
omega_ref = 0.06075

try:
    method = sys.argv[3]
except:
    method = "HF"

grid_solution_exists = False
try:
    if method == "HF":
        name_grid = "/home/simonsch/projects/TDHF/grid-methods/examples/grid_solution_%s_%.3f.npz" % (molecule, fieldstrengh)
    elif method == "DFT":
        name_grid = "/home/simonsch/projects/TDHF/grid-methods/examples/grid_solution_%s_%.3f_DFT.npz" % (molecule, fieldstrengh)
    orbitals_grid = np.load(name_grid)
    times_grid = orbitals_grid['times']
    dipmom_grid = orbitals_grid['xvals']
    grid_solution_exists = True
except:
    print("No grid solution found")

times_rothe_list = []
errors_rothe_list = []
dipmom_rothe_list = []
directory = os.getcwd()
pattern = r"WF_%s_[A-Za-z0-9]+_\d+\.\d+_\d+_(\d+)_(\d+)_(\d\.\d+e[+-]\d+)\.npz" % method
files = [f for f in os.listdir(directory) if f.startswith("WF_%s_%s_%.4f_" % (method, molecule, fieldstrengh))]
lengths = []
maxiters = []
epsilons = []
initlengths = []
finaltime_gaussians = []
hhg_Rothe_list = []
omegavals = []
hhg_labels = []
print(files)
final_res=[]
for i, file in enumerate(files):
    try:
        match = re.match(pattern, file)
        N_init = int(match.group(1))
        maxiter = int(match.group(2))
        epsilon = float(match.group(3))
        initlengths.append(N_init)
        epsilons.append(epsilon)
        maxiters.append(maxiter)
        rothe = np.load(file)
        params = rothe["params"]
        if molecule == "LiH":
            num_Gaussians_finaltime = len(params[-1]) // 8
        elif molecule == "LiH2":
            num_Gaussians_finaltime = len(params[-1]) // 12
        finaltime_gaussians.append(num_Gaussians_finaltime)
        times_Rothe = rothe["times"]
        dipmom_Rothe = rothe["xvals"]
        errors_Rothe = rothe["rothe_errors"]
        lengths.append(len(times_Rothe))
        errors_rothe_list.append(errors_Rothe)
        times_rothe_list.append(times_Rothe)
        dipmom_rothe_list.append(dipmom_Rothe)
        omegaval, hhg_vals = compute_hhg_spectrum(times_Rothe, dipmom_Rothe)
        hhg_vals=hhg_vals*(omegaval/omega_ref)**2
        if times_Rothe[-1] > 300:
            omegavals.append(omegaval)
            hhg_Rothe_list.append(hhg_vals)
        else:
            omegavals.append(None)
            hhg_Rothe_list.append(None)
    except:
        print("Error reading file %s" % file)
        del files[i]
        continue
minlen = min(lengths)
print("Time: %.2f" % times_rothe_list[i][:minlen][-1])
 
for i, rothe_errors in enumerate(errors_rothe_list):
    print("RE, timesteps=%3d, number_gaussians=%d/%d, epsilon=%.3e: %f" % (maxiters[i], initlengths[i], finaltime_gaussians[i], epsilons[i], np.sum(errors_rothe_list[i][:minlen])))
    final_res.append(np.sum(errors_rothe_list[i]))
# Sort the file indices by finaltime_gaussians (i.e. number of Gaussians)
sorted_indices = sorted(range(len(finaltime_gaussians)), key=lambda i: final_res[i])[::-1]
#print(sorted_indices)
#sys.exit(0)
### Plot 1: Combined Dipole Moment and Its Derivative

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Dipole Moment vs. Time
if grid_solution_exists:
    axs[0].plot(times_grid, dipmom_grid, label="Grid", color="black")
for i in sorted_indices:
    axs[0].plot(
        times_rothe_list[i],
        dipmom_rothe_list[i],
        label=r"$N_{max}=%d$, $\varepsilon=%.3e$" % (finaltime_gaussians[i], epsilons[i])
    )
axs[0].set_title("Dipole Moment")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Dipole Moment")
axs[0].legend()

# Right subplot: Finite-Difference Derivative of the Dipole Moment
if grid_solution_exists:
    axs[1].plot(
        times_grid[1:],
        dipmom_grid[1:] - dipmom_grid[:-1],
        label="Grid",
        color="black"
    )
for i in sorted_indices:
    axs[1].plot(
        times_rothe_list[i][1:],
        dipmom_rothe_list[i][1:] - dipmom_rothe_list[i][:-1],
        label=r"$N_{max}=%d$, $\varepsilon=%.3e$" % (finaltime_gaussians[i], epsilons[i])
    )
axs[1].set_title("Dipole Moment Derivative")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Finite Difference Derivative")
axs[1].legend()

plt.tight_layout()
#plt.show()

### Plot 2: Errors
plt.figure()
for i in sorted_indices:
    plt.plot(
        times_rothe_list[i],
        errors_rothe_list[i],
        label=r"$N_{max}=%d$, $\varepsilon=%.3e$" % (finaltime_gaussians[i], epsilons[i])
    )
plt.legend()
plt.savefig("plots/errors_%s_%f.pdf" % (molecule, fieldstrengh))
#plt.show()

### Plot 3: Dipole Moment and HHG Spectrum (Stacked Vertically)

# Compute grid HHG spectrum if available.
if grid_solution_exists:
    grid_omega, grid_Px = compute_hhg_spectrum(times_grid, dipmom_grid)
    grid_Px=grid_Px*(grid_omega/omega_ref)**2
else:
    grid_omega, grid_Px = None, None

# Determine reference frequency scaling.
if molecule == "LiH" and abs(fieldstrengh - 0.0534) < 1e-5:
    moleculename = "LiH"
elif molecule == "LiH2" and abs(fieldstrengh - 0.0534) < 1e-5:
    moleculename = "(LiH)$_2$"
else:
    moleculename = molecule

# Use a vertical arrangement (2 rows, 1 column) with a size that fits into a two-column LaTeX document.
# A typical width for a two-column figure is around 3.4 inches.
fig = plt.figure(figsize=(1.2*3.4, 1.2*5))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 1, height_ratios=[0.6, 0.4])  
axs=[]
axs.append(fig.add_subplot(gs[0, 0]))  # Upper figure
axs.append(fig.add_subplot(gs[1:, 0]))  # Lower figurealpha=0.9
linewidth=1.5
alpha=0.9
# Top subplot: Dipole Moment vs. Time (re-plotted)
if grid_solution_exists:
    axs[0].plot(times_grid, dipmom_grid, label="Grid", color="black", linewidth=1.2*linewidth)
    
# Do not show legend for the top subplot.

# Bottom subplot: HHG Spectrum with cumulative Rothe error in the label.
if grid_solution_exists:
    axs[1].plot(grid_omega/omega_ref, grid_Px, label="Grid", color="black", linewidth=linewidth)
colors=["darkorange","seagreen","mediumorchid","red", "blue", "fuchsia"]
linestyle=(0,(3,1))

for j,i in enumerate(sorted_indices):
    if omegavals[i] is not None:
        cumulative_error = np.sum(errors_rothe_list[i])
        maxiter=maxiters[i]
        if maxiter==0:
            label="Frozen Gaussians"
        else:
            label=r"$M_{max.}=%d$, $r_{tot}=%.2f$" % (finaltime_gaussians[i], cumulative_error)
            label=r"$M_{max.}=%d$" % (finaltime_gaussians[i])
        line,=axs[1].plot(omegavals[i]/omega_ref, hhg_Rothe_list[i],
                    linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                    label=label,color=colors[j])
        line,=axs[0].plot(
        times_rothe_list[i],
        dipmom_rothe_list[i],
        linestyle=linestyle, linewidth=linewidth, alpha=alpha,
        label=label,color=colors[j])
axs[1].set_title(r"HHG Spectrum")
if abs(fieldstrengh - 0.0534) < 1e-5:
    xlim_min=1
    xlim_max=60
else:
    xlim_min=1
    xlim_max=80
axs[1].set_xlim(xlim_min,xlim_max)
ylim_max=6*max(grid_Px[len(grid_Px)//2:][3:])
ylim_min=0.5*min(grid_Px[len(grid_Px)//2:][3:3*xlim_max])
axs[1].set_ylim(ylim_min,ylim_max)
axs[1].set_yscale("log")
axs[1].set_xlabel("Harmonic order")
axs[1].set_ylabel("Intensity")
axs[1].legend(loc='upper right', fontsize=9, framealpha=0.5,labelspacing=0.3,borderpad=0.2,handletextpad=0.5)

axs[0].set_title(r"%s, %s, $E_0=%.4f$ a.u." % (method,moleculename, fieldstrength_int*0.0534))
axs[0].set_xlabel("Time (a.u.)")
axs[0].set_ylabel("Dipole Moment (a.u.)")
axs[0].set_xlim(-2, 312)
# Show the legend only in the lower subplot.

plt.tight_layout(pad=0.3)
plt.savefig("plots/HHG_%s_%.4f_%s.pdf" % (molecule, fieldstrengh,method))
plt.show()
