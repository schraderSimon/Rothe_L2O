import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse

# Use argparse to allow passing parameter filters as key=value pairs.
parser = argparse.ArgumentParser(description="Plot optimizer results with optional filtering by wf or T.")
parser.add_argument("params", nargs="*", help="Optional parameters in the form key=value (e.g., wf=1.00 or T=10).")
args = parser.parse_args()

# Default filters (wildcards)
wf_filter = "*"
T_filter = "*"

# Process the provided key=value parameters.
for param in args.params:
    if "=" in param:
        key, value = param.split("=")
        if key.lower() == "wf":
            try:
                wf_val = float(value)
                wf_filter = f"{wf_val:.2f}"
            except ValueError:
                raise ValueError("Invalid value for wf. It should be a float, e.g., wf=1.00")
        elif key.lower() == "t":
            try:
                # T is expected to be an integer.
                T_val = int(value)
                T_filter = f"{T_val}"
            except ValueError:
                raise ValueError("Invalid value for T. It should be an integer, e.g., T=10")
        else:
            print(f"Warning: Unrecognized parameter '{key}' will be ignored.")
    else:
        print(f"Warning: Parameter '{param}' is not in key=value format and will be ignored.")

# Set font sizes for a two-column document layout.
label_size = 13   # axis labels
tick_size = 10    # tick labels
title_size = 15   # subplot titles
legend_size = 11  # legend font size

# Create figure and two vertically stacked subplots.
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), sharex=True)

def plot_optimizer_data(ax, method, wf_filter, T_filter):
    # Load reference data for the given method (training/testing).
    df_quad = pd.read_csv(f"reference_{method}.csv")
    iters_quad = df_quad["iteration"]

    # Plot Adam results (using default cycle colors).
    adam_lrs = [1e-2, 1e-1, 1e0, 1e1]
    for lr in adam_lrs:
        ax.plot(iters_quad, df_quad[f"Adam(lr={lr})_q50"], label=rf"Adam $\lambda={lr}$")
        ax.fill_between(
            iters_quad,
            df_quad[f"Adam(lr={lr})_q25"],
            df_quad[f"Adam(lr={lr})_q75"],
            alpha=0.2
        )

    # Plot BFGS results.
    ax.plot(iters_quad, df_quad["BFGS_q50"], label="BFGS", color="black")
    ax.fill_between(iters_quad, df_quad["BFGS_q25"], df_quad["BFGS_q75"],
                    alpha=0.2, color="gray")

    # Build the file pattern based on provided filters.
    pattern = f"SSM_L2O_{method}_T={T_filter}_wM={wf_filter}.csv"
    for file in sorted(glob.glob(pattern)):
        df_l2o = pd.read_csv(file)
        iters_l2o = df_l2o["iteration"]

        # Extract T and wf from the filename for the label.
        filename = os.path.basename(file)
        parts = filename.split("_")
        T_str = parts[3].split("=")[1]
        wf_str = parts[4].split("=")[1].replace(".csv", "")
        label = f"n-SSM T={T_str} k={wf_str}"
        ax.plot(iters_l2o, df_l2o["L2O_q50"], label=label)
        ax.fill_between(
            iters_l2o,
            df_l2o["L2O_q25"],
            df_l2o["L2O_q75"],
            alpha=0.2
        )

    ax.set_yscale("log")
    ax.set_ylabel("Relative error f(x)/f(x0)", fontsize=label_size)
    # Set the title normally (without using pad)
    ax.set_title(f"{method.capitalize()}", fontsize=title_size)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.set_ylim(1e-5, 5)
    ax.set_xlim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

# Plot training data on the top subplot and testing data on the bottom.
plot_optimizer_data(axs[0], "training", wf_filter, T_filter)
plot_optimizer_data(axs[1], "testing", wf_filter, T_filter)
axs[1].set_xlabel("Iteration", fontsize=label_size)
if T_filter == "*":
    T_filter = "all"
if wf_filter == "*":
    wf_filter = "all"

# Create the legend and capture its handle.
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
leg = fig.legend(
    by_label.values(), by_label.keys(),
    loc='upper center', ncol=3, fontsize=legend_size,
    bbox_to_anchor=(0.5, 1)
)

# Instead of tight_layout, adjust the subplots manually.
# Increase the top margin to make room for the legend.
fig.subplots_adjust(top=0.85, bottom=0.08, left=0.10, right=0.95)

# Save the figure and include the legend in the bounding box.
plt.savefig("plots/combined_results_T=%s_wM=%s.png" % (T_filter, wf_filter),
            dpi=300, bbox_inches='tight', bbox_extra_artists=[leg])
plt.savefig("plots/combined_results_T=%s_wM=%s.pdf" % (T_filter, wf_filter),
            dpi=300, bbox_inches='tight', bbox_extra_artists=[leg])
plt.show()

print("Combined plot saved as combined_results.png")