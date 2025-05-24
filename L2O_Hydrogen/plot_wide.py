import os
import numpy as np
import matplotlib.pyplot as plt

file_exists = lambda p: os.path.isfile(p)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
L2O_MAIN = "results_L2O/wide_data.npz"
BFGS_MAIN = "results_L2O/BFGS_widew.npz"

l2o = np.load(L2O_MAIN)
train_l2o, test_l2o = l2o["train"], l2o["test"]

bfgs_train = bfgs_test = bfgs_val = bfgs_val2 = None
if file_exists(BFGS_MAIN):
    bfgs = np.load(BFGS_MAIN)
    bfgs_train, bfgs_test, bfgs_val = bfgs["train"], bfgs["test"], bfgs["val"]

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def stats(a):
    return np.median(a, 0), np.percentile(a, 25, 0), np.percentile(a, 75, 0)

# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
label_size = 13
tick_size = 10
title_size = 15
legend_size = 11

x = np.arange(train_l2o.shape[1])
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 9), sharex=True, sharey=True)

cases = [
    (train_l2o, bfgs_train, "train"),
    (test_l2o,  bfgs_test,  "test"),
]

# -----------------------------------------------------------------------------
# Core plotting loop
# -----------------------------------------------------------------------------
for ax, (l2o_arr, bfgs_arr, title) in zip(axs, cases):
    m, q1, q3 = stats(l2o_arr)
    ax.plot(x, m, lw=1.0, label=f"L2O mid quality")
    ax.fill_between(x, q1, q3, alpha=.2)

    if bfgs_arr is not None:
        mb, q1b, q3b = stats(bfgs_arr)
        ax.plot(x, mb, "--", lw=1.0, label="BFGS mid quality")
        ax.fill_between(x, q1b, q3b, alpha=.15)

    ax.set_title(title, fontsize=title_size)
    ax.set_ylabel(r"Relative error f(x)/f($x_0$)", fontsize=label_size)
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.set_ylim(1e-3, 10)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

# Validation extra quality=3 curves
ax_val = axs[2]

axs[-1].set_xlabel("step", fontsize=label_size)

# -----------------------------------------------------------------------------
# Legend – unique entries, top centre
# -----------------------------------------------------------------------------
handles, labels = axs[0].get_legend_handles_labels()
for ax in axs[1:]:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, fontsize=legend_size, bbox_to_anchor=(0.5, 1))

fig.subplots_adjust(top=0.92, bottom=0.07, left=0.15, right=0.95, hspace=0.15)
plt.savefig("results_L2O/plot_wide.pdf", dpi=300, bbox_inches="tight")
plt.show()
