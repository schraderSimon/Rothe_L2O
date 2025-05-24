import os
import numpy as np
import matplotlib.pyplot as plt

file_exists = os.path.isfile

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
L2O_MAIN = "results_L2O/narrow_data.npz"
L2O_VAL2 = "results_L2O/narrow_data2.npz"
BFGS_MAIN = "results_L2O/BFGS_narrow.npz"
BFGS_VAL2 = "results_L2O/BFGS_narrow2.npz"

l2o = np.load(L2O_MAIN)
train_l2o, test_l2o, val_l2o = l2o["train"], l2o["test"], l2o["val"]
val2_l2o = np.load(L2O_VAL2)["val2"] if file_exists(L2O_VAL2) else None

bfgs_train = bfgs_test = bfgs_val = bfgs_val2 = None
if file_exists(BFGS_MAIN):
    bfgs = np.load(BFGS_MAIN)
    bfgs_train, bfgs_test, bfgs_val = bfgs["train"], bfgs["test"], bfgs["val"]
if file_exists(BFGS_VAL2):
    bfgs_val2 = np.load(BFGS_VAL2)["val2"]

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def stats(a):
    """Return median, 25th and 75th percentiles along axis 0."""
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

# -----------------------------------------------------------------------------
# Training curve (row 0)
# -----------------------------------------------------------------------------
ax_train = axs[0]

m_train, q1_train, q3_train = stats(train_l2o)
ax_train.plot(x, m_train, lw=1.0, label="L2O (q=2)")
ax_train.fill_between(x, q1_train, q3_train, alpha=.2)

if bfgs_train is not None:
    m_b_train, q1_b_train, q3_b_train = stats(bfgs_train)
    ax_train.plot(x, m_b_train,  lw=1.0, label="BFGS (q=2)")
    ax_train.fill_between(x, q1_b_train, q3_b_train, alpha=.15)

ax_train.set_title("Training", fontsize=title_size)
ax_train.set_ylabel(r"Relative error f(x)/f($x_0$)", fontsize=label_size)
ax_train.set_yscale("log")
ax_train.grid(True, which="major", linestyle="--", linewidth=0.5)
ax_train.set_ylim(1e-3, 10)
ax_train.set_xlim(0, 100)
ax_train.tick_params(axis="both", which="major", labelsize=tick_size)

# -----------------------------------------------------------------------------
# Testing curve (row 1)
# -----------------------------------------------------------------------------
ax_test = axs[1]

m_test, q1_test, q3_test = stats(test_l2o)
ax_test.plot(x, m_test, lw=1.0, label="L2O (q=2)")
ax_test.fill_between(x, q1_test, q3_test, alpha=.2)

if bfgs_test is not None:
    m_b_test, q1_b_test, q3_b_test = stats(bfgs_test)
    ax_test.plot(x, m_b_test, lw=1.0, label="BFGS (q=2)")
    ax_test.fill_between(x, q1_b_test, q3_b_test, alpha=.15)

ax_test.set_title("Testing", fontsize=title_size)
ax_test.set_ylabel(r"Relative error f(x)/f($x_0$)", fontsize=label_size)
ax_test.set_yscale("log")
ax_test.grid(True, which="major", linestyle="--", linewidth=0.5)
ax_test.set_ylim(1e-3, 10)
ax_test.set_xlim(0, 100)
ax_test.tick_params(axis="both", which="major", labelsize=tick_size)

# -----------------------------------------------------------------------------
# Validation curve (row 2)
# -----------------------------------------------------------------------------
ax_val = axs[2]

m_val, q1_val, q3_val = stats(val_l2o)
ax_val.plot(x, m_val, lw=1.0, label="L2O (q=2)")
ax_val.fill_between(x, q1_val, q3_val, alpha=.2)

if bfgs_val is not None:
    m_b_val, q1_b_val, q3_b_val = stats(bfgs_val)
    ax_val.plot(x, m_b_val, lw=1.0, label="BFGS (q=2)")
    ax_val.fill_between(x, q1_b_val, q3_b_val, alpha=.15)

# Extra quality=3 curves for validation
if val2_l2o is not None:
    m_val2, q1_val2, q3_val2 = stats(val2_l2o)
    ax_val.plot(x, m_val2, lw=1.0, label="L2O (q=3)")
    ax_val.fill_between(x, q1_val2, q3_val2, alpha=.1)

if bfgs_val2 is not None:
    m_b_val2, q1_b_val2, q3_b_val2 = stats(bfgs_val2)
    ax_val.plot(x, m_b_val2, lw=1.0, label="BFGS (q=3)")
    ax_val.fill_between(x, q1_b_val2, q3_b_val2, alpha=.1)

ax_val.set_title("Validation", fontsize=title_size)
ax_val.set_ylabel(r"Relative error f(x)/f($x_0$)", fontsize=label_size)
ax_val.set_yscale("log")
ax_val.grid(True, which="major", linestyle="--", linewidth=0.5)
ax_val.set_ylim(5e-3, 5)
ax_val.set_xlim(0, 100)
ax_val.tick_params(axis="both", which="major", labelsize=tick_size)

# -----------------------------------------------------------------------------
# Common X label
# -----------------------------------------------------------------------------
axs[-1].set_xlabel("Iteration", fontsize=label_size)

# -----------------------------------------------------------------------------
# Legend – unique entries, top centre (no loops)
# -----------------------------------------------------------------------------
handles_train, labels_train = ax_train.get_legend_handles_labels()
handles_test,  labels_test  = ax_test.get_legend_handles_labels()
handles_val,   labels_val   = ax_val.get_legend_handles_labels()

handles = handles_train + handles_test + handles_val
labels  = labels_train  + labels_test  + labels_val

by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc="upper center",
           ncol=3,
           fontsize=legend_size,
           bbox_to_anchor=(0.5, 1))

fig.subplots_adjust(top=0.9, bottom=0.05, left=0.12, right=0.98, hspace=0.15)
plt.savefig("results_L2O/plot_narrow.pdf", dpi=300, bbox_inches="tight")
plt.show()
