import sys
import numpy as np
import matplotlib.pyplot as plt
import json

T = 20
wM = 1
label_size = 13   # axis labels
tick_size = 10    # tick labels
title_size = 15  # subplot titles
legend_size = 11  # legend font size

configuration = "best_config_T=%d_wM=%.2f.json" % (T, wM)
with open(configuration, "r") as f:
    data = json.load(f)
print(data)

infile = "training_trajectory_T=%d_wM=%.2f.npz" % (T, wM)
data = np.load(infile)
epochs = data["epochs"]
train_loss = data["train_loss"]

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss)
plt.xlabel("Epochs", fontsize=label_size)
plt.ylabel("Training Loss", fontsize=label_size)
plt.ylim(0, 1.4)

# Hardcoded hyperparameter info
info_text = (
    "$n_L$: 32\n"
    "$n_h$: 256\n"
    "$n_{hl}$: 3\n"
    "$\\lambda$: 0.00154\n"
    "$N_b$: 32"
)

plt.text(0.98, 0.93, info_text,
         transform=plt.gca().transAxes,
         ha='right', va='top',
         fontsize=11,
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig("plots/training_trajectory_T=%d_wM=%.2f.pdf" % (T, wM), dpi=300)
plt.show()
