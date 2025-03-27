import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
# Load CSV data
T=int(sys.argv[1])
df_quad = pd.read_csv("quad_experiment_results_quartiles.csv")
df_l2o = pd.read_csv("l2o_experiment_results_quartiles_T=%d.csv"%T)
iters_quad = df_quad["iteration"]
iters_l2o = df_l2o["iteration"]

plt.figure(figsize=(10, 6))

# Plot Adam results
adam_lrs = [1e-2, 1e-1, 1e0, 1e1]
for lr in adam_lrs:
    plt.plot(iters_quad, df_quad[f"Adam(lr={lr})_q50"], label=f"Adam LR={lr}")
    plt.fill_between(
        iters_quad,
        df_quad[f"Adam(lr={lr})_q25"],
        df_quad[f"Adam(lr={lr})_q75"],
        alpha=0.2,
    )

# Plot BFGS results
plt.plot(iters_quad, df_quad["BFGS_q50"], label="BFGS", color="black")
plt.fill_between(iters_quad, df_quad["BFGS_q25"], df_quad["BFGS_q75"], alpha=0.2, color="gray")

# Plot L2O results
plt.plot(iters_l2o, df_l2o["L2O_q50"], label="L2O", color="red")
plt.fill_between(iters_l2o, df_l2o["L2O_q25"], df_l2o["L2O_q75"], alpha=0.2, color="red")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Relative error f(x)/f(x0)")
plt.title("Comparison of Optimizers on Random Quadratic Problems")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.ylim(1e-12, 5)
plt.tight_layout()
plt.savefig("combined_results.png", dpi=200)
plt.savefig("combined_results.pdf")

plt.show()

print("Combined plot saved as combined_results.png")
