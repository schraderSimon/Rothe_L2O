import sys
import numpy as np
import matplotlib.pyplot as plt

T=int(sys.argv[1])
wM=float(sys.argv[2])

infile="training_trajectory_T=%d_wM=%.2f.npz"%(T,wM)
data = np.load(infile)
epochs= data["epochs"]
train_loss= data["train_loss"]
plt.plot(epochs,train_loss)
plt.ylim(0,1.5)

plt.show()