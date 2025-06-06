
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
root = Path('.')
use_GCN= sys.argv[1].lower() == 'true'
model_files = sorted(root.glob('MSE_T=*_use_gcn=%s.npz'% use_GCN))
id_file = root / 'MSE_idiot.npz'
nochange_file = root / 'MSE_nochange.npz'
if not model_files:
    raise SystemExit('No model MSE files found. Run create_results.py first.')

density = 1
ranges = [
    (0, 1000, "Training (0–1000)"),
    (1000, 2000, "Test + Validation (1000–2000)"),
]
fig, axs = plt.subplots(len(ranges), 1, figsize=(10, 4))  

for mf in model_files:
    data = np.load(mf)
    mse = data['mse']
    m = re.match(r'MSE_T=(\d+)_use_gcn=(True|False)\.npz', mf.name)
    T = int(m.group(1))
    xs = np.arange(T, len(mse) + T, density)
    ys = mse[::density]
    if use_GCN:
        label= f"T={m.group(1)}, tGCN"
    else:
        label= f"T={m.group(1)}, LSTM"
    for ax, (lo, hi, _) in zip(axs, ranges):
        mask = (xs >= lo) & (xs < hi)
        ax.plot(xs[mask], ys[mask], label=label)

if id_file.exists():
    idiot = np.load(id_file)['mse_idiot']
    xs = np.arange(1, len(idiot)+1, density)
    ys = idiot[::density]
    for ax, (lo, hi, _) in zip(axs, ranges):
        mask = (xs >= lo) & (xs < hi)
        ax.plot(xs[mask], ys[mask], label=r'$\Delta_{t-1}$', linestyle='--', linewidth=1.2, alpha=0.5)
for i, ax in enumerate(axs):
    lo, hi, title = ranges[i]
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.005,0.1)
    ax.set_title(title)
    if i == 1:
        ax.axvline(1200, color='k', linestyle=':', linewidth=2)
    if i == 1:
        ax.set_xlabel(r'Time index $i$ corresponding to a time point $t_i$')
    #if i == 1:
    ax.set_ylabel('MSE (scaled)')
    if i == 0:
        ax.legend(ncol=2)

type="t-GCN" if use_GCN else "LSTM"
fig.suptitle('Per-timestep MSE comparison for %s'%type, fontsize=14)
fig.tight_layout(rect=[-0.01,-0.02, 1.01, 1.05])
plt.savefig('tGCN_GCN=%s.pdf'%type, dpi=300)
plt.savefig('tGCN_GCN=%s.png'%type, dpi=300)

plt.show()
