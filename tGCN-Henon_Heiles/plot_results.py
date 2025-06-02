# plot_results.py
"""Aggregate & visualise all per-timestep MSE traces produced by *create_results.py*.

Just run:
    python plot_results.py

The script discovers files matching ``MSE_T=*`` and also plots the ``MSE_idiot``
baseline if present.  Output goes to ``all_mse_comparison.png``.
"""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

root = Path('.')
model_files = sorted(root.glob('MSE_T=*_use_gcn=*.npz'))
id_file = root / 'MSE_idiot.npz'
if not model_files:
    raise SystemExit('No model MSE files found. Run create_results.py first.')

density = 1
ranges = [
    (0, 1000, "Training (0–1000)"),
    (1000, 2000, "Test + Validation (1000–2000)"),
    (2000, 3000, "Validation (2000–3000)"),
]
fig, axs = plt.subplots(3, 1, figsize=(10, 9))  # No sharey!

for mf in model_files:
    data = np.load(mf)
    mse = data['mse']
    m = re.match(r'MSE_T=(\d+)_use_gcn=(True|False)\.npz', mf.name)
    T = int(m.group(1))
    xs = np.arange(T, len(mse) + T, density)
    ys = mse[::density]
    label = f"T={m.group(1)}, GCN={m.group(2)[0]}" if m else mf.stem
    for ax, (lo, hi, _) in zip(axs, ranges):
        mask = (xs >= lo) & (xs < hi)
        ax.plot(xs[mask], ys[mask], label=label)

if id_file.exists():
    idiot = np.load(id_file)['mse_idiot']
    xs = np.arange(1, len(idiot)+1, density)
    ys = idiot[::density]
    for ax, (lo, hi, _) in zip(axs, ranges):
        mask = (xs >= lo) & (xs < hi)
        ax.plot(xs[mask], ys[mask], label='\Delta_{}', linestyle='--', linewidth=1.2, alpha=0.5)

for i, ax in enumerate(axs):
    lo, hi, title = ranges[i]
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.01,0.1)
    ax.set_title(title)
    if i == 1:
        ax.axvline(1200, color='k', linestyle=':', linewidth=1)
        ax.text(1200 + 10, ax.get_ylim()[1]*0.95, 'end of training', va='top', ha='left', color='k', fontsize=10)
    if i == 2:
        ax.set_xlabel('Time index (t–T)')
    if i == 1:
        ax.set_ylabel('MSE')
    if i == 0:
        ax.legend(ncol=2)

fig.suptitle('Per-timestep MSE comparison', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('.png', dpi=300)
plt.show()
