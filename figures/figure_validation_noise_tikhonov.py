import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp

plt.style.use("src/stylesheet.mplstyle")
cmap = plt.cm.get_cmap("GnBu")
# cmap = cm.create_custom_colormap('aqua',type='discrete')

save_fig = True
fig_name = "noisy_validation_tikhonov"
# N_reservoir = 1200, connectivity = 20
model_paths = [
    Path("local_results/rijke/run_20231029_153121_noise_2_new"),
    Path("local_results/rijke/run_20231029_153121_noise_5_new"),
    Path("local_results/rijke/run_20231029_153121_noise_10_new"),
]
save_paths = ["20240712_152249", "20240710_230419", "20240711_213911"]
n_noise_levels = 3
n_tikhs = 15
n_noise_realisations = 20

err_mat = np.empty((n_noise_levels, n_tikhs, n_noise_realisations))

fig = plt.figure(constrained_layout=True, figsize=(6, 4))
noise_levels = np.zeros(n_noise_levels)
for err_idx, err_path in enumerate(model_paths):
    err_dict = pp.unpickle_file(
        err_path / f"error_val_results_{save_paths[err_idx]}.pickle"
    )[0]
    err_val = err_dict["err_val"]
    tikhs = err_dict["tikhs"]
    err_mat[err_idx, :, :] = err_val.T
    noise_levels[err_idx] = err_dict["noise_level"]
colors = cmap(np.linspace(0.4, 0.8, n_noise_levels))
for err_idx in range(n_noise_levels):
    mean_err = np.mean(err_mat[err_idx, :, :], axis=1)
    std_err = np.std(err_mat[err_idx, :, :], axis=1)
    plt.errorbar(tikhs, 100 * mean_err, 100 * std_err, color=colors[err_idx], fmt="-o")
plt.xticks(tikhs[0::2])
ax = plt.gca()
ax.set_xticklabels([f"$10^{{{int(tikh)}}}$" for tikh in tikhs[0::2]])
plt.xlabel("$\lambda$")
plt.ylabel("Validation error $\%$")
plt.legend([f"${noise_level}\%$" for noise_level in noise_levels], ncols=3)

if save_fig:
    fig.savefig(f"paper/graphics/figure_{fig_name}.png", bbox_inches="tight")
    fig.savefig(f"paper/graphics/figure_{fig_name}.pdf", bbox_inches="tight")
plt.show()
