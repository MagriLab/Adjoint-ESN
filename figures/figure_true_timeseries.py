import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

import adjoint_esn.utils.postprocessing as post
import adjoint_esn.utils.visualizations as vis
from adjoint_esn.rijke_galerkin import sensitivity as sens
from adjoint_esn.rijke_galerkin.solver import Rijke
from adjoint_esn.utils import custom_colormap as cm
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam, get_eVar

plt.style.use("src/stylesheet2.mplstyle")
cmap = cm.create_custom_colormap(type="discrete")

figure_size = (15, 10)

save_fig = False

beta_list = [8.0, 8.5, 8.7, 9.0]
tau = 0.22
N_g = 4
plot_time = 40
transient_time = 200
sim_time = transient_time + plot_time
sim_dt = 1e-3

eInputVar = get_eVar("eta_mu_v", N_g)
eOutputVar = get_eVar("eta_mu_v", N_g)

plt_idx = [eOutputVar.mu_1]
plt_idx_pairs = [[eOutputVar.mu_1, eOutputVar.mu_2], [eOutputVar.mu_3, eOutputVar.mu_4]]

linespecs = {"color": [cmap(0)], "linewidth": [2.5]}

integrator = "odeint"
data_dir = Path("data")

for beta in beta_list:
    beta_name = f"{beta:.2f}"
    beta_name = beta_name.replace(".", "_")
    tau_name = f"{tau:.2f}"
    tau_name = tau_name.replace(".", "_")
    p_sim = {"beta": beta, "tau": tau}

    y_sim, t_sim = pp.load_data(
        beta=p_sim["beta"],
        tau=p_sim["tau"],
        x_f=0.2,
        N_g=N_g,
        sim_time=sim_time,
        sim_dt=sim_dt,
        data_dir=data_dir,
        integrator=integrator,
    )
    transient_steps = pp.get_steps(transient_time, sim_dt)
    t_sim = t_sim[transient_steps:]
    y_sim = y_sim[transient_steps:]
    x = np.linspace(0, 1.0, 100)
    [tt, xx] = np.meshgrid(t_sim, x)
    velocity_sim = Rijke.toPressure(
        N_g, y_sim[:, eOutputVar.eta_1 : eOutputVar.eta_1 + N_g], x
    )
    pressure_sim = Rijke.toPressure(
        N_g, y_sim[:, eOutputVar.mu_1 : eOutputVar.mu_1 + N_g], x
    )

    fig = plt.figure(figsize=figure_size, constrained_layout=True)

    fig.add_subplot(2, 1, 1)
    vis.plot_lines(
        t_sim,
        y_sim[:, plt_idx[0]],
        #    xlabel=f"$t$",
        ylabel=f"$\{plt_idx[0].name}$",
        **linespecs,
    )
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_title(f"$\\beta = {beta}, \; \\tau = {tau}$")
    ax.set_xlim([t_sim[0], t_sim[-1]])

    ax1 = fig.add_subplot(2, 1, 2)
    c1 = ax1.pcolormesh(tt, xx, pressure_sim.T, cmap="coolwarm", shading="auto")
    cb = fig.colorbar(c1, pad=0.01)
    # ax1.set_xlabel('$t$')
    ax1.set_ylabel("$x$")
    cb.ax.set_title("$p$")
    # ax1.set_xticklabels([])

    # fig.add_subplot(4,1,3)
    # vis.plot_lines(t_sim, y_sim[:,plt_idx[1]],
    #             #    xlabel=f"$t$",
    #             ylabel=f"$\{plt_idx[1].name}$",
    #             **linespecs)
    # ax = plt.gca()
    # ax.set_xticklabels([])

    # ax1 = fig.add_subplot(4,1,4)
    # c1 = ax1.pcolormesh(tt, xx, velocity_sim.T, cmap="coolwarm", shading="auto")
    # cb = fig.colorbar(c1, pad=0.005)
    ax1.set_xlabel("$t$")
    # ax1.set_ylabel('$x$')
    # cb.ax.set_title('$u$')

    # if save_fig:
    #     plt.savefig(f'paper/graphics_ppt/true_timeseries_beta_{beta_name}_tau_{tau_name}.png', dpi=300)
    # else:
plt.show()
