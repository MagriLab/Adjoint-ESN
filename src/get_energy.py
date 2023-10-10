import os
import sys  # add the root directory to the path before importing from the library

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.enums import eParam

data_dir = Path("data")
transient_time = 200
loop_time = 50
sim_time = transient_time + loop_time
sim_dt = 1e-3
network_dt = 1e-2

N_g = 4


def acoustic_energy(y, N_g):
    return 1 / 4 * np.mean(np.sum(y[:, : 2 * N_g] ** 2, axis=1))


def get_J(p):
    p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}

    regime_str = f'beta = {p_sim["beta"]}, tau = {p_sim["tau"]}'
    print("Regime:", regime_str)

    y_sim, t_sim = pp.load_data(
        beta=p_sim["beta"],
        tau=p_sim["tau"],
        x_f=0.2,
        N_g=N_g,
        sim_time=sim_time,
        sim_dt=sim_dt,
        data_dir=data_dir,
    )

    y, t = pp.discard_transient(y_sim, t_sim, transient_time)

    J = acoustic_energy(y, N_g)

    # check how the energy changes with upsampling
    y_up, t_up = pp.upsample(y, t, network_dt)
    J_up = acoustic_energy(y, N_g)
    return J, J_up


# Compute the energy and gradient on a grid
beta_list = np.arange(0.5, 5.6, 0.1)
tau_list = np.arange(0.05, 0.37, 0.02)
p_list = pp.make_param_mesh([beta_list, tau_list])

J = np.zeros(len(p_list))
J_up = np.zeros(len(p_list))
for p_idx, p in enumerate(p_list):
    J[p_idx], J_up[p_idx] = get_J(p)

J = np.reshape(J, (len(beta_list), len(tau_list)))
J_up = np.reshape(J_up, (len(beta_list), len(tau_list)))

energy_dict = {"J": J, "J_up": J_up, "beta_list": beta_list, "tau_list": tau_list}

pp.pickle_file(Path.cwd() / "energy.pickle", energy_dict)
