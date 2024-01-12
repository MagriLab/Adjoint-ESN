import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from functools import partial

import numpy as np

from adjoint_esn.rijke_galerkin.solver import Rijke
from adjoint_esn.utils import solve_ode
from adjoint_esn.utils.discretizations import finite_differences
from adjoint_esn.utils.enums import eParam


def acoustic_energy(y, N_g):
    return 1 / 4 * np.mean(np.sum(y[:, : 2 * N_g] ** 2, axis=1))


def acoustic_energy_inst(y, N_g):
    return 1 / 4 * (np.sum(y[:, : 2 * N_g] ** 2, axis=1))


def true_direct_sensitivity(my_rijke, t_bar, y_bar, integrator="odeint"):
    dt = t_bar[1] - t_bar[0]
    # direct problem
    dir0 = np.zeros(2 * my_rijke.N_dim + 2)
    dir = solve_ode.integrate(
        my_rijke.direct_ode,
        dir0,
        t_bar,
        integrator=integrator,
        args=(t_bar, 1 / dt, y_bar),
    )
    dJdp = 1 / t_bar[-1] * dir[-1, -2:]
    return dJdp


def true_adjoint_sensitivity(my_rijke, t_bar, y_bar, integrator="odeint"):
    dt = t_bar[1] - t_bar[0]
    # adjoint problem
    adjT = np.zeros(my_rijke.N_dim + 2)
    adj = solve_ode.integrate(
        my_rijke.adjoint_ode,
        adjT,
        np.flip(t_bar),
        integrator=integrator,
        args=(t_bar, 1 / dt, y_bar),
    )
    dJdp = 1 / t_bar[-1] * adj[-1, -2:]
    return dJdp


def true_finite_difference_sensitivity(
    my_rijke, t_bar, y_bar, h, h_tau, method, integrator="odeint"
):
    # Calculate numerically
    # Find perturbed solutions (in beta)
    dJdp = np.zeros((2,))

    # left solution with beta = beta-h
    my_rijke_beta_left = Rijke(
        N_g=my_rijke.N_g,
        N_c=my_rijke.N_c,
        c_1=0.1,
        c_2=0.06,
        beta=my_rijke.beta - h,
        x_f=my_rijke.x_f,
        tau=my_rijke.tau,
        heat_law="kings_poly",
        damping="modal",
    )

    y_bar_beta_left = solve_ode.integrate(
        my_rijke_beta_left.ode, y_bar[0, : my_rijke.N_dim], t_bar, integrator=integrator
    )
    J_beta_left = acoustic_energy(y_bar_beta_left[1:, :], my_rijke.N_g)

    # right solution with beta = beta+h
    my_rijke_beta_right = Rijke(
        N_g=my_rijke.N_g,
        N_c=my_rijke.N_c,
        c_1=0.1,
        c_2=0.06,
        beta=my_rijke.beta + h,
        x_f=my_rijke.x_f,
        tau=my_rijke.tau,
        heat_law="kings_poly",
        damping="modal",
    )

    y_bar_beta_right = solve_ode.integrate(
        my_rijke_beta_right.ode,
        y_bar[0, : my_rijke.N_dim],
        t_bar,
        integrator=integrator,
    )
    J_beta_right = acoustic_energy(y_bar_beta_right[1:, :], my_rijke.N_g)

    # left solution with tau = tau-h
    my_rijke_tau_left = Rijke(
        N_g=my_rijke.N_g,
        N_c=my_rijke.N_c,
        c_1=0.1,
        c_2=0.06,
        beta=my_rijke.beta,
        x_f=my_rijke.x_f,
        tau=my_rijke.tau - h_tau,
        heat_law="kings_poly",
        damping="modal",
    )

    y_bar_tau_left = solve_ode.integrate(
        my_rijke_tau_left.ode, y_bar[0, : my_rijke.N_dim], t_bar, integrator=integrator
    )
    J_tau_left = acoustic_energy(y_bar_tau_left[1:, :], my_rijke.N_g)

    # # right solution with tau = tau+h
    my_rijke_tau_right = Rijke(
        N_g=my_rijke.N_g,
        N_c=my_rijke.N_c,
        c_1=0.1,
        c_2=0.06,
        beta=my_rijke.beta,
        x_f=my_rijke.x_f,
        tau=my_rijke.tau + h_tau,
        heat_law="kings_poly",
        damping="modal",
    )

    y_bar_tau_right = solve_ode.integrate(
        my_rijke_tau_right.ode, y_bar[0, : my_rijke.N_dim], t_bar, integrator=integrator
    )
    J_tau_right = acoustic_energy(y_bar_tau_right[1:, :], my_rijke.N_g)

    # define which finite difference method to use
    finite_difference = partial(finite_differences, method=method)

    J = acoustic_energy(y_bar[1:, :], my_rijke.N_g)
    dJdp[eParam.beta] = finite_difference(J, J_beta_right, J_beta_left, h)
    dJdp[eParam.tau] = finite_difference(J, J_tau_right, J_tau_left, h_tau)
    return dJdp