import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from adjoint_esn.utils import lyapunov as lyap
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63


def objective_fun(u):
    return np.mean(u[:, 2])


def Lyapunov_Time(sys, params, transient_time, sim_dt, integrator):
    sim_time = 1000 + transient_time
    my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
        sys, params, sim_time, sim_dt, y_init=[-2.4, -3.7, 14.98], integrator=integrator
    )
    y, t = pp.discard_transient(y_sim, t_sim, transient_time)
    LEs, _, _, _ = lyap.calculate_LEs(
        sys=my_sys,
        sys_type="continuous",
        X=y,
        t=t,
        transient_time=transient_time,
        dt=sim_dt,
        target_dim=None,
        norm_step=1,
    )
    LEs_target = LEs[-1]
    LT = 1 / LEs_target[0]
    if LT <= 0:
        return 1
    else:
        return LT


def main(args):
    if len(args.beta) == 3:
        beta_list = np.arange(args.beta[0], args.beta[1], args.beta[2])
    elif len(args.beta) == 1:
        beta_list = [args.beta[0]]

    if len(args.rho) == 3:
        rho_list = np.arange(args.rho[0], args.rho[1], args.rho[2])
    elif len(args.rho) == 1:
        rho_list = [args.rho[0]]

    if len(args.sigma) == 3:
        sigma_list = np.arange(args.sigma[0], args.sigma[1], args.sigma[2])
    elif len(args.sigma) == 1:
        sigma_list = [args.sigma[0]]

    # Create the mesh to get the data from
    eParam = Lorenz63.get_eParamVar()
    param_mesh_input = [None] * 3
    param_mesh_input[eParam.beta] = beta_list
    param_mesh_input[eParam.rho] = rho_list
    param_mesh_input[eParam.sigma] = sigma_list
    p_list = pp.make_param_mesh(param_mesh_input)

    my_eParam = eParam[args.param]
    transient_time = 20
    dJdp = np.zeros((len(p_list)))
    for p_idx, p in enumerate(p_list):
        p_sim = {"beta": p[eParam.beta], "rho": p[eParam.rho], "sigma": p[eParam.sigma]}

        params = {}
        for param in eParam:
            params[param.name] = p[param]
        regime_str = " ".join([f"{param.name} = {p[param]:0.3f}" for param in eParam])
        print(regime_str)

        # determine Lyapunov time and choose simulation time accordingly
        LT = Lyapunov_Time(
            Lorenz63, p_sim, transient_time, args.sim_dt, integrator="rk4"
        )
        sim_time = 10000 * LT

        # intermediate parameters
        inter_p_list = np.linspace(
            p[my_eParam] - args.spacing, p[my_eParam] + args.spacing, 11
        )
        J = np.zeros((len(inter_p_list)))

        for inter_p_idx, inter_p in enumerate(inter_p_list):
            p_sim[args.param] = inter_p
            my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
                Lorenz63,
                p_sim,
                sim_time=sim_time,
                sim_dt=args.sim_dt,
                y_init=args.y_init,
                integrator="rk4",
            )
            y, t = pp.discard_transient(y_sim, t_sim, transient_time)
            J[inter_p_idx] = objective_fun(y)
        # approximate the gradient with linear function
        coeffs = np.polyfit(inter_p_list, J, deg=1)
        dJdp[p_idx] = coeffs[0]
        print(f"dJ/d{args.param} = {dJdp[p_idx]}")
    direct_results = {
        "dJdp": dJdp,
        "beta_list": beta_list,
        "rho_list": rho_list,
        "sigma_list": sigma_list,
        "y_init": args.y_init,
    }

    print(f"Saving results.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(
        f"local_results/lorenz63/direct_results_{dt_string}.pickle",
        direct_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", nargs="+", type=float)
    parser.add_argument("--rho", nargs="+", type=float)
    parser.add_argument("--sigma", nargs="+", type=float)
    parser.add_argument("--y_init", nargs="+", type=float)
    parser.add_argument("--sim_dt", type=float)
    parser.add_argument("--param", type=str)
    parser.add_argument("--spacing", type=float)
    parsed_args = parser.parse_args()
    main(parsed_args)
