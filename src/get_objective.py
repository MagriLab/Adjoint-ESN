import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils.dynamical_systems import Lorenz63


def objective_fun(u):
    return np.mean(u[:, 2])


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

    sim_time = max(args.loop_times)
    J = np.zeros((len(p_list), len(args.loop_times)))
    loop_steps = [pp.get_steps(loop_time, args.sim_dt) for loop_time in args.loop_times]
    for p_idx, p in enumerate(p_list):
        p_sim = {"beta": p[eParam.beta], "rho": p[eParam.rho], "sigma": p[eParam.sigma]}
        my_sys, y_sim, t_sim = pp.load_data_dyn_sys(
            Lorenz63,
            p_sim,
            sim_time=sim_time,
            sim_dt=args.sim_dt,
            y_init=args.y_init,
            integrator="rk4",
        )
        J[p_idx] = np.array(
            [objective_fun(y_sim[: loop_step + 1]) for loop_step in loop_steps]
        )

    objective_results = {
        "J": J,
        "beta_list": beta_list,
        "rho_list": rho_list,
        "sigma_list": sigma_list,
        "y_init": args.y_init,
        "loop_times": args.loop_times,
    }

    print(f"Saving results.", flush=True)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    pp.pickle_file(
        f"local_results/lorenz63/objective_results_{dt_string}.pickle",
        objective_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", nargs="+", type=float)
    parser.add_argument("--rho", nargs="+", type=float)
    parser.add_argument("--sigma", nargs="+", type=float)
    parser.add_argument("--y_init", nargs="+", type=float)
    parser.add_argument("--loop_times", nargs="+", type=float)
    parser.add_argument("--sim_dt", type=float)
    parsed_args = parser.parse_args()
    main(parsed_args)
