import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

import multiprocessing as mp

from adjoint_esn.rijke_galerkin.generate import main as generator


@dataclass()
class GenerateArgs:
    data_path: Path
    grad_adjoint: bool
    dx: float
    N_x: int
    dt: float
    simulation_time: float
    transient_time: float
    seed: int
    N_g: int
    N_c: int
    c_1: float
    c_2: float
    beta: float
    x_f: float
    tau: float
    heat_law: str
    damping: str


def run_sim(params):
    beta = params[0]
    tau = params[1]
    beta_name = f"{beta:.2f}"
    beta_name = beta_name.replace(".", "_")
    tau_name = f"{tau:.2f}"
    tau_name = tau_name.replace(".", "_")
    sim_str = f"src/data_new/rijke_kings_poly_N_g_4_beta_{beta_name}_tau_{tau_name}.h5"
    print(sim_str)
    args_dict = {
        "data_path": Path(sim_str),
        "grad_adjoint": True,
        "dx": None,
        "N_x": None,
        "dt": 1e-3,
        "simulation_time": 1200,
        "transient_time": 200,
        "seed": None,
        "N_g": 4,
        "N_c": 10,
        "c_1": 0.1,
        "c_2": 0.06,
        "beta": beta,
        "x_f": 0.2,
        "tau": tau,
        "heat_law": "kings_poly",
        "damping": "modal",
    }
    args = GenerateArgs(**args_dict)
    generator(args)


def main():
    beta_list = np.array([9.0])
    tau_list = np.array([0.2])

    beta_mesh, tau_mesh = np.meshgrid(beta_list, tau_list)
    print(mp.cpu_count())
    # multiprocessing
    pool = mp.Pool(8)
    pool.map_async(
        run_sim,
        [(beta, tau) for (beta, tau) in zip(beta_mesh.flatten(), tau_mesh.flatten())],
    ).get()
    pool.close()


if __name__ == "__main__":
    main()
