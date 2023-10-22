import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

import adjoint_esn.utils.solve_ode as solve_ode

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from adjoint_esn.rijke_galerkin.solver import Rijke


# @todo: saving config file
def make_dir(path):
    """Create the directory to save the simulation data"""
    if not path.suffix == ".h5":
        path.with_suffix(".h5")

    if path.exists():
        print(f"setup_directory() :: {path} already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(path.absolute())


def write_h5(path, data):
    """Write simulation dictionary to a .h5 file"""
    hf = h5py.File(path, "w")  # with hf
    print(path, flush=True)
    for k, v in data.items():
        hf.create_dataset(k, data=v)
    print(path.absolute())
    hf.close()


# @todo: rijke config, instead of passing everything in argparse
def main(args):
    """Run the Rijke Galerkin solver"""
    print("Initialising Rijke Galerkin solver.", flush=True)

    make_dir(args.data_path)

    # Create rijke system instance with the given parameters
    rjk = Rijke(
        N_g=args.N_g,
        N_c=args.N_c,
        c_1=args.c_1,
        c_2=args.c_2,
        beta=args.beta,
        x_f=args.x_f,
        tau=args.tau,
        heat_law=args.heat_law,
        damping=args.damping,
    )

    # Initial conditions
    y0 = np.zeros(rjk.N_dim + 1)
    if args.seed is not None:
        rand = np.random.RandomState(seed=args.seed)
        y0[0 : rjk.N_dim] = rand.rand(rjk.N_dim)
    else:
        y0[0] = 1

    # Temporal grid
    t = np.arange(0, args.simulation_time + args.dt, args.dt)

    print("Running simulation.", flush=True)
    # Solve ODE using odeint
    y = solve_ode.integrate(rjk.ode, y0, t, integrator=args.integrator)

    eta = y[:, 0 : rjk.N_g]  # Galerkin variables velocity
    mu = y[:, rjk.N_g : 2 * rjk.N_g]  # Galerkin variables pressure

    # v = y[:, 2 * rjk.N_g : rjk.N_dim]  # advection variable for time delay

    # q_dot = y[:, -2]  # heat release rate

    # J = 1 / t[-1] * (y[-1, -1] - y[0, -1])  # time-averaged acoustic energy

    data_dict = {
        "N_g": rjk.N_g,
        "N_c": rjk.N_c,
        "c_1": rjk.c_1,
        "c_2": rjk.c_2,
        "beta": rjk.beta,
        "x_f": rjk.x_f,
        "tau": rjk.tau,
        "heat_law": rjk.heat_law,
        "damping": rjk.damping,
        "t_transient": args.transient_time,
        "t": t,
        "y": y,
        # "J": J,
    }

    if args.grad_adjoint:
        N_transient = int(np.round(args.transient_time / args.dt))
        y_bar = y[N_transient:, :]
        t_bar = t[N_transient:] - t[N_transient]
        # adjoint problem
        adjT = np.zeros(rjk.N_dim + 2)
        adj = solve_ode.integrate(
            rjk.adjoint_ode,
            adjT,
            np.flip(t_bar),
            integrator=args.integrator,
            args=(t_bar, 1 / args.dt, y_bar),
        )
        dJ_dbeta = 1 / t_bar[-1] * adj[-1, -2]
        dJ_dtau = 1 / t_bar[-1] * adj[-1, -1]
        data_dict["adj"] = adj
        data_dict["dJ_dbeta"] = dJ_dbeta
        data_dict["dJ_dtau"] = dJ_dtau

    # Spatial grid
    if args.dx is not None and args.N_x is not None:
        if args.dx is not None:
            x = np.arange(0, 1 + args.dx, args.dx)
        elif args.N_x is not None:
            x = np.linspace(0, 1, args.N_x)

        P = Rijke.toPressure(rjk.N_g, mu, x)  # pressure field
        U = Rijke.toVelocity(rjk.N_g, eta, x)  # velocity field

        data_dict["x"] = x
        data_dict["P"] = P
        data_dict["U"] = U

    print("Writing to file.", flush=True)
    write_h5(args.data_path, data_dict)

    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates Rijke system data using Galerkin decomposition method."
    )

    parser.add_argument(
        "--integrator",
        type=str,
        default="odeint",
        help="which integration scheme to use; forward_euler, odeint, or rk4",
    )

    parser.add_argument(
        "--data_path", type=Path, required=True, help="path to save the data to"
    )

    parser.add_argument(
        "--grad_adjoint",
        type=bool,
        default=False,
        help="whether to run the adjoint code to obtain the gradient",
    )
    parser.add_argument("--dx", type=float, default=None, help="spatial spacing")

    parser.add_argument("--N_x", type=int, default=None, help="number of grid points")

    parser.add_argument(
        "--dt", type=float, default=1e-3, help="temporal spacing (default: 1e-3)"
    )

    parser.add_argument(
        "--simulation_time", default=500, type=float, help="total simulation time"
    )

    parser.add_argument(
        "--transient_time", default=200, type=float, help="transient time to be saved"
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for initial conditions (default: 1)"
    )

    parser.add_argument(
        "--N_g", type=int, default=10, help="number of Galerkin modes (default: 10)"
    )

    parser.add_argument(
        "--N_c", type=int, default=10, help="number of Chebyshev points (default: 10)"
    )

    parser.add_argument(
        "--c_1",
        type=float,
        default=0.1,
        help="first damping coefficient (default: 0.1)",
    )

    parser.add_argument(
        "--c_2",
        type=float,
        default=0.06,
        help="second damping coefficient (default: 0.06)",
    )

    parser.add_argument(
        "--beta", type=float, required=True, help="heat release strength"
    )

    parser.add_argument("--x_f", type=float, required=True, help="flame location")

    parser.add_argument(
        "--tau", type=float, required=True, help="flame velocity time delay"
    )

    parser.add_argument(
        "--heat_law",
        type=str,
        default="kings",
        help="heat law, kings (default) or sigmoid",
    )

    parser.add_argument(
        "--damping",
        type=str,
        default="modal",
        help="damping, modal (default) or constant",
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
