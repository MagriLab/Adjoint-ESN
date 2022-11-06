import numpy as np
from scipy.integrate import odeint
import h5py

def make_dir(path):
    """Create the directory to save the simulation data"""
    if not path.suffix == ".h5":
        path.with_suffix(".h5")

    if path.exists():
        raise FileExistsError(f"setup_directory() :: {path} already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_h5(path, data):
    """Write simulation dictionary to a .h5 file"""
    hf = h5py.File(path, "w")  # with hf

    for k, v in data.items():
        hf.create_dataset(k, data=v)

    hf.close()

def forward_euler(ddt, u0, t, *args):
    """ Integrate using the first order forward euler method
    ddt: ODE that describes the system dynamics, ddt = func(y,t)
    u0: initial conditions
    t: integration time
    """
    # initialize u
    u = np.empty((len(t), len(u0)))
    u[0] = u0
    # integrate
    for i in range(1, len(t)):
        u[i] = u[i-1] + (t[i] - t[i-1]) * ddt(u[i-1],t[i-1],*args)
    return u

def integrate(dynamical_system, u0, t, integrator = 'odeint', data_path = None):
    # Dictionary of integrators
    integrators_dict = {'forward_euler': forward_euler,
                        'odeint': odeint,
                       }
    if integrator not in integrators_dict.keys():
        raise ValueError(f'{integrator} is not in the list of allowed integrators. Choose from {integrators_dict.keys()}')
    # set the integrator
    integrator = integrators_dict[integrator]
    # integrate
    print("Running solver.")
    u = integrator(dynamical_system.ode, u0, t)

    # write to h5 file if data path is given
    if data_path is not None:
        make_dir(data_path)
        data_dict = {'u': u,
        't': t,
        'params': dynamical_system.params,
        }
        print("Writing to file.")
        write_h5(data_path, data_dict)
        print("Done.")

    return u



