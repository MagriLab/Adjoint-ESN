import pickle
from functools import partial
from itertools import product

import h5py
import numpy as np
from scipy.integrate import odeint

from adjoint_esn.rijke_galerkin.solver import Rijke


def read_h5(path):
    """Read from simulation dictionary in a .h5 file

    Args:
        path: file path to data
    Returns:
        data_dictionary: dictionary that contains the items in the h5 file
    """
    data_dict = {}
    with h5py.File(path, "r") as hf:
        for k in hf.keys():
            data_dict[k] = hf.get(k)[()]
    return data_dict


def unpickle_file(file_name):
    file = open(file_name, "rb")
    data = pickle.load(file)
    return data, file


def pickle_file(file_name, data):
    file = open(file_name, "wb")
    pickle.dump(data, file)
    file.close()
    return


def make_param_mesh(param_list):
    # create a mesh of parameters from the given list of lists
    param_prod = product(*param_list, repeat=1)
    param_mesh = np.array([list(p) for p in param_prod])
    return param_mesh


def get_steps(t, dt):
    return int(np.round(t / dt))


def load_data(beta, tau, N_g, sim_time, data_dir, x_f=0.2, sim_dt=1e-3):
    # construct data path name
    beta_name = f"{beta:.2f}"
    beta_name = beta_name.replace(".", "_")
    tau_name = f"{tau:.2f}"
    tau_name = tau_name.replace(".", "_")
    data_path = (
        data_dir / f"rijke_kings_poly_N_g_{N_g}_beta_{beta_name}_tau_{tau_name}.h5"
    )

    if data_path.exists():
        # load simulation data
        data_dict = read_h5(data_path)

        # check the time step
        loaded_sim_dt = data_dict["t"][1] - data_dict["t"][0]
        if not np.equal(loaded_sim_dt, sim_dt):
            raise ValueError(
                f"The time step of the loaded simulation data is different from the desired one:{loaded_sim_dt}!={sim_dt}"
            )

        # check if loaded simulation is long enough
        if sim_time > data_dict["t"][-1]:
            raise ValueError(
                f'Simulation not long enough: {sim_time}>{data_dict["t"][-1]}'
            )

        # reduce the simulation time steps if necessary
        sim_time_steps = get_steps(sim_time, loaded_sim_dt)
        y = data_dict["y"][: sim_time_steps + 1]
        t = data_dict["t"][: sim_time_steps + 1]
    else:
        my_rijke = Rijke(
            N_g=N_g,
            N_c=10,
            c_1=0.1,
            c_2=0.06,
            beta=beta,
            x_f=x_f,
            tau=tau,
            heat_law="kings_poly",
            damping="modal",
        )
        # run simulation
        y0 = np.zeros(my_rijke.N_dim + 1)
        y0[0] = 1.0

        # temporal grid
        t = np.arange(0, sim_time + sim_dt, sim_dt)

        # solve ODE using odeint
        y = odeint(my_rijke.ode, y0, t, tfirst=True)

    return y, t


def create_state(y, N_g, include, N_c=10, u_f_order=0):
    # choose which variables to pass in the state vector
    if not include:
        # get galerkin amplitudes
        gal_idx = np.arange(2 * N_g)
        gal_idx = gal_idx.tolist()
        y_new = y[:, gal_idx]
    elif include == "v":
        # augment the input with the velocity history
        gal_idx = np.arange(2 * N_g + N_c)
        gal_idx = gal_idx.tolist()
        y_new = y[:, gal_idx]
    elif include == "v_tau":
        # get galerkin amplitudes
        gal_idx = np.arange(2 * N_g)
        gal_idx = gal_idx.tolist()
        y_new = y[:, gal_idx]
        # augment the input with the time delayed velocity, u_f(t-tau)
        u_f_idx = 2 * N_g + N_c - 1
        for order in range(u_f_order):
            y_new = np.hstack((y_new, y[:, u_f_idx][:, None] ** (order + 1)))
    else:
        raise ValueError("Not valid include.")
    return y_new


def choose_state(vars):
    if vars == "eta_mu":
        return partial(create_state, include=None)
    elif vars == "eta_mu_v":
        return partial(create_state, include="v")
    elif vars == "eta_mu_v_tau":
        return partial(create_state, include="v_tau")
    else:
        raise ValueError("Not valid state.")


def upsample(y, t, new_dt):
    dt = t[1] - t[0]

    upsample = get_steps(new_dt, dt)
    if upsample == 0:
        raise ValueError(
            f"Desired new time step is smaller than the simulation: {new_dt}<{dt}"
        )

    y_new = y[::upsample]
    t_new = t[::upsample]
    return y_new, t_new


def discard_transient(y, t, transient_time):
    dt = t[1] - t[0]

    N_transient = get_steps(transient_time, dt)

    y_new = y[N_transient:, :]
    t_new = t[N_transient:] - t[N_transient]
    return y_new, t_new


def create_input_output(u, y, t, N_washout, N_loop):
    # input
    u_washout = u[0:N_washout]
    u_loop = u[N_washout : N_washout + N_loop - 1]

    # output
    y_loop = y[N_washout + 1 : N_washout + N_loop]

    # output time
    t_loop = t[N_washout + 1 : N_washout + N_loop]
    return u_washout, u_loop, y_loop, t_loop


def generate_noise(y, noise_level, random_seed):
    y_std = np.std(y, axis=0)
    rnd = np.random.RandomState(seed=random_seed)
    noise = rnd.normal(
        loc=np.zeros(y.shape[1]), scale=noise_level / 100 * y_std, size=y.shape
    )
    return noise


def create_dataset(
    y,
    t,
    p,
    network_dt,
    transient_time,
    washout_time,
    loop_times,
    input_vars,
    output_vars,
    param_vars,
    N_g,
    u_f_order,
    noise_level=0,
    random_seed=0,
    loop_names=None,
):

    y, t = upsample(y, t, network_dt)
    y, t = discard_transient(y, t, transient_time)

    # add noise
    if noise_level > 0:
        y = y + generate_noise(y, noise_level, random_seed)

    # choose input and output states
    uu = choose_state(input_vars)(y, N_g=N_g, u_f_order=u_f_order)
    yy = choose_state(output_vars)(y, N_g=N_g, u_f_order=u_f_order)

    # separate into washout and loops
    N_washout, *N_loops = [
        get_steps(t, network_dt) for t in [washout_time, *loop_times]
    ]

    # create a dictionary to store the data
    data = {}
    start_idx = 0
    for loop_idx, N_loop in enumerate(N_loops):
        # get washout and loop data
        u_washout, u_loop, y_loop, t_loop = create_input_output(
            uu[start_idx:], yy[start_idx:], t[start_idx:], N_washout, N_loop + 1
        )
        # set the parameters
        p_list = [p[param_var] for param_var in param_vars]
        p_washout = p_list * np.ones((len(u_washout), 1))
        p_loop = p_list * np.ones((len(u_loop), 1))

        if loop_names is not None:
            loop_name = loop_names[loop_idx]
        else:
            loop_name = f"loop_{loop_idx}"
        data[loop_name] = {
            "u_washout": u_washout,
            "p_washout": p_washout,
            "u": u_loop,
            "p": p_loop,
            "y": y_loop,
            "t": t_loop,
        }

        start_idx += N_washout + N_loop

    # add something for the case when len(N_loop) = 0?
    return data
