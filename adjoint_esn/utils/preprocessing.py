import pickle
from functools import partial
from itertools import product

import h5py
import numpy as np
from scipy import signal

import adjoint_esn.utils.solve_ode as solve_ode
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


def load_data(
    beta,
    tau,
    N_g,
    sim_time,
    data_dir,
    x_f=0.2,
    sim_dt=1e-3,
    integrator="odeint",
    y_init=None,
):
    # construct data path name
    beta_name = f"{beta:.2f}"
    beta_name = beta_name.replace(".", "_")
    tau_name = f"{tau:.2f}"
    tau_name = tau_name.replace(".", "_")
    data_path = (
        data_dir / f"rijke_kings_poly_N_g_{N_g}_beta_{beta_name}_tau_{tau_name}.h5"
    )
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
    y0_default = np.zeros(my_rijke.N_dim)
    y0_default[0] = 1.0

    run_sim = False
    if data_path.exists():
        # load simulation data
        data_dict = read_h5(data_path)

        # check the time step
        loaded_sim_dt = data_dict["t"][1] - data_dict["t"][0]
        if not np.equal(loaded_sim_dt, sim_dt):
            print(
                f"The time step of the loaded simulation data is different from the desired one:{loaded_sim_dt}!={sim_dt}"
            )
            run_sim = True

        # check if loaded simulation is long enough
        if sim_time > data_dict["t"][-1]:
            print(f'Simulation not long enough: {sim_time}>{data_dict["t"][-1]}')
            run_sim = True

        # check integrator
        if integrator != "odeint":
            print(f"Integrator not odeint, saved data was generated with odeint.")
            run_sim = True

        if y_init is not None:
            if not all(np.equal(y_init, y0_default)):
                run_sim = True

        if run_sim == False:
            # reduce the simulation time steps if necessary
            sim_time_steps = get_steps(sim_time, loaded_sim_dt)
            y = data_dict["y"][: sim_time_steps + 1]
            t = data_dict["t"][: sim_time_steps + 1]
    else:
        run_sim = True

    if run_sim:
        # run simulation
        if y_init is not None:
            y0 = y_init
        else:
            y0 = y0_default

        # temporal grid
        t = np.arange(0, sim_time + sim_dt, sim_dt)

        # solve ODE using odeint
        y = solve_ode.integrate(my_rijke.ode, y0, t, integrator)

    return y[:, : 2 * N_g + 10], t


def create_state(y, N_g, include, N_c=10, u_f_order=0, N_tau=0):
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
            # y_new = np.hstack((y_new, y[:, u_f_idx][:, None] ** (order + 1)))
            eta_tau = np.vstack((np.zeros((N_tau, N_g)), y[:-N_tau, :N_g]))
            u_f_tau = Rijke.toVelocity(N_g, eta_tau, x=0.2)
            y_new = np.hstack((y_new, u_f_tau ** (order + 1)))
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


def generate_noise(y, noise_level, random_seed, noise_std=0.0):
    rnd = np.random.RandomState(seed=random_seed)
    if noise_std <= 0:
        y_std = np.std(y, axis=0)
        noise = rnd.normal(
            loc=np.zeros(y.shape[1]), scale=noise_level / 100 * y_std, size=y.shape
        )
    else:
        noise = rnd.normal(loc=np.zeros(y.shape[1]), scale=noise_std, size=y.shape)
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
    random_seed=None,
    tau=0,
    loop_names=None,
    start_idxs=None,
    noise_std=0,
    filter=None,
    f0=5,  # omega = 10 * pi, which is 10 harmonics more or less
):
    y = y[:, : 2 * N_g + 10]  # get rid of extra columns
    y, t = upsample(y, t, network_dt)
    y, t = discard_transient(y, t, transient_time)

    # add noise
    if noise_level > 0 or noise_std > 0:
        y = y + generate_noise(y, noise_level, random_seed, noise_std)
    # filter noise
    if filter == "low":  # apply low-pass filter to the data
        for y_idx in range(y.shape[1]):
            # 3rd order butterworth low-pass filter
            b, a = signal.butter(3, f0, fs=1 / network_dt)
            y[:, y_idx] = signal.filtfilt(b, a, y[:, y_idx])

    # choose input and output states
    N_tau = int(np.round(tau / network_dt))
    uu = choose_state(input_vars)(y, N_g=N_g, u_f_order=u_f_order, N_tau=N_tau)
    yy = choose_state(output_vars)(y, N_g=N_g, u_f_order=u_f_order, N_tau=N_tau)

    # separate into washout and loops
    N_washout, *N_loops = [
        get_steps(t, network_dt) for t in [washout_time, *loop_times]
    ]

    # create a dictionary to store the data
    data = {}
    start_idx = 0
    for loop_idx, N_loop in enumerate(N_loops):
        if start_idxs is not None:
            start_idx = start_idxs[loop_idx]

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

        if start_idxs is None:
            start_idx += N_washout + N_loop

    # add something for the case when len(N_loop) = 0?
    return data


# data loading and dataset creating functions that apply to general dynamical systems
# as defined in the utils/dynamical_systems file
def load_data_dyn_sys(
    sys, params, sim_time, sim_dt, integrator="odeint", y_init=None, random_seed=None
):
    # Create system object
    my_sys = sys(**params)

    # set up initial conditions
    if y_init is not None:
        y0 = y_init
    else:
        rnd = np.random.RandomState(seed=random_seed)
        y0 = rnd.randn(my_sys.N_dim)

    # temporal grid
    t = np.linspace(0, sim_time, int(np.round(sim_time / sim_dt)) + 1)

    # solve ODE using odeint
    y = solve_ode.integrate(my_sys.ode, y0, t, integrator)
    return my_sys, y, t


def create_dataset_dyn_sys(
    my_sys,
    y,
    t,
    p,
    network_dt,
    transient_time,
    washout_time,
    loop_times,
    input_vars,
    param_vars,
    noise_level=0,
    random_seed=0,
    loop_names=None,
    start_idxs=None,
    noise_std=0,
    filter=None,
    f0=5,
):
    y, t = upsample(y, t, network_dt)
    y, t = discard_transient(y, t, transient_time)

    # add noise
    if noise_level > 0 or noise_std > 0:
        y = y + generate_noise(y, noise_level, random_seed, noise_std)

    # filter noise
    if filter == "low":  # apply low-pass filter to the data
        for y_idx in range(y.shape[1]):
            # 3rd order butterworth low-pass filter
            b, a = signal.butter(3, f0, fs=1 / network_dt)
            y[:, y_idx] = signal.filtfilt(b, a, y[:, y_idx])

    # separate into washout and loops
    N_washout, *N_loops = [
        get_steps(t, network_dt) for t in [washout_time, *loop_times]
    ]

    # extract input-output variables
    vars = my_sys.get_eVar()
    input_var_idxs = [vars[input_var] for input_var in input_vars]
    uu = y[:, input_var_idxs]
    yy = y[:, input_var_idxs]

    # create a dictionary to store the data
    data = {}
    start_idx = 0
    for loop_idx, N_loop in enumerate(N_loops):
        if start_idxs is not None:
            start_idx = start_idxs[loop_idx]

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

        if start_idxs is None:
            start_idx += N_washout + N_loop

    # add something for the case when len(N_loop) = 0?
    return data
