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
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import signals
from adjoint_esn.utils.enums import eParam, get_eVar
from adjoint_esn.utils import errors

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 14})
rc("text", usetex=True)
save_fig = True
same_washout = False

model_paths = [
    Path(
        "local_results/rijke/run_20231029_153121"
    ),  # rijke with reservoir, trained on beta = 1,2,3,4,5
    Path(
        "local_results/rijke/run_20240307_175258"
    ),  # rijke with reservoir, trained on beta = 6,6.5,7,7.5,8
]
legend_str = [
    "True",
    "$\\beta_{\mathrm{train}}=[1,5]$",
    "$\\beta_{\mathrm{train}}=[6,8]$",
]
data_dir = Path("data")


def get_amp_spec(dt, y, remove_mean=True, periodic=False):
    # remove mean
    if remove_mean == True:
        y = y - np.mean(y)
    if periodic:
        T_period = signals.period(y, dt)
        data_omega = 2 * np.pi / T_period
        print("Omega = ", data_omega)
        print("Period = ", T_period)
        # take the maximum number of periods
        # the real period isn't an exact multiple of the sampling time
        # therefore, the signal doesn't repeat itself at exact integer indices
        # so calculating the number of time steps in each period
        # does not work in order to cut the signal at the maximum number of periods
        # that's why we will cut between peaks, which is a more reliable option
        # though still not exact
        min_dist = pp.get_steps(T_period - 0.1, dt)
        (start_pk_idx, end_pk_idx) = signals.periodic_signal_peaks(y, T=min_dist)
        y_pre_fft = y[
            start_pk_idx:end_pk_idx
        ]  # don't include end peak for continuous signal
    else:
        y_pre_fft = y

    # find asd
    omega, amp_spec = signals.amplitude_spectrum(y_pre_fft, dt)
    # omega, psd = signals.power_spectral_density(y_pre_fft, dt)

    # to get the harmonic frequency from the asd
    # asd_peaks = find_peaks(asd, threshold=0.1)[0]
    # harmonic_freq = omega[asd_peaks][0]
    return omega, amp_spec


fig_name = "quasi2"

if fig_name == "period_double":
    test_param_list = [[7.5, 0.3]]
    periodic = True
    titles = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    LT = 1.0
    t_label = "$t$"
    test_loop_times = [20, 2000]
elif fig_name == "quasi":
    test_param_list = [[6.1, 0.2]]
    periodic = False
    titles = [["(g)", "(h)", "(i)"], ["(j)", "(k)", "(l)"]]
    LT = 1.0
    t_label = "$t$"
    test_loop_times = [20, 2000]
elif fig_name == "quasi2":
    test_param_list = [[6.6, 0.22]]
    periodic = False
    titles = [["(g)", "(h)", "(i)"], ["(j)", "(k)", "(l)"]]
    LT = 1.0
    t_label = "$t$"
    test_loop_times = [20, 2000]
elif fig_name == "chaotic":
    test_param_list = [[7.6, 0.22]]
    periodic = False
    titles = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    LT = 8.5
    t_label = "$t [LT]$"
    test_loop_times = [4 * LT, 1000 * LT]
elif fig_name == "chaotic2":
    test_param_list = [[8.7, 0.23]]
    periodic = False
    titles = [["(g)", "(h)", "(i)"], ["(j)", "(k)", "(l)"]]
    LT = 3.9
    t_label = "$t [LT]$"
    test_loop_times = [4 * LT, 1000 * LT]

n_ensemble = 5
test_loop_names = ["short", "long"]
figure_size = (15, 4)


# Plotting options
# true_color = "#C7C7C7"  # light grey
# pred_color = "#03BDAB"  # teal
# pred_color2 = "#5D00E6"  # dark purple

# color set 2
# true_color = "#3CB371"  # green
# pred_color = "#FEAC16"  # orange 
# pred_color2 = "#926FDB" # purple

true_color = "#03BDAB" # teal
pred_color = "#FEAC16"  # orange 
pred_color2 = "#5D00E6"  # dark purple

true_lw = 5.0
pred_lw = 2.5
pred_lw2 = 1.5
true_ls = "-"
pred_ls = "-"
pred_ls2 = "--"

# DATA creation
integrator = "odeint"
n_models = len(model_paths)
ESN_list = [[None] * n_ensemble for _ in range(n_models)]

for model_idx, model_path in enumerate(model_paths):
    config = post.load_config(model_path)
    results = pp.unpickle_file(model_path / "results.pickle")[0]

    # number of galerkin modes
    N_g = config.simulation.N_g

    # simulation options
    sim_time = config.simulation.sim_time
    sim_dt = config.simulation.sim_dt

    # which regimes to use for training and validation
    train_param_list = results["training_parameters"]
    train_idx_list = np.arange(len(train_param_list))

    transient_time = config.simulation.transient_time

    noise_level = config.simulation.noise_level

    random_seed = config.random_seed

    # network time step
    network_dt = config.model.network_dt

    washout_time = config.model.washout_time

    # which states to use as input and output
    # for standard ESN these should be the same, e.g. both 'eta_mu'
    # for Rijke ESN, input and output should be 'eta_mu_v_tau' and 'eta_mu' respectively
    input_vars = config.model.input_vars
    output_vars = config.model.output_vars

    eInputVar = get_eVar(input_vars, N_g)
    eOutputVar = get_eVar(output_vars, N_g)

    plt_idx = [eOutputVar.mu_1]
    plt_idx_pairs = [[eOutputVar.mu_1, eOutputVar.mu_2]]

    # which system parameter is passed to the ESN
    param_vars = config.model.param_vars

    # if using Rijke ESN what is the order of u_f(t-tau) in the inputs,
    # [u_f(t-tau), u_f(t-tau)^2 ..., u_f(t-tau)^(u_f_order)]
    u_f_order = config.model.u_f_order

    # length of training time series
    train_time = config.train.time

    loop_names = ["train"]
    loop_times = [train_time]

    print("Loading training data.")
    DATA = {}
    for loop_name in loop_names:
        DATA[loop_name] = {
            "u_washout": [],
            "p_washout": [],
            "u": [],
            "p": [],
            "y": [],
            "t": [],
        }

    for p_idx, p in enumerate(train_param_list):
        p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
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

        regime_data = pp.create_dataset(
            y_sim,
            t_sim,
            p_sim,
            network_dt=network_dt,
            transient_time=transient_time,
            washout_time=washout_time,
            loop_times=loop_times,
            loop_names=loop_names,
            input_vars=input_vars,
            output_vars=output_vars,
            param_vars=param_vars,
            N_g=N_g,
            u_f_order=u_f_order,
            noise_level=noise_level,
            random_seed=random_seed,
            tau=p_sim["tau"],
        )

        for loop_name in loop_names:
            [
                DATA[loop_name][var].append(regime_data[loop_name][var])
                for var in DATA[loop_name].keys()
            ]

    # dimension of the inputs
    dim = DATA["train"]["u"][0].shape[1]

    # get the properties of the best ESN from the results
    (
        ESN_dict,
        hyp_param_names,
        hyp_param_scales,
        hyp_params,
    ) = post.get_ESN_properties_from_results(config, results, dim)
    ESN_dict["verbose"] = False
    print(ESN_dict)
    [
        print(f"{hyp_param_name}: {hyp_param}")
        for hyp_param_name, hyp_param in zip(hyp_param_names, hyp_params)
    ]

    # generate and train ESN realisations
    for e_idx in range(n_ensemble):
        # fix the seeds
        input_seeds = [5 * e_idx, 5 * e_idx + 1, 5 * e_idx + 2]
        reservoir_seeds = [5 * e_idx + 3, 5 * e_idx + 4]

        # expand the ESN dict with the fixed seeds
        ESN_dict["input_seeds"] = input_seeds
        ESN_dict["reservoir_seeds"] = reservoir_seeds

        # create an ESN
        print(f"Creating ESN {e_idx+1}/{n_ensemble}.", flush=True)
        my_ESN = post.create_ESN(
            ESN_dict, config.model.type, hyp_param_names, hyp_param_scales, hyp_params
        )
        print("Training ESN.")
        my_ESN.train(
            DATA["train"]["u_washout"],
            DATA["train"]["u"],
            DATA["train"]["y"],
            P_washout=DATA["train"]["p_washout"],
            P_train=DATA["train"]["p"],
            train_idx_list=train_idx_list,
        )
        ESN_list[model_idx][e_idx] = my_ESN

test_sim_time = max(test_loop_times) + transient_time + washout_time
# Predict on the test dataset
for p_idx, p in enumerate(test_param_list):
    fig = plt.figure(figsize=figure_size, constrained_layout=True)
    subfigs = fig.subfigures(1, 3, width_ratios=[1.8, 1, 1.2])

    p_sim = {"beta": p[eParam.beta], "tau": p[eParam.tau]}
    y_sim, t_sim = pp.load_data(
        beta=p_sim["beta"],
        tau=p_sim["tau"],
        x_f=0.2,
        N_g=N_g,
        sim_time=test_sim_time,
        sim_dt=sim_dt,
        data_dir=data_dir,
    )

    data = pp.create_dataset(
        y_sim,
        t_sim,
        p_sim,
        network_dt=network_dt,
        transient_time=transient_time,
        washout_time=washout_time,
        loop_times=test_loop_times,
        loop_names=test_loop_names,
        input_vars=input_vars,
        output_vars=output_vars,
        param_vars=param_vars,
        N_g=N_g,
        u_f_order=u_f_order,
        start_idxs=[0, 0],
        tau=p_sim["tau"],
    )

    Y_PRED_SHORT = [[None] * n_ensemble for _ in range(n_models)]
    Y_PRED_LONG = [[None] * n_ensemble for _ in range(n_models)]
    plt_e_idx = [None] * n_models
    for model_idx in range(n_models):
        for e_idx in range(n_ensemble):
            my_ESN = ESN_list[model_idx][e_idx]
            print(f"Predicting ESN {e_idx+1}/{n_ensemble}.", flush=True)
            if hasattr(my_ESN, "tau"):
                my_ESN.tau = p_sim["tau"]

            _, y_pred_short = my_ESN.closed_loop_with_washout(
                U_washout=data["short"]["u_washout"],
                N_t=len(data["short"]["u"]),
                P_washout=data["short"]["p_washout"],
                P=data["short"]["p"],
            )
            y_pred_short = y_pred_short[1:]
            Y_PRED_SHORT[model_idx][e_idx] = y_pred_short

            # LONG-TERM AND STATISTICS, CONVERGENCE TO ATTRACTOR
            if same_washout:
                # Predict long-term
                _, y_pred_long = my_ESN.closed_loop_with_washout(
                    U_washout=data["long"]["u_washout"],
                    N_t=len(data["long"]["u"]),
                    P_washout=data["long"]["p_washout"],
                    P=data["long"]["p"],
                )
                y_pred_long = y_pred_long[1:]
            else:
                y0 = np.zeros((1, data["long"]["u_washout"].shape[1]))
                y0[0, 0] = 1.0
                u_washout_auto = np.repeat(y0, [len(data["long"]["u_washout"])], axis=0)
                transient_steps = pp.get_steps(transient_time, network_dt)
                # add the transient time that will be discarded later
                N_t_long = transient_steps + len(data["long"]["u"])
                p_long0 = np.zeros((1, data["long"]["p"].shape[1]))
                p_long0[0] = data["long"]["p"][0]
                p_long = np.repeat(p_long0, [N_t_long], axis=0)
                # Predict long-term
                _, y_pred_long = my_ESN.closed_loop_with_washout(
                    U_washout=u_washout_auto,
                    N_t=N_t_long,
                    P_washout=data["long"]["p_washout"],
                    P=p_long,
                )
                y_pred_long = y_pred_long[1:]
                y_pred_long = y_pred_long[transient_steps:, :]

            Y_PRED_LONG[model_idx][e_idx] = y_pred_long
        
        pred_short_error = np.array(
            [errors.rel_L2(data["short"]["y"], Y_PRED_SHORT[model_idx][i]) for i in range(n_ensemble)]
        )
        print("Short term prediction errors: ", pred_short_error)
        plt_e_idx[model_idx] = np.argmin(pred_short_error)
        print("Best idx: ", plt_e_idx[model_idx])
    # SHORT-TERM AND TIME-ACCURATE PREDICTION
    # Plot short term prediction timeseries of the best of ensemble
    
    for i in range(len(plt_idx)):
        ax = subfigs[0].add_subplot(len(plt_idx) + 1, 1, i + 1)
        vis.plot_lines(
            (data["short"]["t"] - data["short"]["t"][0]) / LT,
            data["short"]["y"][:, plt_idx[i]],
            *[
                Y_PRED_SHORT[model_idx][plt_e_idx[model_idx]][:, plt_idx[i]]
                for model_idx in range(n_models)
            ],
            xlabel=t_label,
            ylabel=f"$\{plt_idx[i].name}$",
            linestyle=[true_ls, pred_ls, pred_ls2],
            linewidth=[true_lw, pred_lw, pred_lw2],
            color=[true_color, pred_color, pred_color2],
        )
        # if i == 0:
        #     ax.legend(["True", "ESN"], loc="center", bbox_to_anchor=[0.5, 1.2], ncol=2)
        # if i < len(plt_idx)-1:
        #     ax.set_xticklabels([])
        ax.annotate(titles[0][0], xy=(0.015, 0.85), xycoords="axes fraction")

    ax = subfigs[0].add_subplot(len(plt_idx) + 1, 1, len(plt_idx) + 1)
    vis.plot_lines(
        (data["short"]["t"] - data["short"]["t"][0]) / LT,
        sens.acoustic_energy_inst(data["short"]["y"], N_g),
        *[
            sens.acoustic_energy_inst(Y_PRED_SHORT[model_idx][plt_e_idx[model_idx]], N_g)
            for model_idx in range(n_models)
        ],
        xlabel=t_label,
        ylabel="$E_{ac}$",
        linestyle=[true_ls, pred_ls, pred_ls2],
        linewidth=[true_lw, pred_lw, pred_lw2],
        color=[true_color, pred_color, pred_color2],
    )
    ax.annotate(titles[1][0], xy=(0.015, 0.85), xycoords="axes fraction")

    # Plot phase plot of the best of ensemble
    # phase_space_steps = pp.get_steps(phase_space_steps_arr[p_idx], network_dt)

    # for i in range(len(plt_idx_pairs)):
    #     ax = subfigs[1].add_subplot(len(plt_idx_pairs), 1, i+1)
    #     vis.plot_phase_space(
    #         data["long"]["y"][-phase_space_steps:],
    #         Y_PRED_LONG[best_idx][-phase_space_steps:],
    #         idx_pair=plt_idx_pairs[i],
    #         linestyle=[true_ls, pred_ls],
    #         linewidth=[true_phase_lw, pred_phase_lw],
    #         color=[true_color, pred_color],
    #         xlabel=f"$\{plt_idx_pairs[i][0].name}$",
    #         ylabel=f"$\{plt_idx_pairs[i][1].name}$",
    #     )

    # Plot statistics
    for i in range(len(plt_idx)):
        ax = subfigs[1].add_subplot(len(plt_idx) + 1, 1, i + 1)
        vis.plot_statistics_ensemble(
            *[
                [Y_PRED_LONG[m][e][:, plt_idx[i]] for e in range(n_ensemble)]
                for m in range(n_models)
            ],
            y_base=data["long"]["y"][:, plt_idx[i]],
            xlabel=f"$\{plt_idx[i].name}$",
            ylabel="PDF",
            linestyle=[true_ls, pred_ls, pred_ls2],
            linewidth=[true_lw, pred_lw, pred_lw2],
            color=[true_color, pred_color, pred_color2],
        )
        ax.annotate(titles[0][1], xy=(0.03, 0.85), xycoords="axes fraction")

    ax = subfigs[1].add_subplot(len(plt_idx) + 1, 1, len(plt_idx) + 1)
    vis.plot_statistics_ensemble(
        *[
            [
                sens.acoustic_energy_inst(Y_PRED_LONG[m][e], N_g)
                for e in range(n_ensemble)
            ]
            for m in range(n_models)
        ],
        y_base=sens.acoustic_energy_inst(data["long"]["y"], N_g),
        xlabel="$E_{ac}$",
        ylabel="PDF",
        linestyle=[true_ls, pred_ls, pred_ls2],
        linewidth=[true_lw, pred_lw, pred_lw2],
        color=[true_color, pred_color, pred_color2],
    )
    ax.annotate(titles[1][1], xy=(0.03, 0.85), xycoords="axes fraction")

    # # Plot ASD
    for i in range(len(plt_idx)):
        ax = subfigs[2].add_subplot(len(plt_idx) + 1, 1, i + 1)
        omega, amp_spec = get_amp_spec(
            network_dt,
            data["long"]["y"][:, plt_idx[i]],
            remove_mean=True,
            periodic=periodic,
        )
        AS_PRED = [[None] * n_ensemble for _ in range(n_models)]
        OMEGA_PRED = [[None] * n_ensemble for _ in range(n_models)]
        for model_idx in range(n_models):
            for e_idx in range(n_ensemble):
                OMEGA_PRED[model_idx][e_idx], AS_PRED[model_idx][e_idx] = get_amp_spec(
                    network_dt,
                    Y_PRED_LONG[model_idx][e_idx][:, plt_idx[i]],
                    remove_mean=True,
                    periodic=periodic,
                )
        ax.annotate(titles[0][2], xy=(0.03, 0.85), xycoords="axes fraction")

        vis.plot_asd(  # *[AS_PRED[e] for e in range(n_ensemble)],
            asd_y=[AS_PRED[model_idx][plt_e_idx[model_idx]] for model_idx in range(n_models)],
            omega_y=[OMEGA_PRED[model_idx][plt_e_idx[model_idx]] for model_idx in range(n_models)],
            asd_y_base=amp_spec,
            omega_y_base=omega,
            range=2.5,
            xlabel="$\omega$",
            ylabel=f"Amplitude($\{plt_idx[i].name}$)",
            linestyle=[true_ls, pred_ls, pred_ls2],
            linewidth=[true_lw, pred_lw, pred_lw2],
            color=[true_color, pred_color, pred_color2],
            alpha=0.8,
        )
        plt.legend(legend_str, loc="upper right", handlelength=1.0, handletextpad=0.5)
    ax = subfigs[2].add_subplot(len(plt_idx) + 1, 1, len(plt_idx) + 1)
    omega, amp_spec = get_amp_spec(
        network_dt,
        sens.acoustic_energy_inst(data["long"]["y"], N_g),
        remove_mean=True,
        periodic=periodic,
    )
    AS_PRED = [[None] * n_ensemble for _ in range(n_models)]
    OMEGA_PRED = [[None] * n_ensemble for _ in range(n_models)]
    for model_idx in range(n_models):
        for e_idx in range(n_ensemble):
            OMEGA_PRED[model_idx][e_idx], AS_PRED[model_idx][e_idx] = get_amp_spec(
                network_dt,
                sens.acoustic_energy_inst(Y_PRED_LONG[model_idx][e_idx], N_g),
                periodic=periodic,
            )
    vis.plot_asd(  # *[AS_PRED[e] for e in range(n_ensemble)],
        asd_y=[AS_PRED[model_idx][plt_e_idx[model_idx]] for model_idx in range(n_models)],
        omega_y=[OMEGA_PRED[model_idx][plt_e_idx[model_idx]] for model_idx in range(n_models)],
        asd_y_base=amp_spec,
        omega_y_base=omega,
        range=10,
        xlabel="$\omega$",
        ylabel="Amplitude($E_{ac}$)",
        linestyle=[true_ls, pred_ls, pred_ls2],
        linewidth=[true_lw, pred_lw, pred_lw2],
        color=[true_color, pred_color, pred_color2],
        alpha=0.8,
    )
    ax.annotate(titles[1][2], xy=(0.03, 0.85), xycoords="axes fraction")
    if save_fig:
        if len(test_param_list) == 1:
            fig.savefig(f"graphics/figure_{fig_name}_v8.png", bbox_inches="tight")
            # fig.savefig(f"graphics/figure_{fig_name}_v5.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"graphics/figure_{fig_name}_{p_idx}.png", bbox_inches="tight")
plt.show()
