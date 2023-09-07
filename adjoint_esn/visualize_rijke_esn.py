import matplotlib.pyplot as plt
import numpy as np

from adjoint_esn.utils.enums import get_eVar


def visualize_heat_release(my_ESN):
    # Visualize the heat release block of the Rijke ESN

    u_f_tau = np.arange(-5, 5, 0.1)
    beta_list = np.arange(1.0, 9.0, 1.0)

    output_vars = "eta_mu"
    eOutputVar = get_eVar(output_vars, my_ESN.N_g)

    color = plt.cm.tab20(np.linspace(0, 1, len(beta_list)))
    fig, axes = plt.subplots(1, my_ESN.N_g, figsize=(20, 3))

    for beta_idx, beta in enumerate(beta_list):
        for j in np.arange(1, my_ESN.N_g + 1):
            q_dot = np.zeros((len(beta_list), len(u_f_tau)))
            if my_ESN.input_weights_mode == "sparse_grouped_rijke":
                for order in range(my_ESN.u_f_order):
                    # get the input weights that correspond to heat release
                    # contribution of u_f(t-tau)
                    # in sparse_grouped_rijke different orders are not connected
                    u_f_idx = np.where(
                        my_ESN.W_in[
                            :, -my_ESN.N_param_dim - (my_ESN.u_f_order - order)
                        ].toarray()
                        != 0
                    )[0]
                    w_in_u_f = my_ESN.W_in[
                        u_f_idx, -my_ESN.N_param_dim - (my_ESN.u_f_order - order)
                    ].toarray() * (u_f_tau ** (order + 1))

                    # contribution of beta
                    w_in_beta = my_ESN.W_in[u_f_idx, -1].toarray() * (
                        (beta - my_ESN.norm_p[0]) / my_ESN.norm_p[1]
                    )

                    # heat release
                    q_dot[beta_idx] = q_dot[beta_idx] + np.dot(
                        my_ESN.W_out[u_f_idx, eOutputVar[f"mu_{j}"]],
                        np.tanh(w_in_u_f + w_in_beta),
                    )

            elif my_ESN.input_weights_mode == "sparse_grouped_rijke_dense":
                # in sparse_grouped_rijke_dense different orders are connected, so the index of beta is enough
                u_f_idx = np.where(my_ESN.W_in[:, -my_ESN.N_param_dim].toarray() != 0)[
                    0
                ]
                # contribution of beta
                w_in_beta = my_ESN.W_in[u_f_idx, -my_ESN.N_param_dim].toarray() * (
                    (beta - my_ESN.norm_p[0]) / my_ESN.norm_p[1]
                )

                w_in_u_f = 0
                for order in range(my_ESN.u_f_order):
                    # contribution of u_f(t-tau)
                    w_in_u_f = w_in_u_f + my_ESN.W_in[
                        u_f_idx, -my_ESN.N_param_dim - (my_ESN.u_f_order - order)
                    ].toarray() * (u_f_tau ** (order + 1))

                q_dot[beta_idx] = np.dot(
                    my_ESN.W_out[u_f_idx, eOutputVar[f"mu_{j}"]],
                    np.tanh(w_in_u_f + w_in_beta),
                )

            # We only look at the output for mu_1, hence sin(pi*x_f)
            q_dot[beta_idx] = (
                -(my_ESN.leak_factor)
                * q_dot[beta_idx]
                / (2 * my_ESN.dt * np.sin(j * np.pi * my_ESN.x_f))
            )

            axes[j - 1].plot(u_f_tau, q_dot[beta_idx], color=color[beta_idx])
            axes[j - 1].legend([f"beta = {beta}" for beta in beta_list], ncols=2)
            axes[j - 1].set_xlabel("u_f(t-tau)")
            axes[j - 1].set_ylabel("Heat release")
            axes[j - 1].set_title(f"mu_{j}")
            axes[j - 1].grid(visible=True)
    return


def visualize_acoustics(my_ESN):
    # Visualize the acoustics block of the Rijke ESN

    eta = np.arange(-10, 10, 0.1)
    mu = np.arange(-10, 10, 0.1)

    input_vars = "eta_mu_v_tau"
    eInputVar = get_eVar(input_vars, my_ESN.N_g)

    output_vars = "eta_mu"
    eOutputVar = get_eVar(output_vars, my_ESN.N_g)

    color = plt.cm.get_cmap("tab10").colors
    linestyles = ["-", "--", ":"]
    fig1 = plt.figure(figsize=(15, 3), constrained_layout=True)
    fig2 = plt.figure(figsize=(15, 3), constrained_layout=True)
    fig3 = plt.figure(figsize=(15, 3), constrained_layout=True)
    fig4 = plt.figure(figsize=(15, 3), constrained_layout=True)
    for j in np.arange(1, my_ESN.N_g + 1):
        # get the contribution of each eta mode
        eta_idx = np.where(my_ESN.W_in[:, eInputVar[f"eta_{j}"]].toarray() != 0)[0]
        w_in_eta = my_ESN.W_in[eta_idx, eInputVar[f"eta_{j}"]].toarray() * eta

        # plot effect of eta on mu
        ax1 = fig1.add_subplot(1, my_ESN.N_g, j)
        kk = 0
        for k in np.arange(1, my_ESN.N_g + 1):
            eta_on_mu = np.dot(
                my_ESN.W_out[eta_idx, eOutputVar[f"mu_{k}"]], np.tanh(w_in_eta)
            )
            eta_on_mu = (my_ESN.leak_factor) * eta_on_mu / my_ESN.dt
            if k == j:
                linewidth = 2
                linestyle = "-."
            else:
                linewidth = 6 - 2 * kk
                linestyle = linestyles[kk]
                kk += 1
            ax1.plot(
                eta,
                eta_on_mu,
                color=color[k - 1],
                linewidth=linewidth,
                linestyle=linestyle,
            )
        ax1.legend([f"mu_{k}" for k in np.arange(1, my_ESN.N_g + 1)])
        ax1.set_xlabel(f"eta_{j}")
        ax1.set_xticks(np.arange(-10, 15, 5))
        ax1.set_yticks(j * np.pi * np.arange(-10, 15, 5))
        ax1.set_yticklabels([f"{j}pi * {xval}" for xval in np.arange(-10, 15, 5)])
        ax1.grid()

        # plot effect of eta on eta
        ax2 = fig2.add_subplot(1, my_ESN.N_g, j)
        kk = 0
        for k in np.arange(1, my_ESN.N_g + 1):
            eta_on_eta = np.dot(
                my_ESN.W_out[eta_idx, eOutputVar[f"eta_{k}"]], np.tanh(w_in_eta)
            )
            eta_on_eta = (my_ESN.leak_factor) * eta_on_eta / my_ESN.dt
            if k == j:
                linewidth = 2
                linestyle = "-."
            else:
                linewidth = 6 - 2 * kk
                linestyle = linestyles[kk]
                kk += 1
            ax2.plot(
                eta,
                eta_on_eta,
                color=color[k - 1],
                linewidth=linewidth,
                linestyle=linestyle,
            )
        ax2.legend([f"eta_{k}" for k in np.arange(1, my_ESN.N_g + 1)])
        ax2.set_xlabel(f"eta_{j}")
        ax2.grid()

        # get the contribution of each mu mode
        mu_idx = np.where(my_ESN.W_in[:, eInputVar[f"mu_{j}"]].toarray() != 0)[0]
        w_in_mu = my_ESN.W_in[mu_idx, eInputVar[f"mu_{j}"]].toarray() * mu

        mu_on_mu = [None] * my_ESN.N_g
        mu_on_eta = [None] * my_ESN.N_g

        # plot effect of mu on mu
        ax3 = fig3.add_subplot(1, my_ESN.N_g, j)
        kk = 0
        for k in np.arange(1, my_ESN.N_g + 1):
            mu_on_mu = np.dot(
                my_ESN.W_out[mu_idx, eOutputVar[f"mu_{k}"]], np.tanh(w_in_mu)
            )
            mu_on_mu = (my_ESN.leak_factor) * mu_on_mu / my_ESN.dt
            if k == j:
                linewidth = 2
                linestyle = "-."
            else:
                linewidth = 6 - 2 * kk
                linestyle = linestyles[kk]
                kk += 1
            ax3.plot(
                mu,
                mu_on_mu,
                color=color[k - 1],
                linewidth=linewidth,
                linestyle=linestyle,
            )
        ax3.legend([f"mu_{k}" for k in np.arange(1, my_ESN.N_g + 1)])
        ax3.set_xlabel(f"mu_{j}")
        ax3.grid()

        # plot effect of mu on eta
        ax4 = fig4.add_subplot(1, my_ESN.N_g, j)
        kk = 0
        for k in np.arange(1, my_ESN.N_g + 1):
            mu_on_eta = np.dot(
                my_ESN.W_out[mu_idx, eOutputVar[f"eta_{k}"]], np.tanh(w_in_mu)
            )
            mu_on_eta = (my_ESN.leak_factor) * mu_on_eta / my_ESN.dt
            if k == j:
                linewidth = 2
                linestyle = "-."
            else:
                linewidth = 6 - 2 * kk
                linestyle = linestyles[kk]
                kk += 1
            ax4.plot(
                mu,
                mu_on_eta,
                color=color[k - 1],
                linewidth=linewidth,
                linestyle=linestyle,
            )
        ax4.legend([f"eta_{k}" for k in np.arange(1, my_ESN.N_g + 1)])
        ax4.set_xlabel(f"mu_{j}")
        ax4.set_xticks(np.arange(-10, 15, 5))
        ax4.set_yticks(j * np.pi * np.arange(-10, 15, 5))
        ax4.set_yticklabels([f"{j}pi * {xval}" for xval in np.arange(-10, 15, 5)])
        ax4.grid()

    return
