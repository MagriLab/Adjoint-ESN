import numpy as np

from adjoint_esn.rijke_galerkin.solver import Rijke


def check_jacobians(my_ESN, x_tau, x_0, x_1, p_0):
    # now compare this jacobian to the numerical one
    def closed_loop_step(x_tau, x_0, p_0):
        x_0_augmented = np.hstack((x_0, my_ESN.b_out))

        y_tau = np.dot(x_tau, my_ESN.W_out)
        eta_tau = y_tau[0 : my_ESN.N_g]
        velocity_f_tau = Rijke.toVelocity(
            N_g=my_ESN.N_g, eta=eta_tau, x=np.array([my_ESN.x_f])
        )
        y_0 = np.dot(x_0_augmented, my_ESN.W_out)
        y_0 = np.hstack((y_0, velocity_f_tau, velocity_f_tau**2))

        x_1 = my_ESN.step(x_0, y_0, p_0)
        return x_1

    def closed_loop_step_f(velocity_f_tau, x_0, p_0):
        x_0_augmented = np.hstack((x_0, my_ESN.b_out))
        y_0 = np.dot(x_0_augmented, my_ESN.W_out)
        y_0 = np.hstack((y_0, velocity_f_tau, velocity_f_tau**2))

        x_1 = my_ESN.step(x_0, y_0, p_0)
        return x_1

    h = 1e-5

    my_jac = my_ESN.jac(x_1, x_0)
    my_jac_num = np.zeros((my_ESN.N_reservoir, my_ESN.N_reservoir))
    for i in range(my_ESN.N_reservoir):
        x_0_left = x_0.copy()
        x_0_left[i] -= h
        x_0_right = x_0.copy()
        x_0_right[i] += h
        x_1_left = closed_loop_step(x_tau, x_0_left, p_0)
        x_1_right = closed_loop_step(x_tau, x_0_right, p_0)
        for j in range(my_ESN.N_reservoir):
            my_jac_num[j, i] = (x_1_right[j] - x_1_left[j]) / (2 * h)
    print(
        "Difference of analytical vs numerical Jacobian:",
        np.where(np.abs(my_jac_num - my_jac) > 1e-8),
    )

    my_drdp = my_ESN.drdp(x_1, x_0)
    p_0_left = p_0 - h
    p_0_right = p_0 + h
    x_1_left = closed_loop_step(x_tau, x_0, p_0_left)
    x_1_right = closed_loop_step(x_tau, x_0, p_0_right)
    my_drdp_num = (x_1_right - x_1_left) / (2 * h)
    print(
        "Difference of analytical vs numerical dr/dp:",
        np.where(np.abs(my_drdp_num[:, None] - my_drdp.toarray()) > 1e-8),
    )

    y_tau = np.dot(x_tau, my_ESN.W_out)
    eta_tau = y_tau[0 : my_ESN.N_g]
    u_f_tau_0 = Rijke.toVelocity(N_g=my_ESN.N_g, eta=eta_tau, x=np.array([my_ESN.x_f]))
    my_drdu_f_tau = my_ESN.drdu_f_tau(x_1, x_0, u_f_tau_0)
    u_f_tau_0_left = u_f_tau_0 - h
    u_f_tau_0_right = u_f_tau_0 + h
    x_1_left = closed_loop_step_f(u_f_tau_0_left, x_0, p_0)
    x_1_right = closed_loop_step_f(u_f_tau_0_right, x_0, p_0)
    my_drdu_f_tau_num = (x_1_right - x_1_left) / (2 * h)
    print(
        "Difference of analytical vs numerical dr/du_f_tau:",
        np.where(np.abs(my_drdu_f_tau_num[:, None] - my_drdu_f_tau) > 1e-8),
    )

    my_jac_tau = my_ESN.jac_tau(x_1, x_0, u_f_tau_0)
    my_jac_tau_num = np.zeros((my_ESN.N_reservoir, my_ESN.N_reservoir))
    for i in range(my_ESN.N_reservoir):
        x_tau_left = x_tau.copy()
        x_tau_left[i] -= h
        x_tau_right = x_tau.copy()
        x_tau_right[i] += h
        x_1_left = closed_loop_step(x_tau_left, x_0, p_0)
        x_1_right = closed_loop_step(x_tau_right, x_0, p_0)
        for j in range(my_ESN.N_reservoir):
            my_jac_tau_num[j, i] = (x_1_right[j] - x_1_left[j]) / (2 * h)
    print(
        "Difference of analytical vs numerical Jacobian tau:",
        np.where(np.abs(my_jac_tau_num - my_jac_tau) > 1e-8),
    )
    return
