import numpy as np
from scipy.optimize import fmin_bfgs


def fixed_point(my_ESN, beta):
    j = np.arange(1, my_ESN.N_g + 1)
    modes = np.cos(j * np.pi * my_ESN.x_f)
    modes = np.hstack((modes, np.zeros(my_ESN.N_g)))

    def f(u_star):
        # find u_f(t-tau) = u_f(t)
        u_f = np.dot(modes, u_star)
        u_augmented = np.hstack((u_star, u_f))

        # augment with the parameter
        beta_norm = (
            beta - my_ESN.parameter_normalization_mean
        ) / my_ESN.parameter_normalization_var
        u_augmented = np.hstack((u_augmented, beta_norm))

        # step
        x_tilde = np.tanh(my_ESN.W_in.dot(u_augmented))

        # output
        next_u_star = np.dot(x_tilde, my_ESN.W_out)

        error = np.linalg.norm(u_star - next_u_star)

        return error

    u0 = 0.3 * np.ones(2 * my_ESN.N_g)
    opt_u = fmin_bfgs(f, u0)
    return opt_u
