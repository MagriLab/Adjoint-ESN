from enum import IntEnum


class eParam(IntEnum):
    beta = 0
    tau = 1


def get_eVar(vars, N_g, N_c=10):
    if vars == "eta_mu":
        var_list = [f"eta_{i+1}" for i in range(N_g)]
        var_list.extend([f"mu_{i+1}" for i in range(N_g)])
    elif vars == "eta_mu_v":
        var_list = [f"eta_{i+1}" for i in range(N_g)]
        var_list.extend([f"mu_{i+1}" for i in range(N_g)])
        var_list.extend([f"v_{i+1}" for i in range(N_c)])
    elif vars == "eta_mu_v_tau":
        var_list = [f"eta_{i+1}" for i in range(N_g)]
        var_list.extend([f"mu_{i+1}" for i in range(N_g)])
        var_list.extend(["v_tau"])

    eVar = IntEnum("eVar", var_list, start=0)
    return eVar
