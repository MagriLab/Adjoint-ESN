import argparse
import os
import sys
from pathlib import Path

import numpy as np

import wandb

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from adjoint_esn.utils.preprocessing import unpickle_file


def create_wandb_config(
    e_idx,
    hyp_param_names,
    min_dict,
    data_config,
    ESN_dict,
    p_train_val_list,
    train_idx_list,
    val_idx_list,
    noise_std,
    input_var,
    p_var,
    save_path,
):
    hyp_param_dict = {}
    for hyp_param_name in set(hyp_param_names):
        # get the unique strings in the list with set
        # now the indices of the parameters with that name
        # (because ESN has attributes that are set as arrays and not single scalars)
        hyp_param_idx_list = np.where(np.array(hyp_param_names) == hyp_param_name)[0]
        new_hyp_param = np.zeros(len(hyp_param_idx_list))
        for new_hyp_idx in range(len(hyp_param_idx_list)):
            hyp_param_idx = hyp_param_idx_list[new_hyp_idx]
            new_hyp_param[new_hyp_idx] = min_dict["params"][e_idx, hyp_param_idx]
        if len(hyp_param_idx_list) == 1:
            new_hyp_param = new_hyp_param[0]
        hyp_param_dict[hyp_param_name] = new_hyp_param

    cfg_dict = {
        "dt": data_config["dt"],
        "t_washout_len": data_config["t_washout_len"],
        "t_train_len": data_config["t_train_len"],
        "input_var": input_var,
        "p_var": p_var,
        "p_train": p_train_val_list[train_idx_list],
        "p_val": p_train_val_list[val_idx_list],
        **ESN_dict,
        **hyp_param_dict,
        "tikh": min_dict["tikh"][e_idx],
        "input_seeds": min_dict["input_seeds"][e_idx],
        "reservoir_seeds": min_dict["reservoir_seeds"][e_idx],
        "noise_std": noise_std,
        "run_name": str(save_path),
    }
    return cfg_dict


def main(args):
    hyp_folder_path = Path(args.hyp_folder_name)

    # iterate over the files in the folder
    for hyp_file_name in hyp_folder_path.iterdir():
        print(hyp_file_name)
        hyp_results, hyp_file = unpickle_file(hyp_file_name)
        n_ensemble = len(hyp_results["min_dict"]["f"])
        hyp_results_subset = {
            k: hyp_results[k]
            for k in (
                "data_config",
                "p_train_val_list",
                "train_idx_list",
                "val_idx_list",
                "hyp_param_names",
                "ESN_dict",
                "noise_std",
                "min_dict",
            )
        }
        # find out which variables were used for training in order to recreate the dataset
        if "input_var" in hyp_results["data_config"]:
            input_var = hyp_results["data_config"]["input_var"]
        elif (
            "train_var" in hyp_results["data_config"]
        ):  # old validation logs will have this tag
            # adding this bit so we can still handle the old logs
            if hyp_results["data_config"]["train_var"] == "gal":
                # Assumes we used 10 Galerkin modes!!
                if hyp_results["ESN_dict"]["dimension"] == 20:
                    input_var = "eta_mu"
                elif hyp_results["ESN_dict"]["dimension"] == 30:
                    input_var = "eta_mu_v"
            else:
                input_var = hyp_results["data_config"]["train_var"]

        # find out which parameter the training and validation was done for
        # @todo: enums for beta, tau
        len_beta_set = len(set(hyp_results["p_train_val_list"][:, 0]))
        len_tau_set = len(set(hyp_results["p_train_val_list"][:, 1]))
        if len_beta_set > 1 and len_tau_set > 1:
            p_var = "beta-tau"
        elif len_beta_set > 1 and len_tau_set == 1:
            p_var = "beta"
        elif len_beta_set == 1 and len_tau_set > 1:
            p_var = "tau"

        for e_idx in range(n_ensemble):
            cfg_dict = create_wandb_config(
                e_idx=e_idx,
                save_path=hyp_file_name.stem,
                **hyp_results_subset,
                input_var=input_var,
                p_var=p_var
            )
            my_wandb_run = wandb.init(
                config=cfg_dict,
                entity=args.wandb_entity,
                project=args.wandb_project,
                group=p_var,
                reinit=True,
                mode="online",
            )
            my_wandb_run.log(
                {
                    "val_score": hyp_results["min_dict"]["f"][e_idx],
                }
            )
            my_wandb_run.finish()

        hyp_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uploads results to wandb")
    parser.add_argument(
        "--hyp_folder_name", type=str, default="src/results/val_runs/new"
    )
    # arguments for weights and biases
    parser.add_argument("--wandb-entity", default="defneozan", type=str)
    parser.add_argument("--wandb-project", default="adjoint-esn", type=str)
    parsed_args = parser.parse_args()
    main(parsed_args)
