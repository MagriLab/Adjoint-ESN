import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

import wandb

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from adjoint_esn.utils.preprocessing import unpickle_file


def load_config(experiment_path):
    with open(experiment_path / "config.yml", "r") as file:
        config = yaml.unsafe_load(file)
        return config


def main(args):
    hyp_folder_path = Path(args.hyp_folder_name)

    # iterate over the files in the folder
    for hyp_file_name in hyp_folder_path.iterdir():
        if "20240204" in hyp_file_name.as_posix():
            print(hyp_file_name)
            config = load_config(hyp_file_name).to_dict()
            try:
                hyp_results, hyp_file = unpickle_file(hyp_file_name / "results.pickle")
            except:
                print("Results file doesn't exist.")
                continue
            config["path"] = hyp_file_name.as_posix()
            # config["path"] = "local_results/standard/" + hyp_file_name.stem
            for key in hyp_results.keys():
                if key != "f":
                    config[key] = hyp_results[key][0]
            my_wandb_run = wandb.init(
                config=config,
                entity=args.wandb_entity,
                project=args.wandb_project,
                reinit=True,
                mode="online",
            )
            out_path = hyp_file_name / "out.txt"
            # artifact = wandb.Artifact(name='out', type='log')
            # artifact.add_file(local_path = out_path.as_posix())
            # my_wandb_run.log_artifact(artifact)
            with open(out_path) as f:
                print(f.read())
            my_wandb_run.log(
                {
                    "val_score": hyp_results["f"][0],
                }
            )

            my_wandb_run.finish()

            hyp_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uploads results to wandb")
    parser.add_argument("--hyp_folder_name", type=str, default="local_results/lorenz63")
    # arguments for weights and biases
    parser.add_argument("--wandb-entity", default="defneozan", type=str)
    parser.add_argument("--wandb-project", default="adjoint-esn-lorenz63", type=str)
    parsed_args = parser.parse_args()
    main(parsed_args)
