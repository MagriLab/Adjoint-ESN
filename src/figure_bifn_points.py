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
from adjoint_esn.utils import preprocessing as pp
from adjoint_esn.utils import visualizations as vis

model_path = Path("local_results/rijke/run_20231029_153121")

save_path = "20231109_181059"

save_fig = False

titles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

fig = plt.figure(figsize=(5, 5), constrained_layout=True)

# print model properties
config = post.load_config(model_path)
results = pp.unpickle_file(model_path / "results.pickle")[0]
if "top_idx" in results.keys():
    top_idx = results["top_idx"]
else:
    top_idx = 0
(
    ESN_dict,
    hyp_param_names,
    hyp_param_scales,
    hyp_params,
) = post.get_ESN_properties_from_results(config, results, dim=1, top_idx=top_idx)
print(ESN_dict)
[
    print(f"{hyp_param_name}: {hyp_param}")
    for hyp_param_name, hyp_param in zip(hyp_param_names, hyp_params)
]

bifn_point_results = pp.unpickle_file(
    model_path / f"bifn_point_results_{save_path}.pickle"
)[0]

bifn_point_true = bifn_point_results["bifn_points"]["true"]
bifn_point_pred = np.mean(bifn_point_results["bifn_points"]["esn"], axis=0)
bifn_point_pred = bifn_point_results["bifn_points"]["esn"][1]

tau_list = bifn_point_results["tau_list"]

linestyle = ["-"]
linestyle.extend(["--"] * 10)
vis.plot_reverse_lines(
    bifn_point_true,
    bifn_point_pred,
    y=tau_list,
    linestyle=linestyle,
    xlabel="beta",
    ylabel="tau",
)
plt.show()
