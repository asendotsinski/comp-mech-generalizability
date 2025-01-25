# imports
import os
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
from dataset import DOMAINS

# Define constants for default values

# domain name
DOMAIN = "Science"

# models
MODEL = "gpt2"
# MODEL = "pythia-6.9b"

# model folder
MODEL_FOLDER = f"{MODEL}_full"

# experiment name
# EXPERIMENT = "copyVSfact"
# EXPERIMENT = "copyVSfactQnA"
EXPERIMENT = "copyVSfactDomain"

# if EXPERIMENT == "copyVSfactDomain":
#     MODEL_FOLDER += f"_{DOMAIN}"

scripts = ["plot_logit_lens_fig_2.py",
           "plot_logit_attribution_fig_3_4a.py",
           "plot_head_pattern_fig_4b_5.py"]

def plot_results():
    for script in scripts:
        # for domain in DOMAINS:
        command = [
            "python",
            f"../plotting_scripts/{script}",
            MODEL,
            EXPERIMENT,
            MODEL_FOLDER,
            DOMAIN
            # f"{MODEL_FOLDER}_{domain}",
            # domain
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)


if __name__ == "__main__":
    plot_results()
