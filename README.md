# MLRC: Competition of Mechanisms
This repository contains the official code for reproducibility of the [Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals
](https://arxiv.org/abs/2402.11655) paper.
The datasets used are also available on [HuggingFace](https://huggingface.co/datasets/francescortu/comp-mech).


<p align="center">
    <img src="comp.png" width="700">
</p>


## Installation

To set up the environment for the `FACT_project`, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd FACT_project
   ```
2. **Set Up the Conda Environment:**
    - For CPU:
        bash```conda env create -f environment_cpu.yaml```
   - For GPU:
        bash```conda env create -f environment_gpu.yaml```
2. **Activate the Environment:**
   - ```conda activate <environment-name>```

## Running the Experiments
 - LogitLens
 - Logit Attribution
 - Attention Pattern

You can run the experiments running the `notebooks/experiments.ipynb` notebook. This notebook contains the code to run the experiments for the logit lens, logit attribution, and attention pattern.

#### Script to execute all the experiments
You can run the experiment running the following command:
```bash
cd scripts
python run_all.py
```
with the following arguments:
- `--model_name`: The name of the model to run the experiments on. It can be `gpt2` or `pythia-6.9b`.
- `--batch`: The batch size to use for the experiments. (Suggested: 40 for gpt2, 10 for pythia)
- `--experiment`: The experiment to run. Example: `copyVSfact`.
- `--folder`: The experiment folder to use.
- `--dataset`: The dataset to run the experiments on (`data/`).
- `--start`: Specifies the starting point in the dataset for running the experiment. 
- `--end`: Specifies the ending index in the dataset for running the experiment.
- `--prompt_type`: Prompt type for testing different prompt structures `["qna", "fact_check_v1", "fact_check_v2", "context_qna"]`. 
- `--logit-attribution`: If you want to run the logit attribution experiment.
- `--logit-lens`: If you want to run the logit lens experiment (Figure 2).
- `--ov-diff`: This is useful for comparing model outputs in terms of performance or behavior.
- `--all`: Run all the experiments at once sequentially.
- `--ablate`: If you want to perform ablation.
- `--ablate-component`: The specific component to ablate, default is "all".
- `--pattern`: If you want to retrieve the attention pattern.
- `--device`: Specify the device for execution (GPU or CPU).
- `--only-plot`: If you want to only generate plots.
- `--flag`: An additional flag for custom behavior.
  
The script will create a folder in the `Results/copyVSfact` directory with the name of the model.

Example:
```bash
cd Script
python run_all.py --model-name gpt2 --batch 40 --experiment copyVSfact --logit-attribution 
```

---

### Project Structure

This section highlights the key files and directories in this repository to help you navigate the project effectively.


- `run.sh`: Shell script for running the main processes.
- `environment_cpu.yaml` / `environment_gpu.yaml`: Conda environment configurations for CPU and GPU setups.
- `data/`: Contains various datasets used in the project.
- `notebooks/`: Jupyter notebooks for analysis and experimentation.
- `plotting_scripts/`: Python scripts (converted from R) for generating visualizations.
  - `plot_head_pattern_fig_4b_5.py`: Plot for head pattern visualizations (Figures 4b and 5).
  - `plot_logit_attribution_fig_3_4a.py`: Logit attribution visualization for Figures 3 and 4a.
  - `plot_logit_lens_fig_2.py`: Logit lens analysis plot for Figure 2.
- `results/`: Output directory for storing results and plots.
- `scripts/`: Core processing and analysis scripts.
- `src/`: Source code directory containing core functionalities.
- `src_figure/`: R Scripts for generating research paper figures.
---

### **Source Code (`src`)**
The `src/` directory contains the core codebase for the project. It includes the main modules, utilities, and experiment-specific code.
- `base_experiment.py`: Contains helper functions for running experiments.
- `dataset.py`: Handles the dataset-related operations like loading and preprocessing.
- `model.py`: Defines the model architecture and related components.
- `utils.py`: Utility functions used across various modules.
- `experiment/`: Contains individual experiment modules:
  - `ablation.py`: Code for conducting and analyzing ablation studies.
  - `ablator.py`: Contains logic for performing ablation tasks.
  - `head_pattern.py`: Module for analyzing and visualizing attention head patterns.
  - `logit_attribution.py`: Handles logit attribution analysis for model outputs.
  - `logit_lens.py`: Code for logit lens attribution analysis.
  - `ov.py`: Contains functions for analyzing model differences and outputs.
---

### **Notebooks**
Jupyter notebooks for experiments and analysis:
- `notebooks/experiments.ipynb`: Key experiments notebook.
- `notebooks/attention_modifcation.ipynb`: Notebook for attention modifications experiments.
- `notebooks/root`: Experimentation with different data and new approaches.
---

### **Plotting Scripts**
Scripts for generating visualizations:

- `plotting_scripts/plot_ablation.py`: Ablation study plots.
- `plotting_scripts/plot_logit_lens_fig_2.py`: Logit lens analysis plot.
- `plotting_scripts/plot_head_pattern_fig_4b_5.py`: Head pattern visualization.
---

### **Authors**
The following individuals contributed to the development of this project:

- **Asen Dotsinski**
- **Hafeez Khan**
- **Marko Ivanov**
- **Udit Thakur**
