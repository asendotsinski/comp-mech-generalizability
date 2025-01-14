import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from setuptools.command.rotate import rotate

# Set working directory
os.chdir("../results")
SAVE_DIR_NAME = "python_paper_plots"
print(os.getcwd())

# Configurations
palette = {
    "GPT2": "#003f5c",
    "GPT2-medium": "#58508d",
    "GPT2-large": "#bc5090",
    "GPT2-xl": "#ff6361",
    "Pythia-6.9b": "#ffa600"
}
FACTUAL_COLOR = "#005CAB"
FACTUAL_CMAP = sns.diverging_palette(10, 250, as_cmap=True)
COUNTERFACTUAL_COLOR = "#E31B23"
COUNTERFACTUAL_CMAP = sns.diverging_palette(300, 10, as_cmap=True)

model = "gpt2"
model_folder = "gpt2_full"
n_layers = 12
experiment = "copyVSfact"
n_positions = 12
positions_name = [
    "-", "Subject", "2nd Subject", "3rd Subject", "Relation",
    "Relation Last", "Attribute*", "-", "Subject Repeat",
    "2nd Subject repeat", "3rd Subject repeat", "Relation repeat", "Last"
]
relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*",
                     "Subject repeat", "Relation repeat", "Last"]
example_position = ["iPhone", "was developed", "by", "Google.",
                    "iPhone", "was developed", "by"]
n_relevant_position = 7

AXIS_TITLE_SIZE = 20
AXIS_TEXT_SIZE = 16
HEATMAP_SIZE = 10

directory_path = f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Define helper function for heatmap
def create_heatmap(data, x, y, fill, cmap, midpoint=0,
                   text=False, xlabel=None, ylabel=None,
                   ax=None, colorbar_label=None):
    pivot_data = data.pivot(index=y, columns=x, values=fill)
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

    # Determine min, max, and ticks for colorbar
    vmin, vmax = int(data[fill].min()), int(data[fill].max())
    ticks = np.arange(vmin, vmax, 5)

    # Configure colorbar settings
    cbar_kws = {
        'label': xlabel,
        'orientation': 'horizontal',
        'location': 'bottom',
        'pad': 0.25,
        'ticks': ticks
    }

    heatmap = sns.heatmap(
        pivot_data, cmap=cmap, annot=text, fmt=".2f", center=midpoint,
        linewidths=0.5, linecolor="grey", cbar_kws=cbar_kws,
        yticklabels=relevant_position, ax=ax
    )
    heatmap.figure.axes[-1].xaxis.label.set_size(AXIS_TEXT_SIZE)
    heatmap.figure.axes[-1].axes.tick_params(labelsize=AXIS_TEXT_SIZE - 2)
    ax.set_xlabel(xlabel, fontsize=AXIS_TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_TITLE_SIZE)
    ax.tick_params(axis='x', labelsize=AXIS_TEXT_SIZE)
    ax.tick_params(axis='y', labelsize=AXIS_TEXT_SIZE)
    ax.tick_params(left=False, bottom=False)

    # Set colorbar label
    if colorbar_label:
        cbar = heatmap.collections[0].colorbar
        cbar.set_label(colorbar_label)

    # Create a twin axis that shares the same x-axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(example_position, rotation=0)
    ax2.tick_params(axis='y', labelsize=AXIS_TEXT_SIZE)
    ax2.tick_params(left=False, bottom=False)
    ax2.grid(False)


# Load data
data = pd.read_csv(f"{experiment}/logit_lens/{model_folder}/logit_lens_data.csv")

data_resid_post = data[data['component'].str.contains("resid_post")]
# print(data['position'].unique())

# data_resid_post['position_name'] = data_resid_post['position'].map(
#     lambda x: positions_name[int(x) + 1]
# )
# data_resid_post = data_resid_post[data_resid_post['position'].isin([1, 4, 5, 6, 8, 11, 12])]
data_resid_post = data_resid_post[data_resid_post['position'].isin([1, 4, 5, 6, 8, 12, 13])]
unique_positions = data_resid_post['position'].unique()
position_mapping = {pos: i for i, pos in enumerate(unique_positions)}
data_resid_post['mapped_position'] = data_resid_post['position'].map(position_mapping)

#########################################
########## Line: Logit Plots ############
#########################################

# Line plot for logit
data_resid_post_last = data_resid_post[data_resid_post['position'] == 13]
# data_resid_post_last.loc[:, "mem"] = data_resid_post_last["mem"].astype(int)
# data_resid_post_last.loc[:, "cp"] = data_resid_post_last["cp"].astype(int)
# print(data_resid_post_last['mem'])
plt.figure(figsize=(12, 8))
plt.plot(data_resid_post_last['layer'], data_resid_post_last['mem'],
         label='Factual Token', color=FACTUAL_COLOR, linewidth=3, marker='o', markersize=6)
plt.plot(data_resid_post_last['layer'], data_resid_post_last['cp'],
         label='Counterfactual Token', color=COUNTERFACTUAL_COLOR, linewidth=3, marker='o', markersize=6)

# Adding grid with dotted lines in the background
plt.grid(True, linestyle=':', color='gray', linewidth=0.5)

plt.xlabel("Layer", fontsize=AXIS_TITLE_SIZE)
plt.ylabel("Logit in the Last Position", fontsize=AXIS_TITLE_SIZE)
plt.xticks(fontsize=AXIS_TEXT_SIZE)
y_max = max(int(data_resid_post_last['mem'].max()), int(data_resid_post_last['cp'].max()))
y_ticks = np.arange(0, y_max + 1, 5)
plt.yticks(y_ticks, fontsize=AXIS_TEXT_SIZE)
plt.legend(fontsize=AXIS_TEXT_SIZE)

# Save the figure
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream/resid_post_all_linelogit_line.pdf",
            bbox_inches='tight')

#########################################
########### Line: Index Plots ###########
#########################################

# Line plot for logit index
data_resid_post_altered = data_resid_post[data_resid_post['position'] == 6]
data_resid_post_2_subject = data_resid_post[data_resid_post['position'] == 8]
data_resid_post_last = data_resid_post[data_resid_post['position'] == 13]
p_idx = plt.figure(figsize=(12, 8))
plt.plot(data_resid_post_altered['layer'], data_resid_post_altered['mem_idx'],
         label='Factual Token', color=FACTUAL_COLOR, linewidth=3, marker='o', markersize=6)
plt.plot(data_resid_post_altered['layer'], data_resid_post_altered['cp_idx'],
         label='Counterfactual Token', color=COUNTERFACTUAL_COLOR, linewidth=3, marker='o', markersize=6)
# Adding grid with dotted lines in the background
plt.grid(True, linestyle=':', color='gray', linewidth=0.5)

plt.xlabel("Layer", fontsize=AXIS_TITLE_SIZE)
plt.ylabel("Rank (Logit)", fontsize=AXIS_TITLE_SIZE)
plt.xticks(fontsize=AXIS_TEXT_SIZE)
plt.yticks(fontsize=AXIS_TEXT_SIZE)
plt.legend(fontsize=AXIS_TEXT_SIZE)
plt.yscale('log')
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream/resid_post_index.pdf", bbox_inches='tight')


#########################################
######### Heatmap: Logit Plots ##########
#########################################

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

create_heatmap(
    data=data_resid_post, x="layer", y="mapped_position", fill="mem",
    cmap=FACTUAL_CMAP, midpoint=0,
    xlabel="Layer", colorbar_label="Logit of Factual",
    ax=axes[0]
)
create_heatmap(
    data=data_resid_post, x="layer", y="mapped_position", fill="cp",
    cmap=COUNTERFACTUAL_CMAP, midpoint=0,
    xlabel="Layer", colorbar_label="Logit of Counterfactual",
    ax=axes[1]
)
# Adjust the layout with some vertical spacing between subplots
fig.subplots_adjust(hspace=0.25)
# plt.tight_layout()
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream/resid_post_all_linelogit.pdf",
            bbox_inches='tight')

#########################################
###### Heatmap: MEM and CP Plots ########
#########################################

create_heatmap(
    data=data_resid_post, x="layer", y="mapped_position", fill="mem",
    cmap=FACTUAL_CMAP, midpoint=0,
    xlabel="Layer", colorbar_label="Logit of Factual",
)
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream/resid_post_mem.pdf", bbox_inches='tight')

create_heatmap(
    data=data_resid_post, x="layer", y="mapped_position", fill="cp",
    cmap=COUNTERFACTUAL_CMAP, midpoint=0,
    xlabel="Layer", colorbar_label="Logit of Counterfactual",
)
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream/resid_post_cp.pdf", bbox_inches='tight')

# Print success message
print(f"Plots saved for {model} in {experiment}")


#########################################
############ Combined Plots (Optional) #############
#########################################

# Create figure
fig = plt.figure(figsize=(20, 8))
# Create GridSpec layout
gs = fig.add_gridspec(2, 2)
# Create the subplots with specific grid positions
ax1 = fig.add_subplot(gs[0, 0])  # First heatmap: top-left
ax2 = fig.add_subplot(gs[1, 0])  # Second heatmap: bottom-left
ax3 = fig.add_subplot(gs[:, 1])  # Line plot: entire right column

# First heatmap (Factual)
create_heatmap(
    data=data_resid_post, x="layer", y="mapped_position", fill="mem",
    cmap=FACTUAL_CMAP, midpoint=0,
    xlabel="Layer", colorbar_label="Logit of Factual",
    ax=ax1
)
# Second heatmap (Counterfactual)
create_heatmap(
    data=data_resid_post, x="layer", y="mapped_position", fill="cp",
    cmap=COUNTERFACTUAL_CMAP, midpoint=0,
    xlabel="Layer", colorbar_label="Logit of Counterfactual",
    ax=ax2
)

# Line plot last (entire right side)
ax3.plot(data_resid_post_last['layer'], data_resid_post_last['mem'],
         label='Factual Token', color=FACTUAL_COLOR, linewidth=3,
         marker='o', markersize=6)
ax3.plot(data_resid_post_last['layer'], data_resid_post_last['cp'],
         label='Counterfactual Token', color=COUNTERFACTUAL_COLOR,
         linewidth=3, marker='o', markersize=6)
# Customize the line plot
ax3.grid(True, linestyle=':', color='gray', linewidth=0.5)
ax3.set_xlabel("Layer", fontsize=AXIS_TITLE_SIZE)
ax3.set_ylabel("Logit in the Last Position", fontsize=AXIS_TITLE_SIZE)
ax3.set_yticks(ticks=y_ticks)
# ax3.tick_params(axis='both', labelsize=AXIS_TEXT_SIZE)
ax3.tick_params(left=False, bottom=False)
ax3.legend(fontsize=AXIS_TEXT_SIZE)
plt.subplots_adjust(wspace=0.5, hspace=0.3)
# Save the figure
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream/resid_post_all_logit_combined.pdf",
            bbox_inches='tight')

# Line plot logit index (entire right side)
ax3 = fig.add_subplot(gs[:, 1])  # Line plot: entire right column
ax3.plot(data_resid_post_altered['layer'], data_resid_post_altered['mem_idx'],
         label='Factual Token', color=FACTUAL_COLOR, linewidth=3, marker='o', markersize=6)
ax3.plot(data_resid_post_altered['layer'], data_resid_post_altered['cp_idx'],
         label='Counterfactual Token', color=COUNTERFACTUAL_COLOR, linewidth=3, marker='o', markersize=6)
# Adding grid with dotted lines in the background
ax3.grid(True, linestyle=':', color='gray', linewidth=0.5)
ax3.set_xlabel("Layer", fontsize=AXIS_TITLE_SIZE)
ax3.set_ylabel("Rank (Logit)", fontsize=AXIS_TITLE_SIZE)
ax3.set_yscale('log')
# ax3.tick_params(axis='both', labelsize=AXIS_TEXT_SIZE)
ax3.tick_params(left=False, bottom=False)
ax3.legend(fontsize=AXIS_TEXT_SIZE)
plt.subplots_adjust(wspace=0.5, hspace=0.3)

# Save the figure
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_residual_stream/resid_post_all_index_combined.pdf",
            bbox_inches='tight')
plt.close()