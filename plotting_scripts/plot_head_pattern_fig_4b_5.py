import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set working directory
os.chdir("../results")
print("Current working directory:", os.getcwd())
SAVE_DIR_NAME = "python_paper_plots"

# Set color palette
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

# GPT2 Small Configurations
model = "gpt2"
model_folder = "gpt2_full"
n_layers = 12

# Pythia
# model = "pythia-6.9b"
# model_folder = "pythia-6.9b_full"
# n_layers = 32

experiment = "copyVSfact"
n_positions = 12
positions_name = ["-", "Subject", "2nd Subject", "3rd Subject", "Relation", "Relation Last", "Attribute*", "-",
                  "Subject Repeat", "2nd Subject repeat", "3nd Subject repeat", "Relation repeat", "Last"]
relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last"]
n_relevant_position = 7

if model == "gpt2":
    layer_pattern = [11, 10, 10, 10, 9, 9]
    head_pattern = [10, 0, 7, 10, 6, 9]
    # subject and others positions
    source_position = 13
    source_positions = [1, 4, 5, 6, 8, 12, 13]
    # source_positions = [2, 4, 5, 6, 8, 12, 13]
    # source_position = 12
    # dest_positions = [1, 4, 5, 6, 8, 11, 12]
    dest_positions = [1, 4, 5, 6, 8, 12, 13]
    # dest_positions = [2, 4, 5, 6, 8, 12, 13]
    factual_heads_layer = [11, 10]
    factual_heads_head = [10, 7]
else:
    layer_pattern = [10, 10, 15, 17, 17, 19, 19, 20, 20, 21, 23]
    head_pattern = [1, 27, 17, 14, 28, 20, 31, 2, 18, 8, 25]
    factual_heads_layer = [21, 20, 17, 10]
    factual_heads_head = [8, 18, 28, 27]

    source_position = 13
    source_positions = [1, 4, 5, 6, 8, 12, 13]
    dest_positions = [1, 4, 5, 6, 8, 12, 13]


AXIS_TITLE_SIZE = 20
AXIS_TEXT_SIZE = 15
HEATMAP_SIZE = 10


directory_path = f"{SAVE_DIR_NAME}/{model}_{experiment}_heads_pattern"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Load data
data = pd.read_csv(f"{experiment}/head_pattern/{model_folder}/head_pattern_data.csv")

#########################################
#### Heatmap: Layer-Head Combination ####
#########################################

# Create a heatmap function
def create_heatmap(data, midpoint=0):
    plt.figure(figsize=(10, 8))

    # Create the pivot table for all data
    data_pivot = data.pivot(index="y_label", columns="mapped_position", values="value")

    # Create masks for Target and Other
    mask_target = ~(data.pivot(index="y_label", columns="mapped_position", values="color") == "Target")
    mask_other = ~(data.pivot(index="y_label", columns="mapped_position", values="color") == "Other")

    # Plot both heatmaps with their respective masks
    ax1 = sns.heatmap(data_pivot,
                mask=mask_target,
                cmap=FACTUAL_CMAP,
                linewidths=0.5,
                center=midpoint,
                cbar_kws={"label": "Attention Score Factual", "pad": 0.1},
                xticklabels=relevant_position,
                yticklabels=True)

    ax2 = sns.heatmap(data_pivot,
                mask=mask_other,
                cmap=COUNTERFACTUAL_CMAP,
                linewidths=0.5,
                center=midpoint,
                cbar_kws={"label": "Attention Score Counterfactual", "pad": 0.1},
                xticklabels=relevant_position,
                yticklabels=True)

    # Rest of your formatting code
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontsize=AXIS_TEXT_SIZE, rotation=70)
    plt.yticks(fontsize=AXIS_TEXT_SIZE, rotation=0)

    # Center xticks by adjusting alignment
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xticklabels(ax.get_xticklabels(),
                       va='bottom',
                       # ha='center',
                       rotation_mode='anchor',
                       position=(0, -0.15)
                       )
    # ax.tick_params(left=False, bottom=False)
    plt.tight_layout()

    # Adjust colorbar label sizes after plotting
    for cbar in ax.figure.axes[-2:]:
        cbar.yaxis.label.set_size(AXIS_TEXT_SIZE-2)
    ax.tick_params(left=False, bottom=False)

# Filter and join data
data_filtered = data[data['source_position'] == source_position]  # last position
pattern_df = pd.DataFrame({"layer": layer_pattern, "head": head_pattern})
data_final = pd.merge(data_filtered, pattern_df, on=["layer", "head"])
# Prepare labels
data_final["y_label"] = "Layer " + data_final["layer"].astype(str) + " | Head " + data_final["head"].astype(str)

# Mapping positions
data_final = data_final[data_final["dest_position"].isin(dest_positions)]
unique_positions = data_final['dest_position'].unique()
position_mapping = {position: idx for idx, position in enumerate(unique_positions)}
data_final["mapped_position"] = data_final['dest_position'].map(position_mapping)

# Create the heatmap for target and other colors
data_final["color"] = np.where(data_final['y_label'].isin(
    [f"Layer {layer} | Head {head}" for layer, head in zip(factual_heads_layer, factual_heads_head)]), 'Target',
                               'Other')
# data_final.sort_values(by=["layer", "head"], ascending=False, axis=0, inplace=True)
# print(data_final)
create_heatmap(data_final)
# Save plot
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_heads_pattern/head_pattern_layer.pdf",
            bbox_inches='tight')


#########################################
#### Heatmap(s): Layer-Head Combination ####
#########################################

# Full position heatmap function
def create_multiple_heatmaps(data, x, y, fill, title, cmap, ax):
    sns.heatmap(data.pivot_table(index=y, columns=x, values=fill),
                cmap=cmap, center=0, yticklabels=relevant_position,
                xticklabels=relevant_position,
                cbar_kws={'label': 'Attention Score'}, annot=False, fmt='.2f', ax=ax)
    ax.set_title(title, fontsize=AXIS_TEXT_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False)
    ax.tick_params('x', rotation=70)
    ax.set_xticklabels(ax.get_xticklabels(),
                       va='bottom',
                       # ha='center',
                       rotation_mode='anchor',
                       position=(0, -0.2)
                       )


# Full position plot
if model == "gpt2":
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))  # 2 columns, 3 rows
else:
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 12))  # 2 columns, 3 rows
axes = axes.flatten()  # Flatten the axes array for easy indexing

plot = None
for i in range(len(head_pattern)):
    head = head_pattern[i]
    layer = layer_pattern[i]

    # filter positions
    pattern_df = pd.DataFrame({"layer": layer_pattern, "head": head_pattern})
    data_filtered = pd.merge(data, pattern_df, on=["layer", "head"])
    # prepare labels
    data_filtered["y_label"] = "Layer " + data_filtered["layer"].astype(str) + " | Head " + data_filtered["head"].astype(str)

    data_head = data_filtered[(data_filtered["head"] == head) & (data_filtered["layer"] == layer)]
    data_head = data_head[data_head['dest_position'].isin(dest_positions)]
    data_head = data_head[data_head['source_position'].isin(source_positions)]

    # Map positions
    dest_unique_positions = data_head['dest_position'].unique()
    source_unique_positions = data_head['source_position'].unique()

    dest_position_mapping = {position: idx for idx, position in enumerate(dest_unique_positions)}
    source_position_mapping = {position: idx for idx, position in enumerate(dest_unique_positions)}

    data_head['dest_mapped'] = data_head['dest_position'].map(dest_position_mapping)
    data_head['source_mapped'] = data_head['source_position'].map(source_position_mapping)

    cmap = FACTUAL_CMAP if (layer in factual_heads_layer and head in factual_heads_head and head!=layer) else COUNTERFACTUAL_CMAP
    plot = create_multiple_heatmaps(data_head, 'dest_mapped', 'source_mapped',
                                'value', f'Layer {layer} Head {head}', cmap, axes[i])
# Adjust layout with custom spacing
if model == "gpt2":
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
else:
    plt.subplots_adjust(wspace=0.8, hspace=0.8)

# Saving plot for full position
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_heads_pattern/full_pattern.pdf",
            bbox_inches='tight')


#########################################
###### Bar Plot: Boosting Heads #########
#########################################

# Boosting Heads Bar Plot
data_long = pd.DataFrame({
    'model': ['GPT2', 'GPT2', 'Pythia-6.9b', 'Pythia-6.9b'],
    'Type': ['Baseline', 'Multiplied Attention\nAltered', 'Baseline', 'Multiplied Attention\nAltered'],
    'Percentage': [4.13, 50.29, 30.32, 49.46]
})

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="model", y="Percentage", hue="Type",
            data=data_long, palette=[FACTUAL_COLOR, COUNTERFACTUAL_COLOR],
            edgecolor='black', ax=ax)

# Place legend outside the plot area
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', left=False, bottom=False,
               labelsize=AXIS_TEXT_SIZE)
ax.set_xlabel('')
ax.set_ylabel('% factual answers', fontsize=AXIS_TEXT_SIZE)
plt.tight_layout()

# Save bar plot
plt.savefig(f"{SAVE_DIR_NAME}/{model}_{experiment}_heads_pattern/multiplied_pattern.pdf", dpi=300, bbox_inches='tight')
