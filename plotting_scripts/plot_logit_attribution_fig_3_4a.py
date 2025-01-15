import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Set working directory
os.chdir("../results")
print("Current working directory:", os.getcwd())
SAVE_DIR_NAME = "python_paper_plots"

# Configurations
palette = {
    "GPT2": "#003f5c",
    "GPT2-medium": "#58508d",
    "GPT2-large": "#bc5090",
    "GPT2-xl": "#ff6361",
    "Pythia-6.9b": "#ffa600"
}
FACTUAL_COLOR = "#005CAB"
FACTUAL_CMAP = sns.diverging_palette(250, 10, as_cmap=True)
COUNTERFACTUAL_COLOR = "#E31B23"
COUNTERFACTUAL_CMAP = sns.diverging_palette(10, 250, as_cmap=True)

model = "gpt2"
model_folder = "gpt2_full"
n_layers = 12
experiment = "copyVSfact"
positions_name = [
    "-", "Subject", "2nd Subject", "3rd Subject", "Relation", "Relation Last",
    "Attribute*", "-", "Subject Repeat", "2nd Subject repeat",
    "3nd Subject repeat", "Relation repeat", "Last"
]
relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last"]
n_relevant_position = 7

AXIS_TITLE_SIZE = 20
AXIS_TEXT_SIZE = 18
HEATMAP_SIZE = 10

directory_path = f"{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Function to create heatmaps
def create_heatmap(data, x, y, fill, title,
                   midpoint=0, relevant_positions=None,
                   block=None):
    plt.figure(figsize=(12, 10))
    pivot_data = data.pivot(index=y, columns=x, values=fill)
    if relevant_positions:
        cbar_kws = {
            "label": f"Logit Difference"
        }
        heatmap = sns.heatmap(
            pivot_data,
            annot=False,
            cmap=FACTUAL_CMAP,
            center=midpoint,
            linewidths=0.5,
            linecolor="grey",
            yticklabels=relevant_positions,
            cbar_kws=cbar_kws
        )
        plt.xlabel(x.capitalize(), fontsize=AXIS_TEXT_SIZE)
        plt.ylabel("")
    else:
        cbar_kws = {
            "label": f"Factual{' ' * 50}Counterfactual"
        }
        heatmap = sns.heatmap(
            pivot_data,
            annot=False,
            cmap=FACTUAL_CMAP,
            center=midpoint,
            linewidths=0.5,
            linecolor="grey",
            cbar_kws=cbar_kws,
        )
        plt.gca().invert_yaxis()
        plt.xlabel(x.capitalize(), fontsize=AXIS_TEXT_SIZE)
        plt.ylabel(y.capitalize(), fontsize=AXIS_TEXT_SIZE)

    heatmap.tick_params(left=False, bottom=False)
    heatmap.figure.axes[-1].yaxis.label.set_size(AXIS_TEXT_SIZE)
    plt.title(title, fontsize=AXIS_TITLE_SIZE)
    plt.xticks(fontsize=AXIS_TEXT_SIZE, rotation=0)
    plt.yticks(fontsize=AXIS_TEXT_SIZE, rotation=0)
    heatmap.figure.axes[-1].axes.tick_params(labelsize=AXIS_TEXT_SIZE-2)
    plt.tight_layout()


def create_barplot(data, x, y, color, title, axis_title_size, axis_text_size):
    """
    Function to create a horizontal bar plot with grid and inverted y-axis.
    """
    plt.figure(figsize=(12, 8))
    data_sorted = data.sort_values(by=y, ascending=False)

    barplot = sns.barplot(x=x, y=y, data=data_sorted, color=color, edgecolor="black")
    barplot.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    barplot.tick_params(left=False, bottom=False)

    # Set y-ticks at 0.5 separation
    y_ticks = np.arange(-1, 2, 0.5)
    plt.yticks(y_ticks, fontsize=axis_text_size)
    barplot.tick_params(left=False, bottom=False)
    barplot.set_axisbelow(True)

    plt.xlabel("layer", fontsize=axis_text_size)
    plt.ylabel(r"$\Delta_{cofa}$", fontsize=axis_title_size)
    plt.title(title, fontsize=axis_title_size)
    plt.xticks(fontsize=axis_text_size)
    plt.yticks(fontsize=axis_text_size)

    # plt.gca().invert_yaxis()  # Invert bars if necessary
    plt.tight_layout()


# Load data
data_file = f"{experiment}/logit_attribution/{model_folder}/logit_attribution_data.csv"
data = pd.read_csv(data_file)

#########################################
######## Heat Map: Head Position ########
#########################################

# Processing for head heatmap
data_head = data[data['label'].str.match(r'^L[0-9]+H[0-9]+$')].copy()
number_of_position = data_head['position'].max()
data_head = data_head[data_head['position'] == number_of_position]
data_head[['layer', 'head']] = data_head['label'].str.extract(r'L(\d+)H(\d+)').astype(int)
data_head['diff_mean'] = -data_head['diff_mean']

# Find the max layer and head
max_layer = data_head['layer'].max()
max_head = data_head['head'].max()

# Convert 'Layer' and 'Head' to categorical types with specific levels
data_head['layer'] = pd.Categorical(data_head['layer'], categories=range(max_layer + 1))
data_head['head'] = pd.Categorical(data_head['head'], categories=range(max_head + 1))

# Plot heatmap for head
create_heatmap(
    data=data_head,
    x="layer",
    y="head",
    fill="diff_mean",
    title=r"$\Delta_{cofa}$ Heatmap",
)
filename=f"{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution/logit_attribution_head_position{number_of_position}.pdf"
plt.savefig(filename, bbox_inches='tight')

# Compute factual impacts
factual_impact = data_head.groupby("layer")["diff_mean"].apply(lambda x: x[x < 0].sum()).sum()
l10h7 = data_head[(data_head['layer'] == 10) & (data_head['head'] == 7)]['diff_mean'].iloc[0]
l11h10 = data_head[(data_head['layer'] == 11) & (data_head['head'] == 10)]['diff_mean'].iloc[0]

print(f"L10H7 Impact (%): {100 * l10h7 / factual_impact:.2f}")
print(f"L11H10 Impact (%): {100 * l11h10 / factual_impact:.2f}")
print(f"L10H7 + L11H10 Impact (%): {100 * (l10h7+l11h10) / factual_impact:.2f}")

# Sum contributions for Layer 7 (L7H2 + L7H10) and Layer 9 (L9H6 + L9H9)
l7_contrib = data_head[(data_head['layer'] == 7) & (data_head['head'].isin([2, 10]))]["diff_mean"].sum()
l9_contrib = data_head[(data_head['layer'] == 9) & (data_head['head'].isin([6, 9]))]['diff_mean'].sum()

# considering only positive mean values for cofac
layer_7_total = data_head[data_head['layer'] == 7]
layer_7_total = layer_7_total[layer_7_total['diff_mean'] > 0]['diff_mean'].sum()
layer_9_total = data_head[data_head['layer'] == 9]
layer_9_total = layer_9_total[layer_9_total['diff_mean'] > 0]['diff_mean'].sum()

# Calculate the percentages
l7_percent = l7_contrib / layer_7_total
l9_percent = l9_contrib / layer_9_total

# Print the results
print(f"L7H2 + L7H10 Impact for Layer 7 (%): {l7_percent:.2f}")
print(f"L9H6 + L9H9 Impact for Layer 9 (%): {l9_percent:.2f}")


#########################################
########## Bar Plots ############
#########################################

# Processing for MLP and Attention barplots
data_mlp = data[data['label'].str.endswith('_mlp_out')].copy()
data_mlp['layer'] = data_mlp['label'].str.extract(r'(\d+)_mlp_out').astype(int)
max_position = data_mlp['position'].max()
data_mlp = data_mlp[data_mlp['position'] == max_position]
data_mlp['diff_mean'] = -data_mlp['diff_mean']

data_attn = data[data['label'].str.endswith('_attn_out')].copy()
data_attn['layer'] = data_attn['label'].str.extract(r'(\d+)_attn_out').astype(int)
data_attn = data_attn[data_attn['position'] == max_position]
data_attn['diff_mean'] = -data_attn['diff_mean']

data_barplot = pd.merge(
    data_mlp[['layer', 'diff_mean']].rename(columns={"diff_mean": "MLP Block"}),
    data_attn[['layer', 'diff_mean']].rename(columns={"diff_mean": "Attention Block"}),
    on="layer"
)
data_barplot['layer'] = data_barplot['layer'].astype(int)

# Plot barplot for MLP
mlp_filename = f"{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution/mlp_block_norm.pdf"
create_barplot(data_barplot, x="layer", y="MLP Block", color="#bc5090", title="MLP Block",
               axis_title_size=AXIS_TITLE_SIZE, axis_text_size=AXIS_TEXT_SIZE)
plt.savefig(mlp_filename)

# Plot barplot for Attention
attn_filename = f"{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution/attn_block_norm.pdf"
create_barplot(data_barplot, x="layer", y="Attention Block", color="#ffa600", title="Attention Block",
             axis_title_size=AXIS_TITLE_SIZE, axis_text_size=AXIS_TEXT_SIZE)
plt.savefig(attn_filename)


#########################################
######### Heat Map: MLP & Attn #########
#########################################

def process_heatmap_data(data, pattern, position_filter, block):
    """
    Processes data for creating heatmaps. This function can be used for both MLP and Attention data.
    """
    # Filter data based on the label suffix

    data_filtered = data[data['label'].str.endswith(pattern)].copy()
    # Extract 'layer' from the label and convert it to int
    data_filtered['layer'] = data_filtered['label'].str.extract(rf'(\d+){pattern}').astype(int)

    # if block == "attn":
    #     data_filtered['layer'] = data_filtered['layer'] - 1
    # else:
    #     data_filtered['layer'] = data_filtered['layer'] + 1

    # Filter for specific positions
    data_filtered = data_filtered[data_filtered['position'].isin(position_filter)]
    # Reverse the diff_mean
    data_filtered['diff_mean'] = -data_filtered['diff_mean']

    # Create position mapping
    unique_positions = data_filtered['position'].unique()
    position_mapping = {position: index for index, position in enumerate(unique_positions)}
    # Apply the mapping to create a new 'mapped_position' column
    data_filtered['mapped_position'] = data_filtered['position'].map(position_mapping)

    # Convert layer to factor (categorical)
    max_layer = data_filtered['layer'].max()
    data_filtered['layer'] = pd.Categorical(data_filtered['layer'], categories=range(0, max_layer + 1))

    return data_filtered


relevant_positions = ["Subject", "Relation", "Relation Last", "Attribute*",
                          "Subject repeat", "Relation repeat",
                          "Last"]
position_filter=[1, 4, 5, 6, 8, 11, 12]
# MLP Heatmap processing
data_mlp = process_heatmap_data(data, pattern="_mlp_out", position_filter=position_filter, block="mlp")
create_heatmap(data_mlp, "layer", "mapped_position", "diff_mean",
               "MLP Block", relevant_positions=relevant_positions, block="mlp")
# Save the MLP heatmap
mlp_out_filename = f"{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution/logit_attribution_mlp_out.pdf"
plt.savefig(mlp_out_filename, bbox_inches="tight")

# Attention Heatmap processing
data_attn = process_heatmap_data(data, pattern="_attn_out", position_filter=position_filter, block="attn")
create_heatmap(data_attn, "layer", "mapped_position", "diff_mean",
               "Attention Block", relevant_positions=relevant_positions, block="attn")
# Save the Attention heatmap
attn_out_filename = f"{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution/logit_attribution_attn_out.pdf"
plt.savefig(attn_out_filename, bbox_inches="tight")
plt.close()