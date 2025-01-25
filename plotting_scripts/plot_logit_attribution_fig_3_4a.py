import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Set save folder
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

# domain name
DOMAIN = "Science"

# GPT-2
MODEL = "gpt2"
MODEL_FOLDER = "gpt2_full"

# Pythia
# MODEL = "pythia-6.9b"
# MODEL_FOLDER = "pythia-6.9b_full"

# EXPERIMENT = "copyVSfact"
# EXPERIMENT = "copyVSfactQnA"
EXPERIMENT = "copyVSfactDomain"

if DOMAIN:
    MODEL_FOLDER += f"_{DOMAIN}"

positions_name = [
    "-", "Subject", "2nd Subject", "3rd Subject", "Relation", "Relation Last",
    "Attribute*", "-", "Subject Repeat", "2nd Subject repeat",
    "3nd Subject repeat", "Relation repeat", "Last"
]

# plotting setup
if EXPERIMENT == "copyVSfactQnA":
    relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*",
                         "Interrogative", "Relation repeat", "Subject repeat", "Last"]
    position_filter = [1, 4, 5, 6, 7, 8, 9, 13]
else:
    relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*",
                         "Subject repeat", "Relation repeat", "Last"]
    position_filter = [1, 4, 5, 6, 9, 12, 13]

AXIS_TITLE_SIZE = 20
if MODEL == "gpt2":
    AXIS_TEXT_SIZE = 16
else:
    AXIS_TEXT_SIZE = 14

# Function to create heatmaps
def create_heatmap(data, x, y, fill, title,
                   midpoint=0, add_positions=None,
                   block=None):
    plt.figure(figsize=(12, 10))
    pivot_data = data.pivot(index=y, columns=x, values=fill)
    if add_positions:
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
            yticklabels=relevant_position,
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

    barplot = sns.barplot(x=x, y=y, data=data, color=color, edgecolor="black")
    barplot.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    barplot.tick_params(left=False, bottom=False)

    # Set y-ticks at 0.5 separation
    if MODEL == "gpt2":
        y_ticks = np.arange(-1, 2, 0.5)
    else:
        y_ticks = np.arange(-0.25, 0.75, 0.25)
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


def plot_logit_attribution_fig_3_4a(
    model="gpt2",
    model_folder="gpt2_full",
    experiment="copyVSfact",
    domain=None
):
    # load the data
    try:
        data_file = f"../results/{experiment}/logit_attribution/{model_folder}/logit_attribution_data.csv"
        data = pd.read_csv(data_file)
    except Exception as e:
        print(f".csv file now found - {e}")
        return

    if domain:
        directory_path = f"../results/{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution/{domain}"
    else:
        directory_path = f"../results/{SAVE_DIR_NAME}/{model}_{experiment}_logit_attribution"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

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
    filename=f"{directory_path}/logit_attribution_head_position{number_of_position}.pdf"
    plt.savefig(filename, bbox_inches='tight')


    if model == "gpt2":
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
    max_position = data_attn['position'].max()
    data_attn = data_attn[data_attn['position'] == max_position]
    data_attn['diff_mean'] = -data_attn['diff_mean']

    # data_barplot = pd.merge(
    #     data_mlp[['layer', 'diff_mean']].rename(columns={"diff_mean": "MLP Block"}),
    #     data_attn[['layer', 'diff_mean']].rename(columns={"diff_mean": "Attention Block"}),
    #     on="layer"
    # )
    # data_barplot['layer'] = data_barplot['layer'].astype(int)

    # Plot barplot for MLP
    mlp_filename = f"{directory_path}/mlp_block_norm.pdf"
    create_barplot(data_mlp, x="layer", y="diff_mean", color="#bc5090", title="MLP Block",
                axis_title_size=AXIS_TITLE_SIZE, axis_text_size=AXIS_TEXT_SIZE)
    plt.savefig(mlp_filename)

    # Plot barplot for Attention
    attn_filename = f"{directory_path}/attn_block_norm.pdf"
    create_barplot(data_attn, x="layer", y="diff_mean", color="#ffa600", title="Attention Block",
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


    # MLP Heatmap processing
    data_mlp = process_heatmap_data(data, pattern="_mlp_out", position_filter=position_filter, block="mlp")
    create_heatmap(data_mlp, "layer", "mapped_position", "diff_mean",
                "MLP Block", add_positions=True, block="mlp")
    # Save the MLP heatmap
    mlp_out_filename = f"{directory_path}/logit_attribution_mlp_out.pdf"
    plt.savefig(mlp_out_filename, bbox_inches="tight")

    # Attention Heatmap processing
    data_attn = process_heatmap_data(data, pattern="_attn_out", position_filter=position_filter, block="attn")
    create_heatmap(data_attn, "layer", "mapped_position", "diff_mean",
                "Attention Block", add_positions=True, block="attn")
    
    # Save the Attention heatmap
    attn_out_filename = f"{directory_path}/logit_attribution_attn_out.pdf"
    plt.savefig(attn_out_filename, bbox_inches="tight")
    plt.close()

    print("Plots saved at: ", directory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and visualize data.')
    parser.add_argument('model', type=str, nargs='?',
                        help='Name of the model',
                        default=MODEL)
    parser.add_argument('experiment', type=str, nargs='?',
                        help='Name of the experiment',
                        default=EXPERIMENT)
    parser.add_argument('model_folder', type=str, nargs='?',
                        help='Name of the model folder',
                        default=MODEL_FOLDER)
    parser.add_argument('domain', type=str, nargs='?',
                        help='Name of the domain',
                        default=DOMAIN)
    args = parser.parse_args()
    # plotting setup
    if args.experiment == "copyVSfactQnA":
        relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*",
                             "Interrogative", "Relation repeat", "Subject repeat", "Last"]
        position_filter = [1, 4, 5, 6, 7, 8, 9, 13]
    else:
        relevant_position = ["Subject", "Relation", "Relation Last", "Attribute*",
                             "Subject repeat", "Relation repeat", "Last"]
        position_filter = [1, 4, 5, 6, 9, 12, 13]

    AXIS_TITLE_SIZE = 20
    if args.model == "gpt2":
        AXIS_TEXT_SIZE = 16
    else:
        AXIS_TEXT_SIZE = 14

    plot_logit_attribution_fig_3_4a(
        model=args.model,
        experiment=args.experiment,
        model_folder=args.model_folder,
        domain=args.domain
    )
