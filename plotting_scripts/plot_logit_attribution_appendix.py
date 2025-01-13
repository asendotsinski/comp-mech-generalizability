import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings("ignore")


def create_heatmap_base(data, x, y, value, cmap):
    pivot_table = data.pivot(index=y, columns=x, values=value)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap=cmap, cbar_kws={'label': f'{value.capitalize()} Mean'})
    return plt


def process_data(data, pattern, label_column, layer_offset,
                 relevant_positions,
                 value):
    # Filter rows with valid labels
    # print(data["label"].unique())
    # print(label_column, data[data[label_column]])
    valid_pattern = rf"^[0-9]+_{pattern}$"
    data_filtered = data[data[label_column].str.contains(valid_pattern, regex=True)]

    if data_filtered.empty:
        raise ValueError(f"No rows match the pattern '{valid_pattern}'. Check your data and label column.")

    # Extract layer and position
    data_filtered.loc[:, 'layer'] = data_filtered[label_column].str.extract(rf"([0-9]+)_{pattern}")

    if data_filtered.isnull().any().any():
        non_matching_rows = data[~data[label_column].str.contains(valid_pattern, na=False)]
        print("Non-matching rows:", non_matching_rows)
        raise ValueError(
            f"Extraction failed: Expected 2 columns but got {data_filtered.shape[1]}. Check the label pattern.")

    # Assign extracted columns to the dataframe
    data_filtered.loc[:, 'layer'] = data_filtered['layer'].astype(int)
    data_filtered.loc[:, 'position'] = data_filtered['position'].astype(int)

    max_layer = data_filtered['layer'].max()
    max_position = data_filtered['position'].max()

    # Adjust layer offset
    data_filtered.loc[:, 'layer'] = data_filtered['layer'] + layer_offset

    data_filtered.loc[:, 'layer'] = pd.Categorical(data_filtered['layer'], categories=range(max_layer + 1),
                                            ordered=True)
    data_filtered.loc[:, 'position'] = pd.Categorical(data_filtered['position'], categories=range(max_position + 1),
                                               ordered=True)

    # Filter relevant positions
    data_filtered = data_filtered[data_filtered['position'].isin(relevant_positions)]
    # data_filtered.dropna(subset=["layer"], inplace=True)

    # Map unique positions
    unique_positions = data_filtered['position'].unique()
    position_mapping = {pos: i for i, pos in enumerate(unique_positions)}
    data_filtered['mapped_position'] = data_filtered['position'].map(position_mapping)

    # Negate diff_mean
    # data_filtered[value] = -data_filtered[value]
    # print(data_filtered["layer"].isna().sum())
    print(value, pattern)
    print(data_filtered[value].mean())

    return data_filtered, max_layer


def generate_heatmap(data, relevant_labels, max_layer, output_file,
                     value, cmap, axis_text_size=12, axis_title_size=14):
    n_relevant_positions = len(relevant_labels)

    plt = create_heatmap_base(data, "layer", "mapped_position", value, cmap)
    plt.xticks(ticks=np.arange(0, max_layer + 1)+0.5, labels=np.arange(0, max_layer + 1), fontsize=axis_text_size)
    plt.yticks(ticks=np.arange(0, n_relevant_positions)+0.5, labels=relevant_labels, fontsize=axis_text_size)
    plt.yticks(rotation=0, va='center')
    plt.xticks(ha='center')
    plt.xlabel("Layer", fontsize=axis_title_size)
    plt.ylabel("", fontsize=axis_title_size)
    plt.title("Logit Diff Heatmap", fontsize=axis_title_size)

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"Heatmap saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for MLP and Attn data.")
    parser.add_argument("--input_file", type=str,
                        default="../results/copyVSfact/logit_attribution/gpt2_full/logit_attribution_data.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--model", type=str,
                        default="gpt2",
                        help="Model name.")
    parser.add_argument("--experiment", type=str,
                        default="gpt2",
                        help="Experiment name.")
    args = parser.parse_args()

    data = pd.read_csv(args.input_file)

    relevant_positions = [1, 4, 5, 6, 8, 11, 12]
    relevant_labels = ["Subject", "Relation", "Relation Last", "Attribute*",
                       "Subject repeat", "Relation repeat",
                       "Last"]

    ## CP
    # MLP Heatmap
    VALUE = "cp_mean"
    mlp_data, mlp_max_layer = process_data(data, "mlp_out", "label",
                                           0, relevant_positions,
                                           value=VALUE)
    mlp_output_file = f"../results/copyVSfact/logit_attribution/gpt2_full/logit_attribution_mlp_out_2a_cp.pdf"
    generate_heatmap(mlp_data, relevant_labels, mlp_max_layer, mlp_output_file,
                     value=VALUE, cmap="Reds")

    # Attention Heatmap
    attn_data, attn_max_layer = process_data(data, "attn_out", "label",
                                             0, relevant_positions,
                                             value=VALUE)
    attn_output_file = f"../results/copyVSfact/logit_attribution/gpt2_full/logit_attribution_attn_out_2a_cp.pdf"
    generate_heatmap(attn_data, relevant_labels, attn_max_layer, attn_output_file,
                     value=VALUE, cmap="Reds")

    ## MEM
    # MLP Heatmap
    VALUE = "mem_mean"
    mlp_data, mlp_max_layer = process_data(data, "mlp_out", "label",
                                           0, relevant_positions,
                                           value=VALUE)
    mlp_output_file = f"../results/copyVSfact/logit_attribution/gpt2_full/logit_attribution_mlp_out_2a_mem.pdf"
    generate_heatmap(mlp_data, relevant_labels, mlp_max_layer, mlp_output_file,
                     value=VALUE, cmap="Blues")

    # Attention Heatmap
    attn_data, attn_max_layer = process_data(data, "attn_out", "label",
                                             0, relevant_positions,
                                             value=VALUE)
    attn_output_file = f"../results/copyVSfact/logit_attribution/gpt2_full/logit_attribution_attn_out_2a_mem.pdf"
    generate_heatmap(attn_data, relevant_labels, attn_max_layer, attn_output_file,
                     value=VALUE, cmap="Blues")


if __name__ == "__main__":
    main()