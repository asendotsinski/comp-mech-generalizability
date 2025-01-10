# File: plot_logit_lens.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_logit_lens_2b(folder_path, position):

    # Load the CSV file
    file_path = folder_path + "logit_lens_data.csv"
    data = pd.read_csv(file_path)

    # Filter data for the desired position
    data_last_position = data[data['position'] == position]

    # Extract data for plotting
    layers = data_last_position['layer']
    factual_token = data_last_position['mem']
    counterfactual_token = data_last_position['cp']

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(layers, factual_token, label="Factual Token", color="blue", marker='o', linestyle='-', linewidth=2)
    plt.plot(layers, counterfactual_token, label="Counterfactual Token", color="red", marker='o', linestyle='-', linewidth=2)

    # Add labels, title, and legend
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Logit", fontsize=14)
    plt.title("Logit Lens Analysis at Position " + str(position), fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(visible=True, linestyle='--', alpha=0.6)

    # Save the plot
    plt.tight_layout()
    plt.savefig(folder_path + "logit_lens_plot_position_" + str(position) + ".png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_logit_lens_2b("../results/copyVSfact/logit_lens/gpt2_full/", 10)