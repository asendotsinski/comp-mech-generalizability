# imports
import argparse
import os
import sys
from dataset import BaseDataset, DOMAINS
from experiment import Ablator
from model import ModelFactory
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from run_all import get_dataset_path

# Appending system paths
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
sys.path.append(os.path.abspath(os.path.join("../plotting_scripts")))


def plot_results(ablation_result,
                 axis_title_size=20,
                 axis_text_size=14):

    FACTUAL_COLOR = "#005CAB"
    COUNTERFACTUAL_COLOR = "#E31B23"

    # Plot settings
    plt.figure(figsize=(12, 8))
    bar_width = 0.4
    x = range(len(ablation_result["domain"]))

    # Plotting bars
    plt.bar(x, ablation_result["mem_win"], width=bar_width,
            label="mem_win", color=FACTUAL_COLOR, edgecolor="black")
    plt.bar([i + bar_width for i in x], ablation_result["cp_win"], width=bar_width, label="cp_win",
            color=COUNTERFACTUAL_COLOR, edgecolor="black")

    plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    plt.gca().set_axisbelow(True)

    # Adding labels and title
    plt.xlabel("Domain", fontsize=axis_text_size)
    plt.ylabel("Wins", fontsize=axis_text_size)
    # plt.title(f"Ablation comparison - {ablation_layer_heads}",
    #           fontsize=axis_title_size)
    plt.xticks([i + bar_width / 2 for i in x], ablation_result["domain"],
               rotation=90)
    plt.xticks(fontsize=axis_text_size)
    plt.yticks(fontsize=axis_text_size)
    plt.legend()
    plt.tick_params(left=False, bottom=False)

    # Show plot
    plt.tight_layout()
    # plt.show()


def run_ablator(model, dataset, batch_size, multiplier,
                experiment, prompt_type, position, ablation_layer_heads):
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    results = []

    if experiment == "copyVSfactDomain":
        for domain in DOMAINS:
            if domain in excluded_domains:
                print(f"Skipping {domain}")
                continue

            # experiment setup
            ds = BaseDataset(path=dataset,
                            model=model,
                            experiment=experiment,
                            domain=domain,
                            prompt_type=prompt_type,
                            no_subject=False)
            ablator = Ablator(model=model, dataset=ds, experiment=experiment, batch_size=batch_size)
            ablator.set_heads(heads=ablation_layer_heads, value=multiplier, position=position)

            # run the ablator
            result_df = ablator.run()
            result_df["domain"] = domain
            result_df["ablation_layer_heads"] = str(ablation_layer_heads)

            # shift columns
            columns = ["domain", "ablation_layer_heads"] + [col for col in result_df.columns if
                                                            col not in ["domain", "ablation_layer_heads"]]
            result_df = result_df[columns]

            results.append(result_df)

        ablation_result = pd.concat(results)
    else:
        # experiment setup
        ds = BaseDataset(path=dataset,
                         model=model,
                         experiment=experiment,
                         domain=None,
                         prompt_type=prompt_type,
                         no_subject=False)
        ablator = Ablator(model=model, dataset=ds, experiment=experiment, batch_size=batch_size)
        ablator.set_heads(heads=ablation_layer_heads, value=multiplier, position=position)

        # run the ablator
        ablation_result = ablator.run()
        ablation_result["ablation_layer_heads"] = str(ablation_layer_heads)

        # shift columns
        columns = ["ablation_layer_heads"] + [col for col in ablation_result.columns if
                                              col not in ["ablation_layer_heads"]]
        ablation_result = ablation_result[columns]

    ablation_result.to_csv(f"{SAVE_FOLDER}/ablation_{position}_{ablation_layer_heads}.csv",
                                        index=False)

    return ablation_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run domain ablation experiments.")
    parser.add_argument("--dataset", type=str, default="copyVSfactDomain", help="Path to the dataset.")
    parser.add_argument("--experiment", type=str, default="copyVSfactDomain", help="Name of the experiment.")
    parser.add_argument("--downsampled_dataset", type=bool, default=False, help="Whether to use the downnsampled dataset or not.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name.")
    parser.add_argument("--position", type=str, default="attribute", help="Position setting for the ablator.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for the ablator.")
    parser.add_argument("--multiplier", type=int, default=5, help="Multiplier value for ablation.")
    parser.add_argument("--prompt_type", type=str, default=None, help="Type of prompt used.")
    parser.add_argument("--ablation_layer_heads", type=str, default="[(10, 7), (11, 10)]",
                        help="Heads to ablate in the format '[(layer, head), ...]'.")

    args = parser.parse_args()

    # no data present for these domains
    excluded_domains = ["Adult", "Beauty_and_Fitness", "Pets_and_Animals"]

    # Parse ablation_layer_heads
    ablation_layer_heads = eval(
        args.ablation_layer_heads)


    SAVE_FOLDER = f"../results/{args.dataset}/attention_modification/{args.model_name}_full"

    # Load model
    model = ModelFactory.create(args.model_name)

    # Run ablator
    ablation_result = run_ablator(
        model=model,
        dataset=get_dataset_path(args),
        batch_size=args.batch_size,
        multiplier=args.multiplier,
        experiment=args.experiment,
        prompt_type=args.prompt_type,
        position=args.position,
        ablation_layer_heads=ablation_layer_heads
    )

    print(ablation_result.sum())

    if args.dataset == "copyVSfactDomain":
        # ablation_result = pd.read_csv(f"{SAVE_FOLDER}/ablation_{args.position}_{args.ablation_layer_heads}.csv")
        # save plots
        plot_results(ablation_result)
        plot_filename = f"{SAVE_FOLDER}/ablation_{args.position}_{args.ablation_layer_heads}.pdf"
        plt.savefig(plot_filename, bbox_inches="tight")
        print("Sving figure")