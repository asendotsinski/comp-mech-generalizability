import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join("../src")))

from utils import get_hf_model_name

def main(hf_model_name, dataset_path, start, end):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    model.to(device)

    def inference(prompt, model, tokenizer):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model_outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, 
                                    pad_token_id=tokenizer.eos_token_id)
        generated_tokens_ids = model_outputs.sequences[0]
        generation = tokenizer.decode(generated_tokens_ids)
        attribute = tokenizer.decode(generated_tokens_ids[-1])

        return generation, attribute

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if start is None:
        start = 0   
    if end is None:
        end = len(dataset)

    dataset = dataset[start:end]

    # sequential inference
    ground_truths, counterfactuals, preds = [], [], []
    for row in tqdm(dataset):
        ground_truths.append(row["target_true"].strip())
        counterfactuals.append(row["target_new"].strip())
        _, attribute = inference(row["prompt"], model, tokenizer)
        preds.append(attribute.strip())

    ground_truths = np.array(ground_truths)
    preds = np.array(preds)
    indices = np.where(ground_truths == preds)[0]
    counterfactuals = np.array(counterfactuals)
    counterfactual_indices = np.where(counterfactuals == preds)[0]
    
    print(f'indices: {indices}')
    print(f'counterfactual_indices: {counterfactual_indices}')

    indices_with_unexpected_preds = []
    for i in range(start, end):
        if i not in indices and i not in counterfactual_indices:
            indices_with_unexpected_preds.append(i)

    unexpected_preds = [(dataset[i], preds[i]) for i in indices_with_unexpected_preds]

    print("Indices where prediction is factual:", len(indices))
    print("Indices where prediction is counterfactual:", len(counterfactual_indices))
    print("Indices where prediction is unexpected:", len(indices_with_unexpected_preds))
    print("Unexpected predictions:", unexpected_preds[:min(len(unexpected_preds), 10)])
    print("t-cofac accuracy:", (1-accuracy_score(ground_truths, preds))*100)
    print("t-fact accuracy:", round((accuracy_score(ground_truths, preds))*100, 2))

    print(preds)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="copyVSfactQnA")

    args = parser.parse_args()
    hf_model_name = get_hf_model_name(args.model)

    if args.dataset == "copyVSfact":
        dataset_path = f"../data/full_data_sampled_{args.model}_with_subjects.json"
    elif args.dataset == "copyVSfactQnA":
        dataset_path = f"../data/cft_og_combined_data_sampled_gpt2_with_questions.json"
    else:
        raise ValueError(f"Dataset {args.dataset} not found")

    main(hf_model_name, dataset_path, args.start, args.end)