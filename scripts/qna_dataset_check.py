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
    gts, preds = [], []
    for row in tqdm(dataset):
        gts.append(row["target_true"].strip())
        _, attribute = inference(row["prompt"], model, tokenizer)
        preds.append(attribute.strip())

    gts = np.array(gts)
    preds = np.array(preds)
    indices = np.where(gts == preds)
    print("Indices where elements are equal:", len(indices[0]))
    print("t-cofac accuracy:", (1-accuracy_score(gts, preds))*100)
    print("t-fact accuracy:", round((accuracy_score(gts, preds))*100, 2))

    print(preds)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    
    args = parser.parse_args()
    hf_model_name = get_hf_model_name(args.model)

    dataset_path = f"../data/full_data_sampled_{args.model}.json"
    
    main(hf_model_name, dataset_path, args.start, args.end)