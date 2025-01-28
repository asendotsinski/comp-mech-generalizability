import sys
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from pprint import pprint
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser


# ### Basic Inference

# **Default Dataset Basic Inference:**  
# - Indices where elements are equal: 413  
# - t-cofac accuracy: 95.87  
# - t-fact accuracy: 4.13  
# 
# **QA Dataset Basic Inference:**  
# - Indices where elements are equal: 3478  
# - t-cofac accuracy: 65.22
# - t-fact accuracy: 34.78  
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from utils import get_hf_model_name

model_name = "gpt2"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class DatasetEntry:
    def __init__(self, row):
        self.row = row
        self.base_prompt = row["base_prompt"]
        self.prompt = row["prompt"]
        self.target_true = row["target_true"]
        self.target_new = row["target_new"]
    
    def __eq__(self, other):
        return self.base_prompt == other.base_prompt and self.target_true == other.target_true and self.target_new == other.target_new and self.prompt == other.prompt
    

def are_rows_equal(row1, row2):
    base_prompt_equal = row1["base_prompt"] == row2["base_prompt"]
    target_true_equal = row1["target_true"] == row2["target_true"]
    target_new_equal = row1["target_new"] == row2["target_new"]

    return base_prompt_equal and target_true_equal and target_new_equal

def main(inference_model_name, model_names):
    hf_model_name = get_hf_model_name(inference_model_name)

    # gpt2 inference
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto",
            do_sample=False)
    
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    def inference(prompt, model, tokenizer):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model_outputs = model.generate(**inputs, 
                                    max_new_tokens=1, 
                                    return_dict_in_generate=True, output_scores=True, 
                                    pad_token_id=tokenizer.eos_token_id)
        generated_tokens_ids = model_outputs.sequences[0]
        generation = tokenizer.decode(generated_tokens_ids)
        attribute = tokenizer.decode(generated_tokens_ids[-1])

        return generation, attribute

    def inference(prompt, model, tokenizer):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model_outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, 
                                    pad_token_id=tokenizer.eos_token_id, do_sample=False)
        generated_tokens_ids = model_outputs.sequences[0]
        generation = tokenizer.decode(generated_tokens_ids)
        attribute = tokenizer.decode(generated_tokens_ids[-1])

        return generation, attribute
    
    def sequential_inference(dataset, prompt_key="prompt", subset=None):
        generations, attributes = [], []
        for row in tqdm(dataset):
            generation, attribute = inference(row[prompt_key], model, tokenizer)
            generations.append(generation.strip())
            attributes.append(attribute.strip())
        
        return generations, attributes
    
    datasets = {model_name: {} for model_name in model_names}
    for model_name in model_names:
        with open(f"../data/full_data_sampled_{model_name}_with_subjects.json", "r") as f:
            datasets[model_name]["og"] = sorted(json.load(f), key=lambda x: x["prompt"])

        with open(f"../data/cft_og_combined_data_sampled_{model_name}_with_questions.json", "r") as f:
            datasets[model_name]["qa_cft"] = sorted(json.load(f), key=lambda x: x["prompt"])
    
    og_ground_truths_per_model, og_predictions_per_model = {}, {}
    og_prompt_ground_truths_per_model, og_prompt_predictions_per_model = {}, {}
    qa_cft_ground_truths_per_model, qa_cft_predictions_per_model = {}, {}
    qa_cft_prompt_ground_truths_per_model, qa_cft_prompt_predictions_per_model = {}, {}
    for model_name in model_names:
        og_ground_truths, og_predictions = sequential_inference(datasets[model_name]["og"], prompt_key="base_prompt", subset=None)
        og_ground_truths_per_model[model_name] = og_ground_truths
        og_predictions_per_model[model_name] = og_predictions

        og_prompt_ground_truths, og_prompt_predictions = sequential_inference(datasets[model_name]["og"], prompt_key="prompt", subset=None)
        og_prompt_ground_truths_per_model[model_name] = og_prompt_ground_truths
        og_prompt_predictions_per_model[model_name] = og_prompt_predictions

        qa_cft_ground_truths, qa_cft_predictions = sequential_inference(datasets[model_name]["qa_cft"], prompt_key="base_prompt", subset=None)
        qa_cft_ground_truths_per_model[model_name] = qa_cft_ground_truths
        qa_cft_predictions_per_model[model_name] = qa_cft_predictions

        qa_cft_prompt_ground_truths, qa_cft_prompt_predictions = sequential_inference(datasets[model_name]["qa_cft"], prompt_key="prompt", subset=None)
        qa_cft_prompt_ground_truths_per_model[model_name] = qa_cft_prompt_ground_truths
        qa_cft_prompt_predictions_per_model[model_name] = qa_cft_prompt_predictions

    def check_qa_stats(dataset, ground_truths, predictions):    
        target_new = np.array([row["target_new"].strip() for row in dataset])
        target_true = np.array([row["target_true"].strip() for row in dataset])

        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions)

        fact_indices = np.where(predictions == target_true)[0]
        cofact_indices = np.where(predictions == target_new)[0]
        indices = np.concatenate([fact_indices, cofact_indices])

        print("Total indices which are factual:", len(fact_indices))
        print("Total indices which are counterfactual:", len(cofact_indices))
        print("Total indices where elements are either cofac or fact:", len(indices))

        df = pd.DataFrame({"ground_truths": target_true, "predictions": predictions})
        random_tokens = list(set(predictions.tolist()) - set(list(target_true.tolist())+target_new.tolist()))
        print("Total Random Tokens:", len(random_tokens))

        df_filtered = df[df["predictions"].isin(random_tokens)]
        print(df_filtered["predictions"].value_counts().head(5))

        invalid_indices = list(df_filtered.index)
        print("Total invalid Indices:", len(invalid_indices))

        return fact_indices, cofact_indices, indices, invalid_indices

    og_fact_indices_per_model, og_cofact_indices_per_model, og_indices_per_model, og_invalid_indices_per_model = {}, {}, {}, {}
    og_prompt_fact_indices_per_model, og_prompt_cofact_indices_per_model, og_prompt_indices_per_model, og_prompt_invalid_indices_per_model = {}, {}, {}, {}
    qa_cft_fact_indices_per_model, qa_cft_cofact_indices_per_model, qa_cft_indices_per_model, qa_cft_invalid_indices_per_model = {}, {}, {}, {}
    qa_cft_prompt_fact_indices_per_model, qa_cft_prompt_cofact_indices_per_model, qa_cft_prompt_indices_per_model, qa_cft_prompt_invalid_indices_per_model = {}, {}, {}, {}

    for model_name in model_names:
        # Checking if the model returns the factual token when given the original dataset base prompt
        og_fact_indices, og_cofact_indices, og_indices, og_invalid_indices = check_qa_stats(datasets[model_name]["og"], 
                                                                                    og_ground_truths_per_model[model_name], 
                                                                                    og_predictions_per_model[model_name])
        og_fact_indices_per_model[model_name] = og_fact_indices
        og_cofact_indices_per_model[model_name] = og_cofact_indices
        og_indices_per_model[model_name] = og_indices
        og_invalid_indices_per_model[model_name] = og_invalid_indices

        og_prompt_fact_indices, og_prompt_cofact_indices, og_prompt_indices, og_prompt_invalid_indices = check_qa_stats(datasets[model_name]["og"], 
                                                                                    og_prompt_ground_truths_per_model[model_name], 
                                                                                    og_prompt_predictions_per_model[model_name])
        og_prompt_fact_indices_per_model[model_name] = og_prompt_fact_indices
        og_prompt_cofact_indices_per_model[model_name] = og_prompt_cofact_indices
        og_prompt_indices_per_model[model_name] = og_prompt_indices
        og_prompt_invalid_indices_per_model[model_name] = og_prompt_invalid_indices

        # Checking if the model returns the factual token when given the CFT QnA base prompt    
        qa_cft_fact_indices, qa_cft_cofact_indices, qa_cft_indices, qa_cft_invalid_indices = check_qa_stats(datasets[model_name]["qa_cft"], 
                                                                                    qa_cft_ground_truths_per_model[model_name], 
                                                                                    qa_cft_predictions_per_model[model_name])
        qa_cft_fact_indices_per_model[model_name] = qa_cft_fact_indices
        qa_cft_cofact_indices_per_model[model_name] = qa_cft_cofact_indices
        qa_cft_indices_per_model[model_name] = qa_cft_indices
        qa_cft_invalid_indices_per_model[model_name] = qa_cft_invalid_indices

        qa_cft_prompt_fact_indices, qa_cft_prompt_cofact_indices, qa_cft_prompt_indices, qa_cft_prompt_invalid_indices = check_qa_stats(datasets[model_name]["qa_cft"], 
                                                                                    qa_cft_prompt_ground_truths_per_model[model_name], 
                                                                                    qa_cft_prompt_predictions_per_model[model_name])
        qa_cft_prompt_fact_indices_per_model[model_name] = qa_cft_prompt_fact_indices
        qa_cft_prompt_cofact_indices_per_model[model_name] = qa_cft_prompt_cofact_indices
        qa_cft_prompt_indices_per_model[model_name] = qa_cft_prompt_indices
        qa_cft_prompt_invalid_indices_per_model[model_name] = qa_cft_prompt_invalid_indices

    og_dataset_new, qa_cft_dataset_new = [], []

    for model_name in model_names:
        dataset_array = np.array(datasets[model_name]["og"])
        qa_cft_dataset_array = np.array(datasets[model_name]["qa_cft"])

        # Convert to sorted lists for deterministic behavior
        og_indices = sorted(list(set(og_fact_indices_per_model[model_name]) & 
                               set(og_prompt_indices_per_model[model_name])))
        qa_cft_indices = sorted(list(set(qa_cft_fact_indices_per_model[model_name]) & 
                                   set(qa_cft_prompt_indices_per_model[model_name])))

        for idx in og_indices:
            row = DatasetEntry(dataset_array[idx])
            if row in og_dataset_new:
                continue
            og_dataset_new.append(row)

        for idx in qa_cft_indices:
            row = DatasetEntry(qa_cft_dataset_array[idx])
            if row in qa_cft_dataset_new:
                continue
            qa_cft_dataset_new.append(row)

    og_dataset_new = [row.row for row in og_dataset_new]
    qa_cft_dataset_new = [row.row for row in qa_cft_dataset_new]

    print(f'Final og dataset length: {len(og_dataset_new)}')
    print(f'Final qa_cft dataset length: {len(qa_cft_dataset_new)}')

    with open(f"../data/full_data_sampled_{inference_model_name}_with_subjects_downsampled.json", "w") as f:
        json.dump(og_dataset_new, f)

    with open(f"../data/cft_og_combined_data_sampled_{inference_model_name}_with_questions_downsampled.json", "w") as f:
        json.dump(qa_cft_dataset_new, f)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument('--models', nargs='+', type=str, default=['gpt2'], help='List of model datasets to use')

    args = parser.parse_args()

    print(args)

    main(args.model, args.models)