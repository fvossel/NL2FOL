from json import dump
from os import makedirs
from utils.constants import GENERATTION_SETTINGS_TYPE, T5_BASE, T5_3B, FLAN_T5_XXL, META_LLAMA_8B, MISTRAL_24B, OLMO_32B, TORCH_DEVICE
from utils.modelloader import extract_base_model
from torch import stack, no_grad
from datasets import Dataset
from typing import Any
from tqdm import tqdm
from pathvalidate import sanitize_filename


def save_results_to_file(results: list, model_name: str, generation_setting: GENERATTION_SETTINGS_TYPE):
    """Saves the generation results to a text file."""
    output_file = f"results/{sanitize_filename(extract_base_model(model_name))}_{generation_setting}.json"
    makedirs("results", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        dump(results, f, indent=4, ensure_ascii=False)

def generate(model: Any, model_name: str, tokenizer: Any, dataset: Dataset, generation_setting: GENERATTION_SETTINGS_TYPE, batch_size: int):
    """Generate outputs using the model and tokenizer based on the specified generation settings."""
    results = []
    if extract_base_model(model_name) in [T5_BASE, T5_3B, FLAN_T5_XXL]:
        total_predictions = len(dataset)
        total_batches = (total_predictions + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_predictions)
            
            # Batch-Verarbeitung der Eingaben
            batch_input_ids = stack([
                dataset[i]["input_ids"] 
                for i in range(start_idx, end_idx)
            ]).to("cuda")
            
            batch_labels = stack([
                dataset[i]["labels"] 
                for i in range(start_idx, end_idx)
            ]).to("cuda")

            sentences = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

            with no_grad():
                outputs = model.generate(
                    batch_input_ids,
                    max_length=512,
                    min_length=1,
                    num_beams=5,
                    length_penalty=2.0,
                    early_stopping=False,
                )

            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ground_truths = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)

            # Speichere die Ergebnisse für den Batch
            for nl, pred, truth in zip(sentences, predictions, ground_truths):
                results.append({
                    "NL": nl,
                    "PRED": pred,
                    "GROUND-TRUTH": truth,
                })
            
            if (batch_idx + 1) % 5 == 0:  
                save_results_to_file(results, model_name, generation_setting)

        save_results_to_file(results, model_name, generation_setting)
    elif extract_base_model(model_name) in [META_LLAMA_8B, MISTRAL_24B, OLMO_32B]:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_data = dataset[i : i + batch_size]
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True).to(
                "cuda"
            )
            with no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    do_sample=True,
                )
            responses = [
                tokenizer.decode(
                    output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for output in outputs
            ]

            for idx, item in enumerate(batch_data):
                results.append(
                    {
                        "PROMPT": item,
                        "PRED": responses[idx],
                    }
                )

            save_results_to_file(results, model_name, generation_setting)

        save_results_to_file(results, model_name, generation_setting)