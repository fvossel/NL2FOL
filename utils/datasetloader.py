import json
import re
import random
from utils.constants import FT_SETTINGS, GENERATTION_SETTINGS, FT_SETTINGS_TYPE, GENERATTION_SETTINGS_TYPE, FOL_TOKENS, TORCH_DEVICE, FOL_LITERALS, T5_BASE, T5_BASE_CURICULLUM_STEP2, T5_BASE_CURICULLUM_STEP3, T5_3B, T5_3B_CURICULLUM_STEP2, T5_3B_CURICULLUM_STEP3, FLAN_T5_XXL, FLAN_T5_XXL_CURICULLUM_STEP2, FLAN_T5_XXL_CURICULLUM_STEP3, META_LLAMA_8B, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3
from datasets import Dataset
from typing import Any
from utils.modelloader import extract_base_model


def load_json_dataset(file_path: str) -> Dataset:
    """Loads a JSON file and converts it into a Hugging Face Dataset."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return Dataset.from_list(data)

def prepare_data(example: dict, model_name: str, setting: FT_SETTINGS_TYPE | GENERATTION_SETTINGS_TYPE, dataset: Dataset, tokenizer: Any, train: bool=True) -> dict:
    """Prepares an input example for training."""

    if model_name in [T5_BASE, T5_BASE_CURICULLUM_STEP2, T5_BASE_CURICULLUM_STEP3, T5_3B, T5_3B_CURICULLUM_STEP2, T5_3B_CURICULLUM_STEP3, FLAN_T5_XXL, FLAN_T5_XXL_CURICULLUM_STEP2, FLAN_T5_XXL_CURICULLUM_STEP3]:
        if setting != FT_SETTINGS.new_tokens and setting != GENERATTION_SETTINGS.new_tokens:
            for old, new in zip(FOL_LITERALS, FOL_TOKENS):
                example["FOL"] = example["FOL"].replace(old, new)

        task_prefix = ""
        label =  {"FOL": example["FOL"]}
        if setting in [FT_SETTINGS.standard, FT_SETTINGS.curiculum_step3, FT_SETTINGS.multilingual] or setting == GENERATTION_SETTINGS.standard:
            task_prefix = "translate English natural language statements into first-order logic (FOL): "

        elif setting in [FT_SETTINGS.gt_predicates, FT_SETTINGS.gt_predicates_noise, FT_SETTINGS.curiculum_step1, FT_SETTINGS.curiculum_step2] or setting in [GENERATTION_SETTINGS.gt_predicates, GENERATTION_SETTINGS.gt_predicates_noise]:
            current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
            all_predicates = current_predicates
            if setting in [FT_SETTINGS.gt_predicates_noise, FT_SETTINGS.curiculum_step2] or setting == GENERATTION_SETTINGS.gt_predicates_noise:
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                random_examples = [dataset[i] for i in indices[:min(5, len(dataset))]]
                
                noise_predicates = set()
                for random_example in random_examples:
                    noise_predicates.update(re.findall(r"\b\w+(?=\()", random_example["FOL"]))
                all_predicates = current_predicates.union(noise_predicates)
            task_prefix = "translate English natural language statements into first-order logic (FOL) using only the following predicates:\n" + f"Predicates=["+ ", ".join(sorted(all_predicates)) + "]\n"

        elif setting == FT_SETTINGS.step_1 or setting == GENERATTION_SETTINGS.predicates_only:
            task_prefix = "extract predicates from English natural language statements for translating into first-order logic (FOL): "
            label = {"PREDICATES": f"Predicates=["+ ", ".join(sorted(current_predicates)) + "]\n"}
        return {
            'NL': task_prefix + example['NL'],
            **label
        }
    elif model_name in [META_LLAMA_8B, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3]:
        system_prompt = ""
        label = f"𝜙={example['FOL']}"

        if setting in [FT_SETTINGS.standard, FT_SETTINGS.curiculum_step3, FT_SETTINGS.multilingual] or setting == GENERATTION_SETTINGS.standard:
            system_prompt = """You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:
                            ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor).
                            Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."""
        elif setting in [FT_SETTINGS.gt_predicates, FT_SETTINGS.gt_predicates_noise, FT_SETTINGS.curiculum_step1, FT_SETTINGS.curiculum_step2] or setting in [GENERATTION_SETTINGS.gt_predicates, GENERATTION_SETTINGS.gt_predicates_noise]:
            current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
            all_predicates = current_predicates
            if setting in [FT_SETTINGS.gt_predicates_noise, FT_SETTINGS.curiculum_step2] or setting == GENERATTION_SETTINGS.gt_predicates_noise:
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                random_examples = [dataset[i] for i in indices[:min(5, len(dataset))]]
                
                noise_predicates = set()
                for random_example in random_examples:
                    noise_predicates.update(re.findall(r"\b\w+(?=\()", random_example["FOL"]))
                all_predicates = current_predicates.union(noise_predicates)
            predicates_string = ", ".join(sorted(all_predicates))
            system_prompt = f"""You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:
                            ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor).
                            Use only the following predicates:"+f'Predicates=[""" + predicates_string + """]
                            Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."""
        elif setting == FT_SETTINGS.step_1 or setting == GENERATTION_SETTINGS.predicates_only:
            system_prompt = """You are a helpful AI assistant that extracts predicates from Natural Language (NL) text for translating into First-Order Logic (FOL):
                            Start your answer with 'Predicates=[' followed by the predicates in alphabetical order followed by ']'. Do not include any other text."""
        if train:
            return {
                "messages":[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":example["NL"]},
                    {"role": "assistant", "content": label},
                ]
            }
        else:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":example["NL"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
    
def tokenize_data(dataset: Dataset, tokenizer: Any, setting: FT_SETTINGS_TYPE | GENERATTION_SETTINGS_TYPE) -> Dataset:
    """Tokenisiert die Daten."""
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['NL'], padding='max_length', truncation=True, max_length=512)
        labels = tokenizer(examples['FOL'] if setting != FT_SETTINGS.step_1 and setting != GENERATTION_SETTINGS.predicates_only else examples["PREDICATES"], padding='max_length', truncation=True, max_length=512)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Wende die Tokenisierung auf das Dataset an
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_dataset


def load_datasets_for_training(model_name: str, ft_setting: FT_SETTINGS_TYPE, tokenizer: Any) -> tuple[Dataset, Dataset]:
    """Load the training and validation datasets."""

    train_dataset = load_json_dataset("datasets/train.json")
    val_dataset = load_json_dataset("datasets/val.json")

    train_dataset = train_dataset.map(
    lambda x: prepare_data(x, model_name, ft_setting, train_dataset, tokenizer),
    remove_columns=["NL", "FOL"] if model_name in [META_LLAMA_8B, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3] else None, 
    batched=False
    )

    val_dataset = val_dataset.map(
        lambda x: prepare_data(x, model_name, ft_setting, val_dataset, tokenizer),  
        remove_columns=["NL", "FOL"] if model_name in [META_LLAMA_8B, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3] else None, 
        batched=False
    )

    if model_name in [T5_BASE, T5_BASE_CURICULLUM_STEP2, T5_BASE_CURICULLUM_STEP3, T5_3B, T5_3B_CURICULLUM_STEP2, T5_3B_CURICULLUM_STEP3, FLAN_T5_XXL, FLAN_T5_XXL_CURICULLUM_STEP2, FLAN_T5_XXL_CURICULLUM_STEP3]:
        train_dataset = tokenize_data(train_dataset, tokenizer, ft_setting)
        val_dataset = tokenize_data(val_dataset, tokenizer, ft_setting)

    return train_dataset, val_dataset

def load_dataset_for_generation(model_name: str, generation_setting: GENERATTION_SETTINGS_TYPE, tokenizer: Any) -> Dataset:
    """Load the test dataset for generation."""
    test_dataset = load_json_dataset("datasets/test.json")
    test_dataset = test_dataset.map(
        lambda x: prepare_data(x, model_name, generation_setting, test_dataset, tokenizer, False),  
        remove_columns=["NL", "FOL"] if model_name in [META_LLAMA_8B, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3] else None, 
        batched=False
    )
    if extract_base_model(model_name) in [T5_BASE, T5_3B, FLAN_T5_XXL]:
        test_dataset = tokenize_data(test_dataset, tokenizer, generation_setting)
    return test_dataset

    