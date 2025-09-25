

import json
import torch
import re
import random
from peft import PeftModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

output_file = (
   "<PATH>"
)

# Modell und Tokenizer laden
model_path ="<PATH>"
adapter_path = "<PATH>"

tokenizer = T5Tokenizer.from_pretrained(
    model_path, trust_remote_code=True
)
base_model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    device_map="auto"
)

model = model.merge_and_unload()


def load_json_dataset(file_path):
    """Lädt eine JSON-Datei und konvertiert sie in ein Hugging Face Dataset."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return Dataset.from_list(data)

def prepare_data(example, dataset):
    current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    random_examples = [dataset[i] for i in indices[:min(5, len(dataset))]]
    
    noise_predicates = set()
    for random_example in random_examples:
        noise_predicates.update(re.findall(r"\b\w+(?=\()", random_example["FOL"]))
    
    all_predicates = current_predicates.union(noise_predicates)
    predicates_string = ", ".join(sorted(all_predicates))
    return {
        'NL': 'translate English natural language statements into first-order logic (FOL) using only the following predicates:\n'
        +f'Predicates=["'+ predicates_string + '"]\n'
        + example['NL'],
        'FOL': example['FOL']
    }
    
def tokenize_data(dataset, tokenizer):
    """Tokenisiert die Daten."""
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['NL'], padding='max_length', truncation=True, max_length=250)
        labels = tokenizer(examples['FOL'], padding='max_length', truncation=True, max_length=250)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Wende die Tokenisierung auf das Dataset an
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_dataset

test_dataset = load_json_dataset("<PATH>")

test_dataset = test_dataset.map(
    lambda x: prepare_data(x, test_dataset),  
    batched=False
)

test_dataset_tokenized = tokenize_data(test_dataset, tokenizer)


results = []

# Annahme: test_dataset hat eine Länge von total_predictions
total_predictions = len(test_dataset_tokenized)

for i in range(total_predictions):
    # Eingabe-IDs und Labels vom Testdatensatz
    input_ids = test_dataset_tokenized[i]["input_ids"].unsqueeze(0).to("cuda")
    labels = test_dataset_tokenized[i]["labels"].unsqueeze(0).to("cuda")

    # Dekodiere den NL-Satz
    sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    nl_sentence = " ".join(
        [
            token.split("_")[0]
            for token in sentence.replace(
                "translate English natural language statements into first-order logic (FOL): ",
                "",
            ).split()
        ]
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=256,
            min_length=1,
            num_beams=5,
            length_penalty=2.0,
            early_stopping=False,
        )

    # Vorhersage dekodieren und nachbearbeiten
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ergebnis speichern
    results.append(
        {
            "NL": nl_sentence,
            "FOL_PRED": prediction,
            "FOL_GROUND-TRUTH": tokenizer.decode(labels[0], skip_special_tokens=True),
        }
    )

    # Fortschrittsanzeige
    print(f"Verarbeitet: {i+1}/{total_predictions} Beispiele")

    # Ergebnisse in die Ausgabedatei schreiben
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Verarbeitung abgeschlossen. Ergebnisse gespeichert in {output_file}")
