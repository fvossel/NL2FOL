import json
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

output_file = (
  "<PATH>"
)

# Modell und Tokenizer laden
model_path = "<PATH>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)


def load_json_dataset(file_path):
    """Lädt eine JSON-Datei und konvertiert sie in ein Hugging Face Dataset."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return Dataset.from_list(data)

def prepare_data(example):
    current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
    return {
        'NL': 'translate English natural language statements into first-order logic (FOL) using only the following predicates:\n' 
        +f'Predicates=["'+ ", ".join(sorted(current_predicates)) + '"]\n'
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
    lambda x: prepare_data(x),  
    batched=False
)

test_dataset_tokenized = tokenize_data(test_dataset, tokenizer)


results = []

# Annahme: test_dataset hat eine Länge von total_predictions
total_predictions = len(test_dataset_tokenized)

for i in range(total_predictions):
    # Eingabe-IDs und Labels vom Testdatensatz
    input_ids = test_dataset_tokenized[i]["input_ids"].unsqueeze(0).to(device)
    labels = test_dataset_tokenized[i]["labels"].unsqueeze(0).to(device)

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
