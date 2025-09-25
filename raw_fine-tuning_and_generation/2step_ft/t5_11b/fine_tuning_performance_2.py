import json
import torch
import gc
from peft import PeftModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

output_file = (
   "<PATH>"
)

# Modell und Tokenizer laden
model_path = "<PATH>"
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

def prepare_data(example):
    current_predicates = example["PREDICATES_PRED"][example["PREDICATES_PRED"].rfind("Predicates"):].strip()
    return {
        'NL': 'translate English natural language statements into first-order logic (FOL) using only the following predicates:\n' 
        + current_predicates
        + example['NL'],
        'FOL': example['FOL_GROUND-TRUTH']
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

gc.collect()

results = []
batch_size = 32
total_predictions = len(test_dataset_tokenized)
total_batches = (total_predictions + batch_size - 1) // batch_size

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, total_predictions)
    
    # Batch-Verarbeitung der Eingaben
    batch_input_ids = torch.stack([
        test_dataset_tokenized[i]["input_ids"] 
        for i in range(start_idx, end_idx)
    ]).to("cuda")
    
    batch_labels = torch.stack([
        test_dataset_tokenized[i]["labels"] 
        for i in range(start_idx, end_idx)
    ]).to("cuda")

    # Dekodiere die NL-Sätze für den Batch
    sentences = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
    nl_sentences = [
        " ".join([
            token.split("_")[0]
            for token in sentence.replace(
                "translate English natural language statements into first-order logic (FOL): ",
                "",
            ).split()
        ])
        for sentence in sentences
    ]

    # Generiere Vorhersagen für den gesamten Batch
    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            max_length=256,
            min_length=1,
            num_beams=5,
            length_penalty=2.0,
            early_stopping=False,
        )

    # Dekodiere die Vorhersagen
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ground_truths = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)

    # Speichere die Ergebnisse für den Batch
    for nl, pred, truth in zip(nl_sentences, predictions, ground_truths):
        results.append({
            "NL": nl,
            "FOL_PRED": pred,
            "FOL_GROUND-TRUTH": truth,
        })

    # Fortschrittsanzeige
    print(f"Verarbeitet: Batch {batch_idx+1}/{total_batches} "
          f"(Beispiele {end_idx}/{total_predictions})")

    # Speichere Zwischenergebnisse
    if (batch_idx + 1) % 5 == 0:  # Alle 5 Batches speichern
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

# Finale Speicherung der Ergebnisse
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Verarbeitung abgeschlossen. Ergebnisse gespeichert in {output_file}")
