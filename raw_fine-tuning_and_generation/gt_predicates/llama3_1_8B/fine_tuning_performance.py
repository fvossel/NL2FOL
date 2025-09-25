

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# Definiert die Datei, in die die Ergebnisse gespeichert werden
output_file = "<PATH>"

# Definiert die Eingabedatei mit den NL-FOL-Daten
input_file = "<PATH>"


model_name = "<PATH>"

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Modell laden und auf GPU verteilen
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Speichert die generierten Ergebnisse
results = []


def formatting_func(example):
    current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                            +"∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
                            +"Use only the following predicates:"+f'Predicates=["'+ ", ".join(sorted(current_predicates)) + '"]\n'
                            +"Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text.",
            },
            {"role": "user", "content": example["NL"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


# Lädt die Eingabedaten aus der JSON-Datei
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Batch-Größe festlegen
batch_size = 32
results = []

# Daten in Batches verarbeiten
for i in range(0, len(data), batch_size):
    batch_data = data[i : i + batch_size]

    # Prompts für den aktuellen Batch erstellen
    prompts = [formatting_func(item) for item in batch_data]

    # Tokenize den Batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        "cuda"
    )

    # Generiere Antworten für den Batch
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=250,
            do_sample=True,
        )

    # Dekodiere die Antworten
    responses = [
        tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for output in outputs
    ]

    # Speichere die Ergebnisse
    for idx, item in enumerate(batch_data):
        results.append(
            {
                "NL": item.get("NL"),
                "FOL_PRED": responses[idx],
                "FOL_GROUND-TRUTH": item.get("FOL"),
            }
        )

    # Optional: Fortschrittsanzeige
    print(f"Processed {min(i+batch_size, len(data))}/{len(data)} examples")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Finished processing. Results saved to {output_file}")
