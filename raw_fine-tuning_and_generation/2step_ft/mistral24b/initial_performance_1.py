import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Definiert die Datei, in die die Ergebnisse gespeichert werden
output_file = "<PATH>"

# Definiert die Eingabedatei mit den NL-FOL-Daten
input_file = "<PATH>"


model_name = "mistralai/Mistral-Small-24B-Instruct-2501"

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
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that extracts predicates from Natural Language (NL) text for translating into First-Order Logic (FOL):"
                            +"Start your answer with 'Predicates=[' followed by the predicates in alphabetical order followed by ']'. Do not include any other text.",
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
        current_predicates = set(re.findall(r"\b\w+(?=\()", item.get("FOL")))
        results.append(
            {
                "NL": item.get("NL"),
                "PREDICATES_PRED": responses[idx],
                "PREDICATES_GROUND-TRUTH": ", ".join(sorted(current_predicates)),
            }
        )

    # Optional: Fortschrittsanzeige
    print(f"Processed {min(i+batch_size, len(data))}/{len(data)} examples")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Finished processing. Results saved to {output_file}")
