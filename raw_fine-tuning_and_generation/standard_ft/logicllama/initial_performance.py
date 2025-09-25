

import torch
import json
from transformers import  AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


# Definiert die Datei, in die die Ergebnisse gespeichert werden
output_file = "<PATH>"
input_file = "<PATH>"

base_model='huggyllama/llama-7b'

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, legacy=False)
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": '<unk>',
    "pad_token": '<unk>',
})
tokenizer.padding_side = "left"  # Allow batched inference

generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=250,
)

llama_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    use_cache=True,
    trust_remote_code=True
)

peft_path='yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0'


model = PeftModel.from_pretrained(
    llama_model,
    peft_path,
    torch_dtype=torch.float16,
    use_cache=True,
    trust_remote_code=True
)
model.to('cuda')

results = []

def formatting_func(example):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                            +"∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
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
batch_size = 24
results = []

# Daten in Batches verarbeiten
for i in range(0, len(data), batch_size):
    batch_data = data[i:i+batch_size]
    
    # Prompts für den aktuellen Batch erstellen
    prompts = [formatting_func(item) for item in batch_data]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        "cuda"
    )
    
    # Generiere Antworten für den Batch
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            generation_config=generation_config
        )
    
    # Dekodiere die Antworten
    responses = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
    
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

    # Ergebnisse in die Ausgabedatei schreiben
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Finished processing. Results saved to {output_file}")


