import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

output_file = "<output-file>"
input_file = "<input-file>"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=True,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

results = []


def formatting_func(example):
    current_predicates = example["PREDICATES_PRED"][example["PREDICATES_PRED"].rfind("\n")+1:].strip()
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                            +"∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
                            +"Use only the following predicates:" + current_predicates
                            +"Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text.",
            },
            {"role": "user", "content": example["NL"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

batch_size = 32
results = []

for i in range(0, len(data), batch_size):
    batch_data = data[i : i + batch_size]

    prompts = [formatting_func(item) for item in batch_data]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        "cuda"
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=250,
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
                "NL": item.get("NL"),
                "FOL_PRED": responses[idx],
                "PREDICATES_PRED": item.get("PREDICATES_PRED")[item.get("PREDICATES_PRED").rfind("\n")+1:].strip(),
                "PREDICATES-GROUND-TRUTH": item.get("PREDICATES_GROUND-TRUTH"),
            }
        )

    # Optional: Fortschrittsanzeige
    print(f"Processed {min(i+batch_size, len(data))}/{len(data)} examples")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Finished processing. Results saved to {output_file}")
