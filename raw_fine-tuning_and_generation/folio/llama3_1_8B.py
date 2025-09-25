import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SYSTEM_MESSAGE = (
    "You are a helpful AI assistant that translates Natural Language (NL) text into First-Order Logic (FOL) formulas for use in an automated theorem prover. "
    "Use only the given quantors and junctors: ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor). "
    "Use consistent predicate and function names across all formulas. "
    "Use previously introduced predicate and function names when applicable, to ensure consistency for theorem proving. "
    "Start your answer with '𝜙=' followed by the FOL formula. Do not include any other text."
)


# Modell & Tokenizer laden
model_name = "<PATH>"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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

model = torch.compile(model)

# Prompt-Templates
def create_prompt(text):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

def create_prompt_with_context(text, previous_fol):
    context = "\n".join([f"𝜙{i+1} = {fol}" for i, fol in enumerate(previous_fol)])
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "assistant", "content": ("Previously translated premises in FOL:\n" + context) if previous_fol else ""},
            {"role": "user", "content":  text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

def create_conclusion_prompt(conclusion, fol_premises):
    premise_fol_context = "\n".join([f"𝜙{i+1} = {fol}" for i, fol in enumerate(fol_premises)])
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "assistant", "content": "Previously translated premises in FOL:\n" + premise_fol_context},
            {"role": "user", "content": conclusion},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

# Ein- und Ausgabepfade
input_file = "<PATH>"
output_file = "<PATH>"

# Daten laden
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Ergebnisse als JSONL schreiben (streaming)
with open(output_file, "w", encoding="utf-8") as out_file:
    for example in tqdm(data, desc="Verarbeite Beispiele"):
        fol_premises = []
        premises = example.get("premises", [])

        # === Sequentielle Prämissenverarbeitung mit Kontext ===
        for idx, premise in enumerate(premises):
            prompt = create_prompt_with_context(premise, fol_premises)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=False,
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            fol = decoded[decoded.rfind("=")+1:].strip()
            fol_premises.append(fol)

        # === Verarbeitung der Konklusion mit allen Prämissen im Kontext ===
        conclusion = example.get("conclusion", "")
        if conclusion:
            prompt = create_conclusion_prompt(conclusion, fol_premises)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=False,
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            fol_conclusion = decoded[decoded.rfind("=")+1:].strip()
        else:
            fol_conclusion = ""

        # Ergebnis schreiben
        result = {
            "premises": premises,
            "premises-FOL": fol_premises,
            "conclusion": conclusion,
            "conclusion-FOL": fol_conclusion,
            "label": example.get("label", None),
        }
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Fertig. Ergebnisse gespeichert in {output_file}")