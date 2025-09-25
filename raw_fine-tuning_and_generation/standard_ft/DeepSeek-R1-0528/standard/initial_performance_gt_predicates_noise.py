from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
import time
import random

API_KEY = "<API-KEY>"
API_URL = "https://api.deepseek.com"
MAX_WORKERS = 2


def process_entry(entry, dataset):
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    current_predicates = set(re.findall(r"\b\w+(?=\()",entry["FOL"]))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    random_examples = [dataset[i] for i in indices[:min(5, len(dataset))]]
    
    noise_predicates = set()
    for random_example in random_examples:
        noise_predicates.update(re.findall(r"\b\w+(?=\()", random_example["FOL"]))
    
    all_predicates = current_predicates.union(noise_predicates)
    predicates_string = ", ".join(sorted(all_predicates))
    system_prompt = (
        "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
        "∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
        """Here are some examples: -An energy-efficient building that incorporates renewable energy systems, sustainable materials, and innovative design strategies reduces its environmental impact and operating costs."  
 ->𝜙=∀x∀y∀z (EnergyEfficientBuilding(x) ∧ RenewableEnergySystems(y) ∧ SustainableMaterials(z) ∧ InnovativeDesignStrategies(x) ∧ Incorporates(x, y, z) → (ReducesEnvironmentalImpact(x) ∧ ReducesOperatingCosts(x)))  
   - "An event can be indoors, outdoors, or virtual, but not all three options."  
     ->𝜙=∃x (Event(x) ∧ ((Indoors(x) ∧ Outdoors(x) ∧ ¬Virtual(x)) ∨ (Indoors(x) ∧ ¬Outdoors(x) ∧ Virtual(x)) ∨ (¬Indoors(x) ∧ Outdoors(x) ∧ Virtual(x))))
   - "No artist who admires all designers is praised by any individual.
     ->∀x (Artist(x) → ∀y ((Designer(y) → Admire(x, y)) → ¬∃z (Individual(z) ∧ Praise(z, x))))"""
        "Use only the following predicates:" + f'Predicates=["' + predicates_string + '"]\n'
        "Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": entry["NL"]}
    ]
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                temperature=1.3
            )
            content = response.choices[0].message.content
            return {
                "NL": entry["NL"],
                "FOL_PRED": content,
                "FOL_GROUND-TRUTH": entry["FOL"]
            }
        except Exception as e:
            print(f"Fehler bei Anfrage (Versuch {attempt+1}/3): {e}")
            time.sleep(5 ** attempt)  # Exponentielles Backoff
    return {
        "NL": entry["NL"],
        "FOL_PRED": "ERROR",
        "FOL_GROUND-TRUTH": entry["FOL"]
    }

def main():
    input_file = "<PATH>"
    output_file = "<PATH>"

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_entry, entry, data) for entry in data]

        for future in tqdm(futures, total=len(data)):
            result = future.result()
            results.append(result)

            # Speichere nach jeder Verarbeitung
            with open(output_file, "w", encoding="utf-8") as out_file:
                json.dump(results, out_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()