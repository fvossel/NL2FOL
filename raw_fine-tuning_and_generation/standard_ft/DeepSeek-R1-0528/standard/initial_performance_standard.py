import requests
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

API_KEY = "<API-KEY>"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_WORKERS = 10  # Anzahl paralleler Anfragen

def process_entry(entry):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "temperature": 1.3,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                          "∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
                          """Here are some examples: -An energy-efficient building that incorporates renewable energy systems, sustainable materials, and innovative design strategies reduces its environmental impact and operating costs."  
     ->𝜙=∀x∀y∀z (EnergyEfficientBuilding(x) ∧ RenewableEnergySystems(y) ∧ SustainableMaterials(z) ∧ InnovativeDesignStrategies(x) ∧ Incorporates(x, y, z) → (ReducesEnvironmentalImpact(x) ∧ ReducesOperatingCosts(x)))  
   - "An event can be indoors, outdoors, or virtual, but not all three options."  
     ->𝜙=∃x (Event(x) ∧ ((Indoors(x) ∧ Outdoors(x) ∧ ¬Virtual(x)) ∨ (Indoors(x) ∧ ¬Outdoors(x) ∧ Virtual(x)) ∨ (¬Indoors(x) ∧ Outdoors(x) ∧ Virtual(x))))
   - "No artist who admires all designers is praised by any individual.
     ->∀x (Artist(x) → ∀y ((Designer(y) → Admire(x, y)) → ¬∃z (Individual(z) ∧ Praise(z, x))))"""
                          "Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."
            },
            {
                "role": "user",
                "content": entry["NL"]
            }
        ]
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    response_data = response.json()
    
    return {
        "NL": entry["NL"],
        "FOL_PRED": response_data["choices"][0]["message"]["content"],
        "FOL_GROUND-TRUTH": entry["FOL"]
    }

def main():
    input_file = "<PATH>"
    output_file = "<PATH>"

    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_entry, entry) for entry in data]
        
        for future in tqdm(futures, total=len(data)):
            result = future.result()
            results.append(result)
            
            # Speichere nach jeder Verarbeitung
            with open(output_file, "w", encoding="utf-8") as out_file:
                json.dump(results, out_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()