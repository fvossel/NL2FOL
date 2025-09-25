import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

API_KEY = "<API-KEY>"
MAX_WORKERS = 20
MAX_RETRIES = 5
BASE_BACKOFF = 4


def call_with_retries(client, model, messages, max_tokens=None, temperature=None):
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except (APIConnectionError, RateLimitError, APIStatusError, TimeoutError) as e:
            wait_time = BASE_BACKOFF * (2 ** attempt)
            print(f"[Retry {attempt+1}/{MAX_RETRIES}] Connection failed ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[Retry {attempt+1}/{MAX_RETRIES}] Unexpected error: {e}. Retrying in {BASE_BACKOFF * (2 ** attempt)}s...")
            time.sleep(BASE_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries.")


def process_entry(entry):
    reasoning = ""
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    try:
        messages_reasoner = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                    " ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor). "
                    """Here are some examples: -An energy-efficient building that incorporates renewable energy systems, sustainable materials, and innovative design strategies reduces its environmental impact and operating costs."  
     ->𝜙=∀x∀y∀z (EnergyEfficientBuilding(x) ∧ RenewableEnergySystems(y) ∧ SustainableMaterials(z) ∧ InnovativeDesignStrategies(x) ∧ Incorporates(x, y, z) → (ReducesEnvironmentalImpact(x) ∧ ReducesOperatingCosts(x)))  
   - "An event can be indoors, outdoors, or virtual, but not all three options."  
     ->𝜙=∃x (Event(x) ∧ ((Indoors(x) ∧ Outdoors(x) ∧ ¬Virtual(x)) ∨ (Indoors(x) ∧ ¬Outdoors(x) ∧ Virtual(x)) ∨ (¬Indoors(x) ∧ Outdoors(x) ∧ Virtual(x))))
   - "No artist who admires all designers is praised by any individual.
     ->∀x (Artist(x) → ∀y ((Designer(y) → Admire(x, y)) → ¬∃z (Individual(z) ∧ Praise(z, x))))"""
                    "Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."
                )
            },
            {"role": "user", "content": entry["NL"]}
        ]

        response = call_with_retries(
            client=client,
            model="deepseek-reasoner",
            messages=messages_reasoner,
            max_tokens=2500
        )
        message = response.choices[0].message
        if message.content == "":
            reasoning = message.reasoning_content
            raise Exception("Empty response from reasoner")

        return {
            "NL": entry["NL"],
            "FOL_PRED": message.content,
            "FOL_PRED_REASONING": message.reasoning_content,
            "FOL_GROUND-TRUTH": entry["FOL"]
        }

    except Exception as e:
        print(f"DeepSeek Reasoner failed for: {entry['NL'][:50]}... | Error: {e}")
        reasoning_text = f"Use your earlier thoughts on this: {reasoning}" if reasoning != "" else ""

        messages_fallback = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                    " ∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor). "
                    """Here are some examples: -An energy-efficient building that incorporates renewable energy systems, sustainable materials, and innovative design strategies reduces its environmental impact and operating costs."  
     ->𝜙=∀x∀y∀z (EnergyEfficientBuilding(x) ∧ RenewableEnergySystems(y) ∧ SustainableMaterials(z) ∧ InnovativeDesignStrategies(x) ∧ Incorporates(x, y, z) → (ReducesEnvironmentalImpact(x) ∧ ReducesOperatingCosts(x)))  
   - "An event can be indoors, outdoors, or virtual, but not all three options."  
     ->𝜙=∃x (Event(x) ∧ ((Indoors(x) ∧ Outdoors(x) ∧ ¬Virtual(x)) ∨ (Indoors(x) ∧ ¬Outdoors(x) ∧ Virtual(x)) ∨ (¬Indoors(x) ∧ Outdoors(x) ∧ Virtual(x))))
   - "No artist who admires all designers is praised by any individual.
     ->∀x (Artist(x) → ∀y ((Designer(y) → Admire(x, y)) → ¬∃z (Individual(z) ∧ Praise(z, x))))"""
                    + reasoning_text +
                    "Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."
                )
            },
            {"role": "user", "content": entry["NL"]}
        ]

        fallback_response = call_with_retries(
            client=client,
            model="deepseek-chat",
            messages=messages_fallback,
            temperature=1.3
        )
        message = fallback_response.choices[0].message
        return {
            "NL": entry["NL"],
            "FOL_PRED": message.content,
            "FOL_PRED_REASONING": reasoning,
            "FOL_GROUND-TRUTH": entry["FOL"],
            "FALLBACK_USED": True
        }


def main():
    input_file = "<PATH>"
    output_file = "<PATH>"
    output_file_2 = "<PATH>"

    with open(input_file, "r", encoding="utf-8") as file:
        orig_data = json.load(file)
    
    with open(output_file, "r", encoding="utf-8") as file:
        current_data = json.load(file)
        
    current_gt_fols = set(entry["NL"] for entry in current_data)
    data = [entry for entry in orig_data if entry["NL"] not in current_gt_fols]
        

    results = current_data

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, entry): entry for entry in data}

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)

                with open(output_file_2, "w", encoding="utf-8") as out_file:
                    json.dump(results, out_file, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"Unexpected failure: {e}")


if __name__ == "__main__":
    main()
