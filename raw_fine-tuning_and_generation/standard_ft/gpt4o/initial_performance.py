from openai import OpenAI
import json
import time

client = OpenAI(api_key="<API-KEY>")


# Datei-Pfade
output_file = "<PATH>"
input_file = "<PATH>"

# Lade die Eingabedaten
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)


results = []

count=len(data)
i=1


for entry in data:
    nl_sentence = entry["NL"]
    fol_ground_truth = entry["FOL"]
   
    try:
        response = client.responses.create(
            model="gpt-4o",
            temperature=0.2,
            max_output_tokens=250, 
            input=[
                {"role": "system", "content": "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                            +"∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
                            +"Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."},
                {"role": "user", "content": nl_sentence},
            ]
        )

        fol_translation = response.output_text

        # Speichere das Ergebnis in einer Liste
        results.append({
            "NL": nl_sentence,
            "FOL_PRED": fol_translation,
            "FOL_GROUND-TRUTH": fol_ground_truth
        })

        # Speichere nach jeder Anfrage das Ergebnis, um Datenverlust zu vermeiden
        with open(output_file, "w", encoding="utf-8") as out_file:
            json.dump(results, out_file, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"API-Fehler: {e}")
        time.sleep(10)
    print("Fortschritt: ", i, "/", count, end="\r")
    i+=1
    time.sleep(0.1)

print("Fertig! Alle Ergebnisse wurden gespeichert.")