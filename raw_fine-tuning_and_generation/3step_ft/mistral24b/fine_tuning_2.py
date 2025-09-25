import json
import re
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import PeftModel
from transformers import EarlyStoppingCallback, TrainerCallback
import torch


model_name = "mistralai/Mistral-Small-24B-Instruct-2501"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if (tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token  
    
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.gradient_checkpointing_enable()


model = PeftModel.from_pretrained(model, "<PATH>", is_trainable=True)
model.train()

# Datensätze laden

def load_json_dataset(file_path):
    # Lädt eine JSON-Datei und konvertiert sie in ein Hugging Face Dataset
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return Dataset.from_list(data)

def formatting_func(example, dataset):
    current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    random_examples = [dataset[i] for i in indices[:min(5, len(dataset))]]
    
    noise_predicates = set()
    for random_example in random_examples:
        noise_predicates.update(re.findall(r"\b\w+(?=\()", random_example["FOL"]))
    
    all_predicates = current_predicates.union(noise_predicates)
    predicates_string = ", ".join(sorted(all_predicates))
    return {"messages":[
        {"role": "system", "content": "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                            +"∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
                            +"Use only the following predicates:"+f'Predicates=["'+ predicates_string + '"]\n'
                            +"Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."},
        {"role": "user", "content":example["NL"]},
        {"role": "assistant", "content": f"𝜙={example['FOL']}"},
        ]}

data_dir = "<PATH>"
train_dataset = load_json_dataset(f"{data_dir}/train.json")
val_dataset = load_json_dataset(f"{data_dir}/val.json")


train_dataset = train_dataset.map(
    lambda x: formatting_func(x, train_dataset),
    remove_columns=["NL", "FOL"],
    batched=False,
)

val_dataset = val_dataset.map(
    lambda x: formatting_func(x, val_dataset),
    remove_columns=["NL", "FOL"],
    batched=False,
)


training_args = TrainingArguments(
    output_dir="<PATH>",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=12,
    weight_decay=0.01,
    bf16=True, 
    save_total_limit=5,
    logging_steps=100,
    report_to="none",
    ddp_find_unused_parameters=False,
    ddp_backend="nccl"
)

#  Trainer initialisieren

class BestModelTracker(TrainerCallback):
    def __init__(self):
        self.best_epoch = None
        self.best_eval_loss = float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_loss", float('inf')) < self.best_eval_loss:
            self.best_eval_loss = metrics["eval_loss"]
            self.best_epoch = state.epoch

best_model_tracker = BestModelTracker()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=4),
        best_model_tracker
    ],
)

#  Training starten
trainer.train()

#  Modell speichern
trainer.save_model(
   "<PATH>"
)

print("✅ Training abgeschlossen und Modell gespeichert!")
print(f"📊 Bestes Modell wurde nach Epoche {best_model_tracker.best_epoch:.2f} gespeichert")
print(f"📉 Beste Eval Loss: {best_model_tracker.best_eval_loss:.4f}")
