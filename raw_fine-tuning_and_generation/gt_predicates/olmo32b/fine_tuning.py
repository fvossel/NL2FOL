import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import EarlyStoppingCallback, TrainerCallback
import torch
import re
import gc


model_name = "allenai/OLMo-2-0325-32B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if (tokenizer.pad_token is None):
    tokenizer.pad_token = tokenizer.eos_token  
    

# Datensätze laden

def load_json_dataset(file_path):
    # Lädt eine JSON-Datei und konvertiert sie in ein Hugging Face Dataset
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return Dataset.from_list(data)

def formatting_func(example):
    current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
    return {"messages":[
        {"role": "system", "content": "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) using only the given quantors and junctors:"
                            +"∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), ↔ (if and only if), ⊕ (xor)."
                            +"Use only the following predicates:"+f'Predicates=["'+ ", ".join(sorted(current_predicates)) + '"]\n'
                            +"Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."},
        {"role": "user", "content":example["NL"]},
        {"role": "assistant", "content": f"𝜙={example['FOL']}"},
        ]}

data_dir = "<PATH>"
train_dataset = load_json_dataset(f"{data_dir}/train.json")
val_dataset = load_json_dataset(f"{data_dir}/val.json")


train_dataset = train_dataset.map(
    lambda x: formatting_func(x),
    remove_columns=["NL", "FOL"],
    batched=False,
)

val_dataset = val_dataset.map(
    lambda x: formatting_func(x),
    remove_columns=["NL", "FOL"],
    batched=False,
)


lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
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
    save_total_limit=12,
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
    peft_config=lora_config,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        best_model_tracker
    ],
)

#  Training starten
trainer.train(resume_from_checkpoint=True)

trainer.save_model("<PATH>")

print("✅ Training abgeschlossen und Modell gespeichert!")
print(f"📊 Bestes Modell wurde nach Epoche {best_model_tracker.best_epoch:.2f} gespeichert")
print(f"📉 Beste Eval Loss: {best_model_tracker.best_eval_loss:.4f}")
