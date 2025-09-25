
from transformers import T5Tokenizer, TrainingArguments, Trainer, T5ForConditionalGeneration
import torch
import json
import re
from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainerCallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name="google-t5/t5-3b"

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

tokenizer = T5Tokenizer.from_pretrained(model_name)

def load_json_dataset(file_path):
    """Lädt eine JSON-Datei und konvertiert sie in ein Hugging Face Dataset."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return Dataset.from_list(data)


def prepare_data(example):
    current_predicates = set(re.findall(r"\b\w+(?=\()", example["FOL"]))
    return {
        'NL': 'translate English natural language statements into first-order logic (FOL) using only the following predicates:\n'
        +f'Predicates=["'+ ", ".join(sorted(current_predicates)) + '"]\n'
        + example['NL'],
        'FOL': example['FOL']
    }
    
def tokenize_data(dataset, tokenizer):
    """Tokenisiert die Daten."""
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['NL'], padding='max_length', truncation=True, max_length=250)
        labels = tokenizer(examples['FOL'], padding='max_length', truncation=True, max_length=250)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Wende die Tokenisierung auf das Dataset an
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_dataset


train_dataset = load_json_dataset("<PATH>")
val_dataset = load_json_dataset("<PATH>")

train_dataset = train_dataset.map(
    lambda x: prepare_data(x),  
    batched=False
)
val_dataset = val_dataset.map(
    lambda x: prepare_data(x),  
    batched=False
)

train_dataset_tokenized = tokenize_data(train_dataset, tokenizer)
val_dataset_tokenized = tokenize_data(val_dataset, tokenizer)


# Set up training arguments
training_args = TrainingArguments(
    output_dir="<PATH>",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=12,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=12,
    learning_rate=1e-4,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    warmup_steps=500,
    gradient_accumulation_steps=1,
    disable_tqdm=False
)


class BestModelTracker(TrainerCallback):
    def __init__(self):
        self.best_epoch = None
        self.best_eval_loss = float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_loss", float('inf')) < self.best_eval_loss:
            self.best_eval_loss = metrics["eval_loss"]
            self.best_epoch = state.epoch

best_model_tracker = BestModelTracker()

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=4),
        best_model_tracker
    ],
)




# Start training
trainer.train()

# Save the trained model and tokenizer
model_path = "<PATH>"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

print("✅ Training abgeschlossen und Modell gespeichert!")
print(f"📊 Bestes Modell wurde nach Epoche {best_model_tracker.best_epoch:.2f} gespeichert")
print(f"📉 Beste Eval Loss: {best_model_tracker.best_eval_loss:.4f}")