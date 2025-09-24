import os
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments, Trainer
from utils.constants import FT_SETTINGS_TYPE, T5_BASE, T5_BASE_CURICULLUM_STEP2, T5_BASE_CURICULLUM_STEP3, T5_3B, T5_3B_CURICULLUM_STEP2, T5_3B_CURICULLUM_STEP3, FLAN_T5_XXL, FLAN_T5_XXL_CURICULLUM_STEP2, FLAN_T5_XXL_CURICULLUM_STEP3, META_LLAMA_8B, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3, EARLY_STOPPING_PATIENCE, BEST_MODEL_METRIC, NUM_EPOCHS, STRATEGY, SAVE_LIMIT, LOAD_BEST_MODEL, DISABLE_TQDM, REPORT_TO, LOGGING_STEPS, BF16, WEIGHT_DECAY, WARMUP_STEPS, ADAM_EPSILON, LR_SCHEDULER_TYPE, WARMUP_RATIO, DDP_BACKEND, DDP_FIND_UNUSED_PARAMETERS
from utils.modelloader import extract_base_model
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from typing import Any

class BestModelTracker(TrainerCallback):
    """Callback to track the best model based on evaluation loss."""
    def __init__(self):
        self.best_epoch = None
        self.best_eval_loss = float('inf')
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_loss", float('inf')) < self.best_eval_loss:
            self.best_eval_loss = metrics["eval_loss"]
            self.best_epoch = state.epoch

best_model_tracker = BestModelTracker()


def get_trainer_for_model(model, model_name: str, ft_setting: FT_SETTINGS_TYPE, train_dataset: Dataset, val_dataset: Dataset, output_dir: str) -> Any:
    """Set up the Trainer with early stopping and best model tracking."""

    if extract_base_model(model_name) == T5_BASE:
        training_args = TrainingArguments(
            output_dir=output_dir,
            load_best_model_at_end=LOAD_BEST_MODEL,
            metric_for_best_model=BEST_MODEL_METRIC,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=NUM_EPOCHS,
            eval_strategy=STRATEGY,
            save_strategy=STRATEGY,
            save_total_limit=SAVE_LIMIT,
            learning_rate=0.001,
            weight_decay=WEIGHT_DECAY,
            adam_epsilon=ADAM_EPSILON,
            warmup_steps=WARMUP_STEPS,
            gradient_accumulation_steps=1,
            disable_tqdm=DISABLE_TQDM
            )
        
        best_model_tracker = BestModelTracker()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
                best_model_tracker
            ],
        )
        return trainer
    
    elif extract_base_model(model_name) == T5_3B:
        training_args = TrainingArguments(
            output_dir=output_dir,
            load_best_model_at_end=LOAD_BEST_MODEL,
            metric_for_best_model=BEST_MODEL_METRIC,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=NUM_EPOCHS,
            eval_strategy=STRATEGY,
            save_strategy=STRATEGY,
            save_total_limit=SAVE_LIMIT,
            learning_rate=1e-4,
            weight_decay=WEIGHT_DECAY,
            adam_epsilon=ADAM_EPSILON,
            warmup_steps=WARMUP_STEPS,
            gradient_accumulation_steps=1,
            disable_tqdm=DISABLE_TQDM
        )
        
        best_model_tracker = BestModelTracker()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=4),
                best_model_tracker
            ],
        )
        return trainer 
    elif extract_base_model(model_name) == FLAN_T5_XXL:
        if model_name == FLAN_T5_XXL:
            lora_config = LoraConfig(
                r=16, 
                lora_alpha=32, 
                target_modules=[
                    "shared", # Shared embedding layer
                    "lm_head",  # Output projection
                    "q",  # Query projection
                    "k",  # Key projection
                    "v",  # Value projection
                    "o",  # Output projection
                    "wi",  # Input projection in feed-forward
                    "wo",  # Output projection in feed-forward
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )


        model = get_peft_model(model, lora_config)
        training_args = TrainingArguments(
            output_dir=output_dir,
            load_best_model_at_end=LOAD_BEST_MODEL,
            metric_for_best_model=BEST_MODEL_METRIC,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=NUM_EPOCHS,
            eval_strategy=STRATEGY,
            save_strategy=STRATEGY,
            save_total_limit=SAVE_LIMIT,
            learning_rate=1e-4,
            weight_decay=WEIGHT_DECAY,
            adam_epsilon=ADAM_EPSILON,
            warmup_steps=WARMUP_STEPS,
            gradient_accumulation_steps=1,
            bf16=True, 
            disable_tqdm=DISABLE_TQDM
        )
        
        best_model_tracker = BestModelTracker()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
                best_model_tracker
            ],
        )
        return trainer
    
    elif extract_base_model(model_name) == META_LLAMA_8B:
        lora_config = None
        if model_name == META_LLAMA_8B:
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
            output_dir=output_dir,
            load_best_model_at_end=LOAD_BEST_MODEL,
            metric_for_best_model=BEST_MODEL_METRIC,
            eval_strategy=STRATEGY,
            save_strategy=STRATEGY,
            learning_rate=1e-5,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            warmup_ratio=WARMUP_RATIO,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=8,  
            per_device_eval_batch_size=8,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            bf16=BF16, 
            save_total_limit=SAVE_LIMIT,
            logging_steps=LOGGING_STEPS,
            report_to=REPORT_TO,
        )

        best_model_tracker = BestModelTracker()
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=4),
                best_model_tracker
            ],
        )
        return trainer
    
    elif extract_base_model(model_name) == MISTRAL_24B:
        lora_config = None
        if model_name == MISTRAL_24B:
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
            output_dir=output_dir,
            load_best_model_at_end=LOAD_BEST_MODEL,
            metric_for_best_model=BEST_MODEL_METRIC,
            eval_strategy=STRATEGY,
            save_strategy=STRATEGY,
            learning_rate=1e-5,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            warmup_ratio=WARMUP_RATIO,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=8,  
            per_device_eval_batch_size=8,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            bf16=BF16, 
            save_total_limit=SAVE_LIMIT,
            logging_steps=LOGGING_STEPS,
            report_to=REPORT_TO,
            ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
            ddp_backend=DDP_BACKEND
            )

        best_model_tracker = BestModelTracker()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=4),
                best_model_tracker
            ],
        )
        return trainer
    
    elif extract_base_model(model_name) == OLMO_32B:
        lora_config = None
        if model_name == OLMO_32B:
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
            output_dir=output_dir,
            load_best_model_at_end=LOAD_BEST_MODEL,
            metric_for_best_model=BEST_MODEL_METRIC,
            eval_strategy=STRATEGY,
            save_strategy=STRATEGY,
            learning_rate=1e-5,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            warmup_ratio=WARMUP_RATIO,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=8,  
            per_device_eval_batch_size=8,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            bf16=BF16, 
            save_total_limit=SAVE_LIMIT,
            logging_steps=LOGGING_STEPS,
            report_to=REPORT_TO,
            ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
            ddp_backend=DDP_BACKEND
        )

        best_model_tracker = BestModelTracker()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=4),
                best_model_tracker
            ],
        )
        return trainer
    




def train(model, model_name: str, ft_setting: FT_SETTINGS_TYPE, train_dataset, val_dataset):
    """Train the model with early stopping and best model tracking."""

    output_dir = ""
    output_model_path = ""
    if model_name in [T5_BASE_CURICULLUM_STEP2, T5_BASE_CURICULLUM_STEP3, T5_3B_CURICULLUM_STEP2, T5_3B_CURICULLUM_STEP3, FLAN_T5_XXL_CURICULLUM_STEP2, FLAN_T5_XXL_CURICULLUM_STEP3, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3]:
        next_version = ft_setting.rsplit("_", 1)[0] + f"_{int(ft_setting.split('_')[-1]) + 1}"
        output_model_path = os.path.join(os.getenv("HF_HOME"), extract_base_model(model_name), next_version)
        output_dir = os.path.join(os.getenv("HF_HOME"), extract_base_model(model_name), "temp", next_version )

    else:
        output_model_path = os.path.join(os.getenv("HF_HOME"), model_name, ft_setting)
        output_dir = os.path.join(os.getenv("HF_HOME"), model_name, "temp", ft_setting)

    trainer = get_trainer_for_model(model, model_name, ft_setting, train_dataset, val_dataset, output_dir)
    trainer.train()
    trainer.save_model(output_model_path)


    

    
        


   

