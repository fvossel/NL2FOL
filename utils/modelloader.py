import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from utils.constants import FT_SETTINGS, FT_SETTINGS_TYPE, TORCH_DEVICE, FOL_LITERALS, T5_BASE, T5_BASE_CURICULLUM_STEP2, T5_BASE_CURICULLUM_STEP3, T5_3B, T5_3B_CURICULLUM_STEP2, T5_3B_CURICULLUM_STEP3, FLAN_T5_XXL, FLAN_T5_XXL_CURICULLUM_STEP2, FLAN_T5_XXL_CURICULLUM_STEP3, META_LLAMA_8B, META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3
from typing import Any
from torch import bfloat16
from pathlib import Path
from peft import PeftModel

def extract_base_model(path: str) -> str:
    """
    Extracts the model path (e.g., 'google-t5/t5-base') from a
    full cache path under HF_HOME.

    Returns:
        str: The model path relative to HF_HOME.
    """
    hf_home = Path(os.getenv("HF_HOME"))
    path = Path(path)
    try:
        return str(path.relative_to(hf_home).parent)
    except ValueError:
        return str(path)

def initialize_model_and_tokenizer_for_training(model_name: str, ft_setting: FT_SETTINGS_TYPE) -> tuple[Any, Any]:
    """Initialize the model and tokenizer based on the specified fine-tuning settings."""

    if model_name in [T5_BASE, T5_BASE_CURICULLUM_STEP2, T5_BASE_CURICULLUM_STEP3, T5_3B, T5_3B_CURICULLUM_STEP2, T5_3B_CURICULLUM_STEP3]:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(TORCH_DEVICE)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        if ft_setting == FT_SETTINGS.new_tokens:
            tokenizer.add_tokens(FOL_LITERALS)
            model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer
    elif model_name in [FLAN_T5_XXL]:
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=bfloat16)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        if ft_setting == FT_SETTINGS.new_tokens:
            tokenizer.add_tokens(FOL_LITERALS)
            model.resize_token_embeddings(len(tokenizer))
            #Saving the full model with new embeddings for LoRA fine-tuning later 
            model.save_pretrained(os.path.join(os.getenv("HF_HOME"), model_name, "base_model_with_embeddings"))
            tokenizer.save_pretrained(os.path.join(os.getenv("HF_HOME"), model_name, "base_model_with_embeddings"))
        return model, tokenizer
    elif model_name in [META_LLAMA_8B, MISTRAL_24B, OLMO_32B]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if (tokenizer.pad_token is None):
            tokenizer.pad_token = tokenizer.eos_token  
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            device_map="auto",  
            trust_remote_code=True, 
            torch_dtype=bfloat16,
        )
        if model_name in [MISTRAL_24B, OLMO_32B]:
            model.gradient_checkpointing_enable()
        if ft_setting == FT_SETTINGS.new_tokens:
            tokenizer.add_tokens(FOL_LITERALS)
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            #Saving the full model with new embeddings for LoRA fine-tuning later
            model.save_pretrained(os.path.join(os.getenv("HF_HOME"), model_name, "base_model_with_embeddings"))
            tokenizer.save_pretrained(os.path.join(os.getenv("HF_HOME"), model_name, "base_model_with_embeddings"))
        return model, tokenizer

    elif model_name in [FLAN_T5_XXL_CURICULLUM_STEP2, FLAN_T5_XXL_CURICULLUM_STEP3]:
        tokenizer = T5Tokenizer.from_pretrained(extract_base_model(model_name))
        model = T5ForConditionalGeneration.from_pretrained(
            extract_base_model(model_name),
            device_map="auto",
            torch_dtype=bfloat16,
        )
        model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
        model.train()
        return model, tokenizer
        
    elif model_name in [META_LLAMA_8B_CURICULLUM_STEP2, META_LLAMA_8B_CURICULLUM_STEP3, MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3]:
        tokenizer = AutoTokenizer.from_pretrained(extract_base_model(model_name), trust_remote_code=True)
        if (tokenizer.pad_token is None):
            tokenizer.pad_token = tokenizer.eos_token  
            
        model = AutoModelForCausalLM.from_pretrained(
            extract_base_model(model_name),
            use_cache=False,
            device_map="auto",  
            trust_remote_code=True, 
            torch_dtype=bfloat16,
        )
        model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
        model.train()
        if model_name in [MISTRAL_24B_CURICULLUM_STEP2, MISTRAL_24B_CURICULLUM_STEP3, OLMO_32B_CURICULLUM_STEP2, OLMO_32B_CURICULLUM_STEP3]:
            model.gradient_checkpointing_enable()
        return model, tokenizer

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


