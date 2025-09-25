import torch
import os
from typing import Literal
from enum import Enum


class FT_SETTINGS(str, Enum):
    standard = "standard"
    new_tokens = "new_tokens"
    gt_predicates = "gt_predicates"
    gt_predicates_noise = "gt_predicates_noise"
    multilingual = "multilingual"
    step_1 = "2step_1"
    curiculum_step1 = "curiculum_step1"
    curiculum_step2 = "curiculum_step2"
    curiculum_step3 = "curiculum_step3"

FT_SETTINGS_TYPE = Literal[
    "standard", "new_tokens", "gt_predicates", "gt_predicates_noise",
    "multilingual", "2step",
    "curiculum_step1", "curiculum_step2", "curiculum_step3"
]


class GENERATTION_SETTINGS(str, Enum):
    standard = "standard"
    new_tokens = "new_tokens"
    gt_predicates = "gt_predicates"
    gt_predicates_noise = "gt_predicates_noise"
    predicates_only = "predicates_only"

GENERATTION_SETTINGS_TYPE = Literal[
    "standard", "new_tokens", "gt_predicates", "gt_predicates_noise", "predicates_only"
]

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOL_LITERALS = ['∀', '∃', '∧', '⊕', '∨', '→', '↔', '¬']
FOL_TOKENS = ['FORALL', 'EXISTS', 'AND', 'XOR', 'OR', 'IMPLIES', 'IFF', 'NOT']

T5_BASE = "google-t5/t5-base"
T5_BASE_CURICULLUM_STEP2 = os.path.join(os.getenv("HF_HOME"), T5_BASE, FT_SETTINGS.curiculum_step2)
T5_BASE_CURICULLUM_STEP3 = os.path.join(os.getenv("HF_HOME"), T5_BASE, FT_SETTINGS.curiculum_step3)
T5_3B = "google-t5/t5-3b"
T5_3B_CURICULLUM_STEP2 = os.path.join(os.getenv("HF_HOME"), T5_3B, FT_SETTINGS.curiculum_step2)
T5_3B_CURICULLUM_STEP3 = os.path.join(os.getenv("HF_HOME"), T5_3B, FT_SETTINGS.curiculum_step3)
FLAN_T5_XXL = "google/flan-t5-xxl"
FLAN_T5_XXL_CURICULLUM_STEP2 = os.path.join(os.getenv("HF_HOME"), FLAN_T5_XXL, FT_SETTINGS.curiculum_step2)
FLAN_T5_XXL_CURICULLUM_STEP3 = os.path.join(os.getenv("HF_HOME"), FLAN_T5_XXL, FT_SETTINGS.curiculum_step3)
META_LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
META_LLAMA_8B_CURICULLUM_STEP2 = os.path.join(os.getenv("HF_HOME"), META_LLAMA_8B, FT_SETTINGS.curiculum_step2)
META_LLAMA_8B_CURICULLUM_STEP3 = os.path.join(os.getenv("HF_HOME"), META_LLAMA_8B, FT_SETTINGS.curiculum_step3)
MISTRAL_24B = "mistralai/Mistral-Small-24B-Instruct-2501"
MISTRAL_24B_CURICULLUM_STEP2 = os.path.join(os.getenv("HF_HOME"), MISTRAL_24B, FT_SETTINGS.curiculum_step2)
MISTRAL_24B_CURICULLUM_STEP3 = os.path.join(os.getenv("HF_HOME"), MISTRAL_24B, FT_SETTINGS.curiculum_step3)
OLMO_32B = "allenai/OLMo-2-0325-32B-Instruct"
OLMO_32B_CURICULLUM_STEP2 = os.path.join(os.getenv("HF_HOME"), OLMO_32B, FT_SETTINGS.curiculum_step2)
OLMO_32B_CURICULLUM_STEP3 = os.path.join(os.getenv("HF_HOME"), OLMO_32B, FT_SETTINGS.curiculum_step3)

EARLY_STOPPING_PATIENCE = 4
BEST_MODEL_METRIC = "eval_loss"
NUM_EPOCHS = 12
STRATEGY = "epoch"
SAVE_LIMIT = 12
LOAD_BEST_MODEL = True
DISABLE_TQDM = False
REPORT_TO = "none"
LOGGING_STEPS = 100
BF16 = True
WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-8
WARMUP_STEPS = 500
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.05  # Alternative to WARMUP_STEPS; only one should be used
DDP_FIND_UNUSED_PARAMETERS = False
DDP_BACKEND = "nccl"

CFG_FOL = """
%import common.WS
%import common.UNICODE_LETTER
%import common.UNICODE_DIGIT
%ignore WS

VARIABLE: /[a-z]/
QUANTIFIER: "∀" | "∃"
CONSTANT: /(?![a-z]$)[^\\s,)]+/
NAME: /\\w{2,}/

?start: expr

?expr: implies

?implies: iff
       | iff "→" implies -> implies

?iff: xor
    | xor "↔" iff -> iff

?xor: and_
    | and_ "⊕" xor -> xor

?and_: or_
     | or_ "∧" and_ -> and_

?or_: not_
    | not_ "∨" or_ -> or_

?not_: atom
     | "¬" not_ -> not_

?atom: "(" expr ")"
     | predicate
     | constant
     | variable
     | quantified

quantified: QUANTIFIER variable expr -> quantified

predicate: NAME "(" [args] ")" -> predicate
args: expr ("," expr)*          -> args

constant: CONSTANT          -> constant
variable: VARIABLE          -> variable
"""
