import os
os.environ["HF_HOME"] = "D:\\"
from pathlib import Path

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


print (extract_base_model("google-t5/t5-base"))
