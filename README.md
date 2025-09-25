# NL2FOL: Fine-Tuning LLMs for First-Order Logic Formalization
This repository contains the code used for fine-tuning large language models (LLMs) for the conversion of natural language statements into first-order predicate logic (FOL), as described in my research paper. The goal is to support and automate the formalization of natural language into FOL representations, enabling further applications in logic-based AI and natural language understanding.

The weights of the fine-tuned models can be found [here](https://huggingface.co/collections/fvossel/nl-to-fol-685464200cad67e2cd5b0e73).

We combined the [MALLS](https://arxiv.org/abs/2305.15541) dataset and the [Willow](https://open.metu.edu.tr/handle/11511/109445) dataset.
If you use these datasets, please make sure to cite the respective works.

---

Das kannst du direkt im README verwenden. Die Links und BibTeX-Einträge funktionieren im Markdown. Wenn du noch Anpassungen möchtest, sag gerne Bescheid!
## Getting Started

To get a local copy of the project up and running, execute the following commands in your terminal:

```bash
git clone https://github.com/fvossel/NL2FOL
cd NL2FOL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export HF_HOME=<Cache path for the Huggingface models>
export TMPDIR=<Hugginface temp files>
export TOKENIZERS_PARALLELISM=false
```

This sets up a Python virtual environment and installs all required dependencies.

## Example Usage
Training the Google T5 model:
```bash
python train.py --model_name "google-t5/t5-base" --ft_setting "standard"
```
Testing the fine-tuned model:
```bash
python generate.py --model_name "${HF_HOME}/google-t5/t5-base/standard" --generation_setting "standard" --batch_size=32
```


## Citation

If you use this code for scientific purposes, **please cite the following paper**:

```
@inproceedings{your-paper-citation,
  author = {Your Name},
  title = {Your Paper Title},
  booktitle = {Conference/Journal Name},
  year = {2024},
  ...
}
```

---

Let me know if you want to add more sections (usage/examples/contributing/etc.) or if you have the citation details!
