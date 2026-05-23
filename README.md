# NL2FOL: Fine-Tuning LLMs for First-Order Logic Formalization

> [!WARNING]
> **This repository has been archived and is no longer actively maintained.**
> Please use the new repository instead:
> ➜ [https://github.com/fvossel/ANLFtFOLwFLLMs](https://github.com/fvossel/ANLFtFOLwFLLMs)

---

This repository contains the code used for fine-tuning large language models (LLMs) for the conversion of natural language statements into first-order predicate logic (FOL), as described in my research paper. The goal is to support and automate the formalization of natural language into FOL representations, enabling further applications in logic-based AI and natural language understanding.

The weights of the fine-tuned models can be found here: [https://huggingface.co/collections/fvossel/nl-to-fol-685464200cad67e2cd5b0e73](https://huggingface.co/collections/fvossel/nl-to-fol-685464200cad67e2cd5b0e73)

We combined the [MALLS](https://arxiv.org/abs/2305.15541) dataset and the [Willow](https://open.metu.edu.tr/handle/11511/109445) dataset.
If you use these datasets, please make sure to cite the respective works.

---

## Getting Started

To get a local copy of the project up and running, execute the following commands in your terminal:

```bash
git clone https://github.com/fvossel/NL2FOL
cd NL2FOL

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

export HF_HOME=<Cache path for the Huggingface models>
export TMPDIR=<Huggingface temp files>
export TOKENIZERS_PARALLELISM=false
```

This sets up a Python virtual environment and installs all required dependencies.

---

## Example Usage

### Training the Google T5 model

```bash
python train.py --model_name "google-t5/t5-base" --ft_setting "standard"
```

### Testing the fine-tuned model

```bash
python generate.py --model_name "${HF_HOME}/google-t5/t5-base/standard" --generation_setting "standard" --batch_size=32
```

### Multi-GPU Training (torchrun)

```bash
torchrun --nproc_per_node=2 train.py --model_name "mistralai/Mistral-Small-24B-Instruct-2501" --ft_setting "standard"
```

---

## Citation

If you use this code for scientific purposes, **please cite the following paper**:

```bibtex
@misc{vossel2025advancingnaturallanguageformalization,
      title={Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs},
      author={Felix Vossel and Till Mossakowski and Bj{"o}rn Gehrke},
      year={2025},
      eprint={2509.22338},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22338},
}
```

---

## Demo

A demo of our T5-3b model can be tested here:
[https://translate.hyai.cs.uos.de](https://translate.hyai.cs.uos.de)
