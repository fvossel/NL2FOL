import os
os.environ["HF_HOME"] = "<Directory for Model Cache>"
os.environ["TMPDIR"] = "<Directory for Temp Files>"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from utils.modelloader import initialize_model_and_tokenizer_for_generation
from utils.datasetloader import load_dataset_for_generation
from utils.generation import generate


def main(args):
    model, tokenizer = initialize_model_and_tokenizer_for_generation(args.model_name)
    test_dataset = load_dataset_for_generation(args.model_name, args.generation_setting, tokenizer)
    generate(model, tokenizer, test_dataset, args.generation_setting)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation Script for NL2FOL")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model')
    parser.add_argument('--generation_setting', type=str, required=True, help='Generation setting')
    args = parser.parse_args()
    main(args)