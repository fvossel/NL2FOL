import argparse
from utils.modelloader import initialize_model_and_tokenizer_for_training
from utils.datasetloader import load_datasets_for_training
from utils.training import train


def main(args):
    model, tokenizer = initialize_model_and_tokenizer_for_training(args.model_name, ft_setting=args.ft_setting)
    train_dataset, validation_dataset = load_datasets_for_training(args.model_name, args.ft_setting, tokenizer)
    train(model, tokenizer, args.model_name, args.ft_setting, train_dataset, validation_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script for NL2FOL")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument("--ft_setting", type=str, required=True, help="Fine-tuning setting")
    args = parser.parse_args()
    main(args)