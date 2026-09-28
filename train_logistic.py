import argparse
from pathlib import Path

from logistic_model import DEFAULT_MODEL_PATH, load_training_rows, train_logistic_model


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression text classifier from processed STT logs."
    )
    parser.add_argument("--input", type=Path, default=Path("output/processed_log_2.csv"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)

    texts, labels = load_training_rows(args.input)
    recognizer = train_logistic_model(texts, labels, args.model_path)
    positives = sum(labels)
    negatives = len(labels) - positives
    print(
        f"Trained Logistic Regression model at {recognizer.model_path} "
        f"from {len(labels)} rows ({positives} match, {negatives} no_match)."
    )


if __name__ == "__main__":
    main()
