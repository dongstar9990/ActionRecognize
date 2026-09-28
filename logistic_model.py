import csv
from pathlib import Path

from main import normalize_text


DEFAULT_MODEL_PATH = Path("output/models/logistic_text_classifier.joblib")


def load_training_rows(csv_path):
    texts = []
    labels = []
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            status = (row.get("status") or "").strip()
            text = (row.get("best_text") or "").strip()
            if not text or status not in ("match", "no_match"):
                continue
            texts.append(text)
            labels.append(1 if status == "match" else 0)
    return texts, labels


def train_logistic_model(texts, labels, model_path=DEFAULT_MODEL_PATH):
    if not texts or not labels:
        raise ValueError("training data is empty")
    if len(set(labels)) < 2:
        raise ValueError("training data must contain both match and no_match labels")

    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: scikit-learn/joblib. Install requirements before training."
        ) from exc

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    pipeline.fit([normalize_text(text) for text in texts], labels)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return LogisticTextRecognizer(classifier=pipeline, model_path=model_path)


def train_from_processed_log(
    csv_path=Path("output/processed_log_2.csv"),
    model_path=DEFAULT_MODEL_PATH,
):
    texts, labels = load_training_rows(csv_path)
    return train_logistic_model(texts, labels, model_path)


class LogisticTextRecognizer:
    def __init__(self, classifier=None, model_path=DEFAULT_MODEL_PATH):
        self.classifier = classifier
        self.model_path = Path(model_path)

    def load(self):
        if self.classifier is not None:
            return self.classifier
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Logistic model not found at {self.model_path}. "
                "Run training first to create it."
            )
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: joblib. Install requirements before loading the model."
            ) from exc
        self.classifier = joblib.load(self.model_path)
        return self.classifier

    def predict(self, transcript, threshold=0.5, latency_ms=None):
        classifier = self.load()
        text = str(transcript or "")
        probabilities = classifier.predict_proba([normalize_text(text)])[0]
        positive_probability = round(float(probabilities[1]), 6)
        matched = positive_probability >= float(threshold)
        return {
            "matched": matched,
            "label": "contains_information" if matched else "no_information",
            "probability": positive_probability,
            "threshold": float(threshold),
            "transcript": text,
            "latency_ms": latency_ms,
            "model_path": str(self.model_path),
        }
