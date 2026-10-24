"""Download UCI SMS data, deduplicate, train, and evaluate once."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_data(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.write_bytes(archive.read("SMSSpamCollection"))
        (path.parent / "UCI-readme.txt").write_bytes(archive.read("readme"))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label, message = line.split("\t", 1)
        if label not in {"ham", "spam"} or not message.strip():
            raise ValueError("Dataset contains an invalid label or empty message")
        rows.append((label, message))
    # Match equivalent whitespace and case so these cannot cross the split.
    unique = {}
    conflicts = set()
    for label, message in rows:
        key = " ".join(message.casefold().split())
        if key in unique and unique[key][0] != label:
            conflicts.add(key)
        unique.setdefault(key, (label, message))
    clean = [row for key, row in unique.items() if key not in conflicts]
    return clean, len(rows), len(conflicts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/SMSSpamCollection")
    args = parser.parse_args()
    rows, raw_count, conflicts = load_data(args.data)
    labels, messages = zip(*rows)
    x_train, x_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # Fixed settings chosen before evaluating the held-out set.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", MultinomialNB(alpha=0.1)),
    ])
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predicted, labels=["ham", "spam"])
    result = {
        "source": URL, "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "raw_messages": raw_count, "unique_messages": len(rows),
        "removed_messages": raw_count - len(rows), "conflicting_keys_removed": conflicts,
        "train_messages": len(x_train), "test_messages": len(x_test),
        "random_state": 42, "test_fraction": 0.2, "sklearn_version": sklearn.__version__,
        "classification_report": report, "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": matrix.tolist(), "spam_f1_target_met": report["spam"]["f1-score"] >= 0.85,
    }
    (ROOT / "model").mkdir(exist_ok=True)
    (ROOT / "evaluation").mkdir(exist_ok=True)
    joblib.dump(pipeline, ROOT / "model/spam_pipeline.joblib")
    (ROOT / "evaluation/metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    text = classification_report(y_test, predicted, digits=4, zero_division=0)
    (ROOT / "evaluation/report.txt").write_text(text)
    ConfusionMatrixDisplay(matrix, display_labels=["Not Spam", "Spam"]).plot(cmap="Blues", colorbar=False)
    plt.title("Held-out SMS messages • actual vs. predicted")
    plt.tight_layout()
    plt.savefig(ROOT / "evaluation/confusion_matrix.png", dpi=180)
    plt.close()
    print(text)
    print(f"Train: {len(x_train)} | Test: {len(x_test)} | Target met: {result['spam_f1_target_met']}")


if __name__ == "__main__":
    main()
