"""Step 1: the dummy baseline. Any real model must beat this on failure detection."""
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score

from ml_data import TARGET, load_data, split_data


def run_baseline() -> dict:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, features=["id"])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "rows": len(df),
        "failures": int(df[TARGET].sum()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_failures": int(y_test.sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "predicted_failures": int(predictions.sum()),
    }


def main() -> None:
    r = run_baseline()
    print(f"Total rows: {r['rows']}")
    print(f"Failures: {r['failures']} ({r['failures'] / r['rows']:.2%})")
    print(f"Train rows: {r['train_rows']}")
    print(f"Test rows: {r['test_rows']}")
    print(f"Test failures: {r['test_failures']}")
    print(f"Baseline accuracy: {r['accuracy']:.2%}")
    print(f"Baseline recall:   {r['recall']:.2%}  (it never predicts a failure)")


if __name__ == "__main__":
    main()
