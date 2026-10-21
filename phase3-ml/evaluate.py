"""Step 3: honest evaluation of the pre-call failure predictor."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from ml_data import (
    FORBIDDEN_FEATURES,  # noqa: F401  (re-exported: tests import it from here)
    PRE_CALL_FEATURES,
    RANDOM_STATE,
    TARGET,
    load_data,
    split_data,
)
from ml_model import assert_no_forbidden_features, build_model

N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 2000

# Cost assumption (change it and re-run): one missed failure is 10x worse than
# one false alarm. A missed failure reaches a user; a false alarm costs a look.
COST_FALSE_NEGATIVE = 10
COST_FALSE_ALARM = 1
MIN_COST_SAVING = 0.10  # the model must beat the best trivial strategy by at least 10%
THRESHOLD_CANDIDATES = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Precision and recall. 0/0 is reported as 0.0 (same as sklearn zero_division=0)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def expected_cost(fp: int, fn: int, cost_fp: float, cost_fn: float) -> float:
    return fp * cost_fp + fn * cost_fn


def choose_threshold(
    y_true,
    probabilities,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_ALARM,
    candidates=THRESHOLD_CANDIDATES,
) -> float:
    """Pick the threshold with the lowest cost. A tie goes to the HIGHER threshold
    (fewer alarms). Call it on data the final test set has never seen."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    best_threshold, best_cost = None, None
    for threshold in candidates:
        predicted = probabilities >= threshold
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())
        cost = expected_cost(fp, fn, cost_fp, cost_fn)
        if best_cost is None or cost < best_cost or (cost == best_cost and threshold > best_threshold):
            best_threshold, best_cost = threshold, cost
    return float(best_threshold)


def bootstrap_recall_interval(y_true, predictions, iterations: int = BOOTSTRAP_ITERATIONS):
    """95% bootstrap interval for recall. Samples with no real failure are skipped
    (recall is undefined there)."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    recalls = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0:
            continue
        recalls.append(recall_score(y_true[idx], predictions[idx], zero_division=0))
    if not recalls:
        raise ValueError("Could not calculate bootstrap recall interval")
    lower, upper = np.percentile(recalls, [2.5, 97.5])
    return float(lower), float(upper)


def build_verdict(precision, recall, base_rate, cost_model, cost_none, cost_all, interval, n_failures):
    """Verdict computed from the numbers, never hardcoded."""
    lift = precision / base_rate if base_rate else 0.0
    lines = [
        f"Recall {recall:.2f} vs baseline 0.00 (the baseline never flags a failure).",
        f"Precision {precision:.2f} is {lift:.1f}x the base rate ({base_rate:.2f}): "
        f"{'alerts are informative' if lift >= 3 else 'most alerts are false alarms'}.",
    ]
    best_trivial = min(cost_none, cost_all)
    saving = 1 - cost_model / best_trivial if best_trivial else 0.0
    meaningful = saving >= MIN_COST_SAVING
    if meaningful:
        lines.append(
            f"Cost {cost_model:.0f} is {saving:.0%} lower than the best trivial strategy ({best_trivial:.0f})."
        )
    else:
        lines.append(
            f"Cost {cost_model:.0f} vs {best_trivial:.0f} for the best trivial strategy "
            f"(do nothing {cost_none:.0f}, flag everything {cost_all:.0f}): "
            f"no meaningful saving ({saving:.0%}, needs {MIN_COST_SAVING:.0%})."
        )
    lines.append(
        f"Recall 95% interval [{interval[0]:.2f}, {interval[1]:.2f}] rests on only "
        f"{n_failures} test failures: treat it as a range, not a number."
    )
    usable = lift >= 3 and recall >= 0.5 and meaningful
    lines.append(
        "VERDICT: usable as an alert." if usable
        else "VERDICT: not production-ready. At best a triage hint; needs more failures and better features."
    )
    return lines


def _scorers():
    return {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }


def run_evaluation(df=None) -> dict:
    assert_no_forbidden_features(PRE_CALL_FEATURES)
    df = load_data() if df is None else df

    X_train, X_test, y_train, y_test = split_data(df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 1. Choose the threshold from TRAIN-only cross-validated probabilities.
    train_probabilities = cross_val_predict(
        build_model(), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, train_probabilities)

    # 2. Fit on train, evaluate ONCE on the untouched test set.
    model = build_model().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    precision, recall = precision_recall(int(tp), int(fp), int(fn))

    # 3. Cross-validation over all rows (spread, not one lucky split).
    X_all, y_all = df[PRE_CALL_FEATURES], df[TARGET]
    cv_results = cross_validate(build_model(), X_all, y_all, cv=cv, scoring=_scorers(), error_score="raise")

    # 4. Reference costs on the test set.
    n_pos, n_neg = int(y_test.sum()), int((y_test == 0).sum())
    cost_model = expected_cost(int(fp), int(fn), COST_FALSE_ALARM, COST_FALSE_NEGATIVE)
    cost_none = expected_cost(0, n_pos, COST_FALSE_ALARM, COST_FALSE_NEGATIVE)
    cost_all = expected_cost(n_neg, 0, COST_FALSE_ALARM, COST_FALSE_NEGATIVE)

    interval = bootstrap_recall_interval(y_test, predictions)
    base_rate = float(y_all.mean())

    return {
        "features": list(PRE_CALL_FEATURES),
        "rows": len(df),
        "failures": int(y_all.sum()),
        "base_rate": base_rate,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_failures": n_pos,
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": precision,
        "recall": recall,
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "recall_interval": interval,
        "cv_precision": cv_results["test_precision"],
        "cv_recall": cv_results["test_recall"],
        "cv_f1": cv_results["test_f1"],
        "cost_model": cost_model,
        "cost_none": cost_none,
        "cost_all": cost_all,
        "verdict": build_verdict(precision, recall, base_rate, cost_model, cost_none, cost_all, interval, n_pos),
    }


def print_report(r: dict) -> None:
    print("=== DATASET ===")
    print(f"Rows: {r['rows']}  Failures: {r['failures']} ({r['base_rate']:.2%})")
    print(f"Train: {r['train_rows']}  Test: {r['test_rows']} (failures in test: {r['test_failures']})")
    print(f"Features: {r['features']}")

    print(f"\n=== THRESHOLD (chosen on train CV only; cost FN={COST_FALSE_NEGATIVE}, FP={COST_FALSE_ALARM}) ===")
    print(f"Chosen threshold: {r['threshold']:.2f}")

    print("\n=== CONFUSION MATRIX (test; rows = actual, columns = predicted) ===")
    print("              Predicted 0   Predicted 1")
    print(f"Actual 0      {r['tn']:>11}   {r['fp']:>12}")
    print(f"Actual 1      {r['fn']:>11}   {r['tp']:>12}")

    print("\n=== TEST METRICS ===")
    print(f"Accuracy:  {r['accuracy']:.3f}")
    print(f"Precision: {r['precision']:.3f}")
    print(f"Recall:    {r['recall']:.3f}")
    print(f"F1:        {r['f1']:.3f}")
    lo, hi = r["recall_interval"]
    print(f"Recall 95% bootstrap interval: [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== {N_SPLITS}-FOLD CROSS-VALIDATION (all rows) ===")
    for name in ("precision", "recall", "f1"):
        values = r[f"cv_{name}"]
        print(f"{name:>9}: mean {values.mean():.3f}  std {values.std():.3f}  folds {np.round(values, 3)}")

    print("\n=== COST ON TEST SET ===")
    print(f"Model:               {r['cost_model']:.0f}")
    print(f"Do nothing:          {r['cost_none']:.0f}")
    print(f"Flag everything:     {r['cost_all']:.0f}")

    print("\n=== VERDICT ===")
    for line in r["verdict"]:
        print(line)


def main() -> None:
    print_report(run_evaluation())


if __name__ == "__main__":
    main()
