# Break-It Log: Phase 3 ML pipeline

Purpose: prove the tests can fail. Each break was applied on purpose, the suite was run,
and the code was restored. After each restore the suite returned to 49 passed.

## Break 1: a post-call feature leaks into the model

**Change:** added `"judge_score"` to `NUMERIC_FEATURES` in `ml_data.py`.

**Result:** 5 failed, 44 passed.

- `test_pre_call_features_contain_no_forbidden_features`
- `test_metrics_are_deterministic_across_runs`
- `test_confusion_matrix_adds_up_to_the_test_set`
- `test_cv_recall_stays_above_floor`
- `test_model_recall_is_above_baseline_recall`

Error: `ValueError: Forbidden post-call features: ['judge_score']`

**Why it matters:** the judge score exists only after the model answered. Using it makes the
failure predictor look perfect while being useless before a call. The guard catches it both in
the unit test and when the pipeline runs.

## Break 2: stratification removed from the split

**Change:** deleted the line `stratify=df[TARGET],` in `split_data` in `ml_data.py`.

**Result:** 4 failed, 45 passed.

- `test_baseline_accuracy_looks_good_but_is_useless`
- `test_test_split_contains_expected_failures`
- `test_split_is_stratified` (`assert 11 == 12`)
- `test_confusion_matrix_adds_up_to_the_test_set`

**Why it matters:** without stratification the test set got 11 failures instead of 12. With only
48 failures in total, one failure more or less changes recall visibly, so results become
unstable and not comparable between runs.

## How to reproduce

From `phase3-ml`:

1. `cp ml_data.py ml_data.py.bak`
2. Apply the change (Break 1: `sed -i '' 's/NUMERIC_FEATURES = \["prompt_length", "input_tokens"\]/NUMERIC_FEATURES = ["prompt_length", "input_tokens", "judge_score"]/' ml_data.py`;
   Break 2: `sed -i '' '/stratify=df\[TARGET\],/d' ml_data.py`)
3. `pytest -q`
4. `mv ml_data.py.bak ml_data.py` and run `pytest -q` again (expect `49 passed`)

(The `sed -i ''` form is for macOS; on Linux use `sed -i` without the empty quotes.)
