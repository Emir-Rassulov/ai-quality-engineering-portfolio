# Machine Learning Fundamentals for QA

## 1. Train/Test Split
Train/test split divides a dataset into training and testing subsets. The model learns from the training data, while the test data evaluates how well it generalizes to unseen examples.
In this project, 75% of the 592 chat logs are used for training and 25% for testing. Stratification preserves the proportion of failures in both subsets.

## 2. Overfitting
Overfitting happens when a model learns the training data too closely, including noise or accidental patterns, and performs poorly on unseen data.
In QA, this is similar to a test suite that passes known examples but fails on new prompts. A large gap between training and test performance can indicate overfitting.

## 3. Data Leakage
Data leakage occurs when training or evaluation uses information that would not legitimately be available at prediction time. It can make model performance look better than it really is.
For failure prediction, output tokens, response text, error messages, and post-call judge scores may reveal the outcome. A realistic model should use only features available before the prediction is made.

## 4. Class Imbalance
Class imbalance means that some target classes occur much less often than others. Accuracy can be misleading when the model mostly predicts the majority class.
In this dataset, 544 of 592 calls succeed and only 48 fail. A model that predicts every call as successful achieves about 91.89% accuracy but detects zero failures.

## 5. Confusion Matrix
A confusion matrix compares predicted classes with actual classes. True positives (TP) are failures correctly detected; false positives (FP) are successful calls incorrectly flagged; false negatives (FN) are failures missed by the model; true negatives (TN) are successful calls correctly identified.
For QA monitoring, false negatives may allow real failures to go unnoticed, while false positives create unnecessary investigation work.

## 6. Precision, Recall, and F1
Precision = TP / (TP + FP): the proportion of predicted failures that are actual failures. Recall = TP / (TP + FN): the proportion of actual failures the model detects. F1 is the harmonic mean of precision and recall.
High recall is useful when missing a failure is costly. Higher precision reduces false alarms. The right balance depends on the operational cost of missed failures versus additional investigations.

## 7. Cross-Validation
Cross-validation splits the dataset into several folds and repeatedly trains and evaluates the model on different subsets. It provides a more robust view of performance than relying on one split.
With five-fold cross-validation, each fold is used for evaluation once. For this dataset, recall should be reported across folds because detecting the minority failure class is more important than accuracy alone.

## 8. Experiments on the chat logs

### Setup
- Data: 592 cleaned rows, 48 real failures (8.11%). Real failure means a logged error OR a silent failure (blank response and blank error).
- Split: 75/25, seed 42, stratified. Test set: 148 rows, 12 failures.
- Model: logistic regression with `class_weight="balanced"`; imputing, scaling and encoding live inside one `Pipeline`, so nothing from the test set reaches training.

### Baseline
A dummy model that always predicts "success" gets 91.89% accuracy (136 of 148) and recall 0.00. Accuracy looks excellent and the model finds nothing. Because only 8% of rows fail, accuracy cannot be used to judge this problem.

### Leakage experiment

| Run | Features | Test P | Test R | CV P | CV R |
|---|---|---|---|---|---|
| 1 (leaky) | model, category, input_tokens, output_tokens, latency_ms, judge_score, response_length | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 (honest) | model, category, prompt_length, input_tokens | 0.127 | 0.583 | 0.096 | 0.396 |

(P = precision, R = recall, both for the failure class.)

Run 1 is perfect because it reads the answer from the features. The test is: "would I know this value before the call?"
- `response_length`: 0 means a blank response, which is the definition of a silent failure.
- `judge_score`: produced by the judge after reading the response.
- `output_tokens` and `latency_ms`: measured during or after the call.

Run 2 uses only features known before the call. A perfect score on real data is a warning sign, not a success.

### Duplicate prompts
There are 0 duplicate prompts, so a random split cannot leak the same prompt into train and test, and a grouped split is not needed. Note: prompts are generic ("Prompt N"), so `prompt_length` has only 3 distinct values and carries almost no information.

### Threshold choice
The decision threshold is chosen on cross-validated probabilities from the TRAIN set only, never on the test set; otherwise the test set stops being unseen. Assumption: a missed failure costs 10 times a false alarm (a missed failure reaches a user, a false alarm costs a short look). Chosen threshold: 0.60.

### Results on the untouched test set (threshold 0.60)

|  | Predicted success | Predicted failure |
|---|---|---|
| Actually success | 117 | 19 |
| Actually failure | 10 | 2 |

- Accuracy 0.804 (lower than the dummy's 0.919, which is fine: accuracy is the wrong measure here)
- Precision 0.095, recall 0.167, F1 0.121
- Recall 95% bootstrap interval: [0.00, 0.43], resting on only 12 failures
- 5-fold CV recall: mean 0.400, std 0.213, folds 0.40, 0.00, 0.56, 0.44, 0.60. One fold found no failures at all, which shows how unstable the result is with so few failures.

### Cost check
Cost on the test set: model 119, do nothing 120, flag everything 136. The model saves 1%, so it is not meaningfully better than doing nothing (the bar is 10%).

### Conclusion
With pre-call features alone the model is weak. Precision is about 1.2 times the base failure rate, and it does not beat "do nothing" on cost. It is not production-ready. Better features (real prompt text, request context) and more failure examples are needed before this can work.
