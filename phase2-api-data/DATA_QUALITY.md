# Data Quality Findings

## 1. Dataset Overview

| Metric | Result |
|---|---:|
| Raw rows | 612 |
| Redundant duplicate rows removed | 12 |
| Rows after deduplication | 600 |
| Invalid latency rows excluded after deduplication | 8 |
| Final clean analytical dataset | 592 |
| Models | `gpt-5`, `gpt-5-mini` |
| Prompt categories | `factual`, `creative`, `adversarial`, `out_of_scope` |

A one-off validation check compared the SQL and Python cleaning outputs. Both produced 592 rows with matching IDs and no differences across the 11 compared columns. No dedicated comparison script or automated regression test currently reproduces this check.

## 2. Duplicate IDs

**Observation**

- 12 IDs occur twice in the raw dataset.
- All 12 duplicate pairs contain identical rows.
- Removing one redundant row from each pair reduces the dataset from 612 to 600 rows.

**Impact**

Duplicate records can inflate request counts and distort latency, error-rate, and quality metrics.

**Recommended action**

Confirm whether duplicate IDs are expected in the source system. If IDs should uniquely identify requests, investigate the ingestion process. Remove exact duplicate rows before analysis.

## 3. Missing and Blank Values

SQLite imported empty CSV fields as empty strings, so SQL `NULL` counts are zero. Blank values must therefore be checked explicitly.

| Field | Blank values |
|---|---:|
| `response` | 50 |
| `input_tokens` | 40 |
| `output_tokens` | 40 |
| `judge_score` | 40 |
| `error` | 572 |

**Impact**

Blank responses require further classification. Some requests have recorded errors, while 10 have neither a response nor a recorded error.

**Recommended action**

Treat blank and `NULL` values explicitly in SQL filters and reporting. Do not interpret a blank error field alone as proof of success.

## 4. Negative Latency Sentinel Values

**Observation**

- 9 raw rows have `latency_ms = -1`.
- One duplicate pair consists of two rows with negative latency values.
- After deduplication, 8 rows with negative latency remain.
- These 8 rows are excluded from the clean analytical dataset.

**Impact**

A negative latency is not a valid elapsed-time measurement and would distort latency metrics if treated as a real value.

**Recommended action**

Confirm what `-1` represents in the source system. Exclude sentinel values from latency analysis and retain the raw records for traceability.

**Cleaning reconciliation**

612 raw rows - 12 redundant duplicate rows - 8 invalid-latency rows = 592 clean rows.

## 5. Text Inconsistencies

**Observation**

- 106 rows have non-normalized `category` values, including casing variants and leading/trailing whitespace.
- 32 rows have non-normalized `model` values, including casing variants such as `GPT-5` and `GPT-5-MINI`.
- The raw-data check found no model whitespace inconsistencies.

**Impact**

Grouping on unnormalized text can split the same logical model or category into separate reporting groups, producing misleading counts and comparisons.

**Recommended action**

Normalize model and category values before grouping:

`LOWER(TRIM(model))`

`LOWER(TRIM(category))`

The clean SQL view applies this normalization.

## 6. Silent Failures and Unreliable Judge Scores

**Observation**

10 requests have both a blank response and a blank error. Their IDs are:

`50, 53, 64, 97, 135, 139, 309, 316, 464, 527`

Seven of these 10 requests have a `judge_score` of 4 or 5, despite having no response to evaluate.

**Impact**

These records may appear successful if monitoring relies only on the error field. High judge scores for empty responses can also inflate apparent answer quality and undermine evaluation metrics.

**Recommended action**

- Classify these records as silent failures, not successful calls.
- Exclude them from response-quality scoring.
- Investigate why the judge assigned scores when responses were empty.
- Review the logging and evaluation pipelines to ensure missing responses are handled explicitly.

## 7. Latency Outliers Among Successful Calls

**Observation**

The three slowest calls classified as successful by their response and error fields are:

| Request ID | Model | Category | Latency |
|---:|---|---|---:|
| 213 | `gpt-5` | `creative` | 240,625 ms (240.6 s) |
| 412 | `gpt-5` | `adversarial` | 170,620 ms (170.6 s) |
| 275 | `gpt-5` | `creative` | 90,097.5 ms (90.1 s) |

The dataset also contains a logged error message, `TimeoutError: no response within 30s`.

**Impact**

These unusually high latency values can substantially affect average latency and may indicate a timing, logging, retry, or timeout-configuration inconsistency. The available records do not establish the cause.

**Recommended action**

- Verify how latency is measured and whether it includes retries or queueing.
- Check the effective timeout and retry configuration.
- Correlate these request IDs with application and upstream logs if available.
- Report median and percentile latency alongside averages.
- Keep successful-call latency metrics separate from failed-call latency metrics.

## 8. Error Classification

**Observation**

The error-type analysis distinguishes successful calls from silent failures and requests with logged errors:

| Result type | Requests |
|---|---:|
| Successful calls | 544 |
| Silent failures | 10 |
| `InternalServerError: Internal server error` | 11 |
| `TimeoutError: no response within 30s` | 11 |
| `RateLimitError: Too many requests` | 8 |
| `RuntimeError: Invalid request` | 8 |
| **Total** | **592** |

A successful call requires a non-blank response and a blank error field.

**Impact**

Treating every blank error as success would incorrectly classify the 10 silent failures and overstate reliability.

**Recommended action**

Keep silent failures as a separate result type. Use explicit response and error checks when calculating success rates.

## 9. Cleaning Decisions

1. Normalize model and category values with `LOWER(TRIM(...))`.
2. Remove exact duplicate rows.
3. Exclude rows with missing or negative latency from the clean analytical view.
4. Keep silent failures identifiable for reliability analysis.
5. Exclude empty responses from answer-quality scoring.
6. Classify requests using both response and error fields, rather than error presence alone.
7. Analyze successful-call latency separately from failed-call latency.
8. Preserve the raw dataset so that cleaning decisions remain traceable.

## 10. Follow-up Investigations

- Confirm whether request IDs must be globally unique.
- Investigate all 10 silent failures and their judge scores.
- Review the latency outliers for IDs 213, 412, and 275.
- Confirm whether timeout and retry behaviour matches the API contract.
- Validate that the judge cannot assign a quality score to an empty response.
- Continue comparing SQL and Python cleaning outputs when either pipeline changes.
