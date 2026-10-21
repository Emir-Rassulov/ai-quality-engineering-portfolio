# AI Quality Engineering Portfolio

A hands-on portfolio focused on **Quality Engineering for AI-powered systems**, combining traditional test automation with AI/LLM tooling, reliability testing, data analysis and quality-focused engineering practices.

The project is built incrementally to demonstrate practical QA engineering skills and the transition from conventional automation toward **AI Quality Engineering**.

---

## Overview

This repository contains a series of practical QA engineering exercises covering:

- API test automation and API contract testing
- Python and Pytest
- asynchronous execution
- bounded concurrency
- retry and timeout handling
- LLM/AI tooling
- OpenAI SDK integration
- Pydantic validation (including strict contracts)
- structured test data and results
- mock-based testing
- provider-specific error handling
- SQL and Pandas data analysis of LLM logs
- statistical comparison with stated uncertainty
- machine-learning evaluation: baselines, data leakage, cost-based thresholds

The focus is not only on writing tests, but also on building **reliable, maintainable, and testable engineering tools**, and on making sure every test is able to fail.

---

## Project Structure

```text
ai-quality-engineering-portfolio/
│
├── exercise1_api_tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── requirements.txt
│
├── exercise2_prompt_runner/
│   ├── prompt_runner/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── client.py
│   │   ├── models.py
│   │   └── runner.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── fakes.py
│   │   ├── test_runner.py
│   │   └── test_mock_transport_retry.py
│   │
│   ├── prompts.csv
│   └── requirements.txt
│
├── phase2-api-data/
│   ├── api_contract_tests/        Part A: contract and retry tests, BREAK_IT.md
│   ├── sql/                       Part B: 14 SQL files, shared clean view
│   ├── tests/                     Part C: tests for cleaning and reporting
│   ├── common.py                  shared rules (blank values, silent failures)
│   ├── clean_data.py              logged, validated cleaning pipeline
│   ├── report.py                  metrics and statistical comparison
│   ├── load_database.py           CSV to typed SQLite
│   ├── chat_logs.csv              612-row synthetic chat log (messy on purpose)
│   ├── REPORT.md                  one-page report for a product owner
│   └── DATA_QUALITY.md            every data problem found, with numbers
│
├── phase3-ml/
│   ├── tests/                     49 tests: data, metrics, leakage guard, pipeline
│   ├── ml_data.py                 data, target, features and split (one definition)
│   ├── ml_model.py                the model pipeline and the leakage guard
│   ├── baseline.py                dummy baseline
│   ├── leakage_demo.py            leaky vs honest features
│   ├── evaluate.py                threshold, cross-validation, bootstrap, verdict
│   ├── ML_NOTES.md                concepts and experiment results
│   ├── REPORT.md                  one-page report for a product owner
│   └── BREAK_IT.md                code broken on purpose, tests that caught it
│
├── pytest.ini
└── .gitignore
```

---

# Phase 1 — QA Automation Foundations

## Exercise 1 — API Test Automation

A Pytest-based API test suite built against JSONPlaceholder.

### Coverage

- GET requests
- POST requests
- PUT requests
- DELETE requests
- parameterized tests
- response validation
- negative scenarios
- shared API fixtures
- HTTP session reuse

### Technologies

- Python
- Pytest
- Requests

The exercise establishes the basic automation foundation used by the later exercises.

---

## Exercise 2 — AI Prompt Runner CLI

An asynchronous command-line tool for sending prompts to an LLM provider and storing structured results in CSV format.

The tool was designed with QA and reliability considerations in mind rather than as a simple API wrapper.

### Features

- CSV input and output
- Pydantic validation
- asynchronous LLM calls
- bounded concurrency using `asyncio.Semaphore`
- configurable request timeout
- retry handling for transient failures
- per-request error isolation
- structured latency tracking
- token usage tracking
- mock client for offline testing
- OpenAI SDK integration
- meaningful CLI exit codes

### Execution Flow

```text
prompts.csv
     │
     ▼
CSV parsing
     │
     ▼
Pydantic validation
     │
     ▼
Validated PromptRow objects
     │
     ▼
Async execution
     │
     ├── concurrency limit
     ├── timeout handling
     └── retry logic
     │
     ▼
PromptResult objects
     │
     ▼
results.csv
```

---

# Phase 2 — API Contracts, SQL and Pandas

Phase 2 shifts from "does the tool work?" to "can the data and the numbers be trusted?". It works on a 612-row log of chatbot calls and on the retry logic of the Phase 1 runner.

## Part A — API contract tests (`api_contract_tests/`)

48 tests: 29 run offline, 19 call [httpbin](https://httpbin.org) and are marked `network`.

- **Contract tests against httpbin:** basic auth (correct, wrong user, wrong password, no credentials), the `WWW-Authenticate` header, status codes, a client timeout that raises `ReadTimeout`, and a five-line stream checked in order.
- **Strict Pydantic contract:** a model that rejects missing fields and wrong types for the right reason (`strict=True` refuses `"true"` where a boolean is promised).
- **Retry and error classification:** the real `run_prompts` loop is driven through the real `AsyncOpenAI` client with a mock HTTP transport (`max_retries=0`, so only our code retries). It covers 503 then success, 429 then success, giving up after 3 attempts, 400 not retried, and exhausted quota (`insufficient_quota`) not retried.
- **Break-it evidence:** [`BREAK_IT.md`](phase2-api-data/api_contract_tests/BREAK_IT.md) records code changed on purpose (retries disabled, strict mode off) and the tests that went red.
- **A real finding:** the tests showed that HTTP 429 was missing from the fallback retry rule in `runner.py`. It is fixed.

## Part B — SQL on the chat log (`sql/`)

- `load_database.py` builds a typed SQLite table. Blank numeric fields are stored as `NULL`, not empty text, so `COUNT` and `AVG` behave correctly.
- `00_create_clean_view.sql` defines "clean data" once; every analysis query reads from that view.
- Profiling queries (`01`-`05`): duplicate IDs, blank values, negative latencies, casing and whitespace, silent failures.
- Analysis queries (`06`-`13`): clean row count, error rate by model and category, latency by category, slowest successful calls, result types (including silent failures), judge-score ranking, and a window-function ranking of the slowest calls per model.
- All findings, with counts, are in [`DATA_QUALITY.md`](phase2-api-data/DATA_QUALITY.md).

## Part C — Pandas cleaning and report

- A logged pipeline: normalize text, remove exact duplicates, remove invalid latency, flag silent failures. **612 rows → 600 → 592.**
- `validate_data` raises `ValueError` on any broken rule (not `assert`, which Python can strip out).
- `report.py` computes per-model and per-category metrics, success-only latency, and a bootstrap interval and permutation test for adversarial versus other prompts.
- SQL and Pandas reconcile: both give 592 clean rows and the same error counts.
- 48 tests. Findings for a non-technical reader are in [`REPORT.md`](phase2-api-data/REPORT.md).

### Key findings

| Finding | Number |
| --- | --- |
| Reported error rate | 6.42% (38 of 592) |
| Silent failures (no response and no logged error) | 10, so the real failure rate is 8.11% (48 of 592) |
| Silent failures that still got a judge score of 4 or 5 | 7 of 10 |
| Adversarial vs other prompts, mean judge score | 3.129 vs 4.042 (difference -0.913, 95% interval -1.198 to -0.634, 93 vs 451 scored calls) |
| p95 latency, all calls vs successful calls only | 8,254 ms vs 6,815 ms |
| gpt-5 latency, mean vs median | 8,008 ms vs 4,146 ms |
| "Successful" calls slower than the 30 s timeout | 3 (240 s, 171 s, 90 s), unexplained |

---

# Phase 3 — Machine Learning Evaluation for QA

Phase 3 asks: "Can a model predict LLM failures before they happen, and can we trust the answer?" It uses the cleaned Phase 2 log (592 rows). A **real failure** is a logged error or a silent failure: 48 of 592 (8.11%).

- **Baseline first:** a dummy model that always predicts "success" scores 91.89% accuracy and finds none of the failures. Accuracy is the wrong measure when only 8% of calls fail.
- **Leakage experiment:** a model given post-call fields (response length, judge score, latency, output tokens) looks perfect, because those fields reveal the outcome. A model limited to features known before the call does not.
- **Honest evaluation:** seeded, stratified split; preprocessing inside one pipeline; decision threshold chosen on train-only cross-validation with a stated cost assumption (a missed failure costs 10 false alarms); the test set is used once. Results include 5-fold cross-validation and a bootstrap interval.
- **Verdict computed from the numbers**, not written by hand: the model is not production-ready.
- **49 tests**, including a guard that fails if a post-call feature enters the model, and [`BREAK_IT.md`](phase3-ml/BREAK_IT.md), which records code broken on purpose and the tests that caught it.
- Concepts and results: [`ML_NOTES.md`](phase3-ml/ML_NOTES.md). Report for a non-technical reader: [`REPORT.md`](phase3-ml/REPORT.md).

### Key findings

| Finding | Number |
| --- | --- |
| Dummy baseline (always "success") | 91.89% accuracy, 0 of 12 test failures found |
| Model with post-call features (leakage) | precision 1.000, recall 1.000 (not real) |
| Model with pre-call features, 5-fold cross-validation | precision 0.096, recall 0.396 |
| Same model on the test set (threshold 0.60) | precision 0.095, recall 0.167 (2 of 12 failures) |
| Recall 95% bootstrap interval | 0.00 to 0.43, resting on 12 failures |
| Cost on the test set (a miss counts as 10 false alarms) | model 119, do nothing 120, flag everything 136 |
| Verdict | not production-ready; no meaningful saving over doing nothing |

---

## Technical Stack

| Area            | Technology                    |
| --------------- | ----------------------------- |
| Language        | Python                        |
| Test framework  | Pytest, pytest-asyncio        |
| Validation      | Pydantic (strict mode)        |
| CLI             | Typer                         |
| HTTP / API      | OpenAI SDK, httpx2            |
| Async execution | asyncio                       |
| Test isolation  | Mock transport, fake clients  |
| Data analysis   | Pandas, NumPy                 |
| Machine learning| scikit-learn (pipelines)      |
| Database        | SQLite, SQL (views, windows)  |
| Data format     | CSV                           |
| Version control | Git / GitHub                  |

---

## Key Engineering Decisions

### Bounded concurrency

LLM requests are executed asynchronously while an `asyncio.Semaphore` prevents the number of active requests from exceeding the configured limit.

This allows the tool to benefit from concurrency without creating uncontrolled request volume.

### Per-request failure isolation

A failed request is converted into a structured `PromptResult` containing the error instead of terminating the entire batch.

This means one failed prompt does not prevent the remaining prompts from being processed.

### Retry only transient failures

Retries are limited to failures that may reasonably succeed on a subsequent attempt, such as:

- timeouts
- connection errors
- HTTP 5xx errors
- retryable rate-limit errors (HTTP 429)

Non-retryable failures are returned immediately.

### Provider-specific error handling

The OpenAI SDK normalizes provider errors into typed exceptions and exposes fields such as `error.code`.

The implementation uses the SDK-provided error code to distinguish quota exhaustion (`insufficient_quota`) from retryable rate limiting.

This prevents exhausted API quota from being incorrectly retried.

### Real SDK regression testing

Retry and quota behavior is tested through the actual `AsyncOpenAI` client using a mocked HTTP transport.

This verifies the application's assumptions against the SDK's real error parsing behavior rather than relying only on manually constructed fake exceptions.

### Mock mode

The CLI supports a mock client so that the complete workflow can be tested without:

- an API key
- network access
- API costs
- external service availability

### Tests must be able to fail

Phase 2 tests were verified by breaking the code on purpose and recording which test went red (`BREAK_IT.md`). A test that cannot fail is treated as a defect.

### One definition of each rule

The silent-failure rule lives in `common.py`, and "clean data" lives in one SQL view. Cleaning, reporting and tests all use the same definitions, and SQL and Pandas are reconciled against each other.

### Evaluate against a baseline

A model is only reported next to a trivial one (always predict "success", or do nothing). On imbalanced data the baseline already scores 91.89% accuracy, so accuracy alone says nothing.

### No data leakage

Only features known before the call may be used. A guard in the code and a test fail if a post-call field (response, error, judge score, latency, output tokens) enters the model, and the threshold is chosen on training data only.

### Honest metrics

Latency is reported as p50 and p95, and failed calls are kept out of success latency. Silent failures are counted separately and excluded from judge-score averages. Comparisons state sample sizes and an interval, not just a difference.

---

## Testing

Phase 1 tests cover:

- valid CSV input
- malformed CSV rows
- extra CSV columns
- successful LLM calls
- failed calls without aborting the batch
- concurrency limits
- retryable server errors
- rate-limit handling
- timeout retries
- exhausted retries
- non-retryable errors
- CLI mock execution
- CLI exit codes
- temperature validation
- OpenAI SDK `insufficient_quota` behavior

Phase 2 tests cover:

- API contracts (auth, status codes, timeouts, streaming) and strict response shape
- retry and error classification through the real SDK
- cleaning order, logging and every validation rule
- metrics, success-only rules, the statistical comparison and its edge cases

Phase 3 tests cover:

- target construction, required columns, empty and single-class data
- precision and recall, including the zero-division cases
- the leakage guard, threshold choice, bootstrap interval and computed verdict
- determinism, a stratified and disjoint split, and a performance floor on the real data

Tests are designed to validate both **functional behavior** and **failure handling**.

---

## Limits and Real API Status

- The log data is **synthetic** and messy on purpose.
- The LLM judge scores were **never validated** against human ratings, and some silent failures received high scores. Judge-score results are therefore indicative only.
- The adversarial result is a **correlation**: adversarial prompts are also harder, so the gap does not prove a cause.
- Phase 3 rests on only **48 failures** (12 in the test set), so its results are wide ranges, not exact figures. Prompts in the synthetic log are generic ("Prompt N"), so prompt length carries almost no information.
- The API contract tests use the public httpbin.org server, which can be slow or unavailable. Point them at a local copy with the `HTTPBIN_URL` environment variable, or run only the offline tests.
- The OpenAI integration is implemented, including asynchronous requests, usage tracking, timeouts, retries, and error handling. The real API path has **not been verified end-to-end** because the available API account returned a `429 insufficient_quota` error during testing. The quota behavior itself is covered by an automated regression test using the real OpenAI SDK and a mocked HTTP transport.

---

## How to Run

### Install dependencies

From the relevant exercise directory:

```bash
pip install -r requirements.txt
```

Phase 2 additionally needs `pandas` and `numpy`, on top of the Exercise 2 requirements (the HTTP client `httpx2` is installed with the OpenAI SDK). Phase 3 needs `pandas` and `scikit-learn` (see `phase3-ml/requirements.txt`).

### Run Exercise 2 in mock mode

```bash
cd exercise2_prompt_runner
python -m prompt_runner.cli \
  --input prompts.csv \
  --output results.csv \
  --mock
```

The mock mode does not require an OpenAI API key. For real API execution, configure the required API credentials as environment variables rather than storing them in source code.

### Run the Phase 2 tests

```bash
cd phase2-api-data
python -m pytest -q                       # all 96 tests (uses httpbin.org)
python -m pytest -q -m "not network"      # offline tests only
HTTPBIN_URL=http://127.0.0.1:8085 python -m pytest -q   # against a local httpbin
```

### Run the Phase 2 analysis

```bash
cd phase2-api-data
python clean_data.py                      # writes chat_logs_cleaned.csv
python report.py                          # prints the report, writes report_*.csv
python load_database.py                   # builds database.sqlite
sqlite3 database.sqlite < sql/00_create_clean_view.sql
sqlite3 -header -column database.sqlite < sql/07_error_rate_by_model.sql
```

### Run the Phase 3 evaluation

```bash
cd phase2-api-data && python clean_data.py   # creates chat_logs_cleaned.csv, which Phase 3 reads
cd ../phase3-ml
python baseline.py                        # dummy baseline
python leakage_demo.py                    # leaky vs honest features
python evaluate.py                        # threshold, metrics, intervals, verdict
python -m pytest -q                       # 49 tests
```

### Run the whole project's tests

```bash
pytest -q                                 # from the repository root, using pytest.ini (124 tests)
```

---

## Roadmap

Completed so far: test automation foundations (Phase 1), API contracts and data analysis (Phase 2) and machine-learning evaluation (Phase 3).

The portfolio will evolve toward more advanced AI Quality Engineering topics:

- AI evaluation frameworks
- LLM evaluation datasets
- RAG testing
- retrieval and generation quality
- prompt injection testing
- AI test agents
- quality gates for AI features
- risk-based AI release validation
- AI Quality Engineering platform design

The goal is to demonstrate how traditional QA engineering practices can be extended to address the reliability, security, and quality challenges introduced by AI-powered systems.
