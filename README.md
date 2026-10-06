# AI Quality Engineering Portfolio

A hands-on portfolio focused on **Quality Engineering for AI-powered systems**, combining traditional test automation with AI/LLM tooling, reliability testing, and quality-focused engineering practices.

The project is built incrementally to demonstrate practical QA engineering skills and the transition from conventional automation toward **AI Quality Engineering**.

---

## Overview

This repository contains a series of practical QA engineering exercises covering:

- API test automation
- Python and Pytest
- asynchronous execution
- bounded concurrency
- retry and timeout handling
- LLM/AI tooling
- OpenAI SDK integration
- Pydantic validation
- structured test data and results
- mock-based testing
- provider-specific error handling

The focus is not only on writing tests, but also on building **reliable, maintainable, and testable engineering tools**.

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
│   │   └── test_runner.py
│   │
│   ├── prompts.csv
│   └── requirements.txt
│
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

## Technical Stack

| Area            | Technology          |
| --------------- | ------------------- |
| Language        | Python              |
| Test framework  | Pytest              |
| Validation      | Pydantic            |
| CLI             | Typer               |
| HTTP / API      | OpenAI SDK          |
| Async execution | asyncio             |
| Test isolation  | Mock / fake clients |
| Data format     | CSV                 |
| Version control | Git / GitHub        |

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
- retryable rate-limit errors

Non-retryable failures are returned immediately.

### Provider-specific error handling

The OpenAI SDK normalizes provider errors into typed exceptions and exposes fields such as `error.code`.

The implementation uses the SDK-provided error code to distinguish quota exhaustion (`insufficient_quota`) from retryable rate limiting.

This prevents exhausted API quota from being incorrectly retried.

### Real SDK regression testing

The `insufficient_quota` behavior is tested through the actual `AsyncOpenAI` client using a mocked HTTP transport.

This verifies the application's assumptions against the SDK's real error parsing behavior rather than relying only on manually constructed fake exceptions.

### Mock mode

The CLI supports a mock client so that the complete workflow can be tested without:

- an API key
- network access
- API costs
- external service availability

---

## Testing

The automated test suite covers:

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

Tests are designed to validate both **functional behavior** and **failure handling**.

---

## Real API Status

The OpenAI integration is implemented, including asynchronous requests, usage tracking, timeouts, retries, and error handling.

The real API path has **not been verified end-to-end** because the available API account returned a `429 insufficient_quota` error during testing.

The quota behavior itself is covered by an automated regression test using the real OpenAI SDK and a mocked HTTP transport.

All other automated tests can run without network access or an API key.

---

## How to Run

### Install dependencies

From the relevant exercise directory:

```bash
pip install -r requirements.txt
```

### Run Exercise 2 in mock mode

```bash
python -m prompt_runner.cli \
  --input prompts.csv \
  --output results.csv \
  --mock
```

### Run the test suite

```bash
pytest -v
```

The mock mode does not require an OpenAI API key.

For real API execution, configure the required API credentials as environment variables rather than storing them in source code.

---

## Roadmap

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
