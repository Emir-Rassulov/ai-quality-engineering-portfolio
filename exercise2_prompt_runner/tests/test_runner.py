import asyncio
import csv

import httpx2
import pytest
from openai import AsyncOpenAI
from typer.testing import CliRunner

from prompt_runner.client import (
    MockClient,
    validate_temperature,
)
from prompt_runner.cli import app
from prompt_runner.models import PromptRow
from prompt_runner.runner import (
    read_prompts,
    run_prompts,
    write_results,
)
from tests.fakes import (
    AlwaysFailRetryClient,
    AlwaysTimeoutClient,
    NonRetryableClient,
    RateLimitClient,
    RetryClient,
    TimeoutClient,
    TrackingClient,
)


def test_valid_csv_is_loaded(tmp_path):
    csv_file = tmp_path / "prompts.csv"

    csv_file.write_text(
        "id,prompt,category\n"
        "1,What is Python?,factual\n"
        "2,What is QA?,factual\n",
        encoding="utf-8",
    )

    rows, skipped_rows = read_prompts(str(csv_file))

    assert len(rows) == 2
    assert skipped_rows == 0


def test_malformed_row_is_skipped(tmp_path, capsys):
    csv_file = tmp_path / "prompts.csv"

    csv_file.write_text(
        "id,prompt,category\n"
        "1,What is Python?,factual\n"
        "invalid,This row is bad,factual\n",
        encoding="utf-8",
    )

    rows, skipped_rows = read_prompts(str(csv_file))

    captured = capsys.readouterr()

    assert len(rows) == 1
    assert skipped_rows == 1
    assert "Invalid row 3" in captured.out


def test_extra_csv_columns_are_skipped(tmp_path, capsys):
    csv_file = tmp_path / "prompts.csv"

    csv_file.write_text(
        "id,prompt,category\n"
        "1,What is Python?,factual,unexpected\n",
        encoding="utf-8",
    )

    rows, skipped_rows = read_prompts(str(csv_file))

    captured = capsys.readouterr()

    assert rows == []
    assert skipped_rows == 1
    assert "extra columns detected" in captured.out


@pytest.mark.asyncio
async def test_failing_call_does_not_abort_run():
    client = MockClient()

    rows = [
        PromptRow(
            id=1,
            prompt="What is Python?",
            category="factual",
        ),
        PromptRow(
            id=2,
            prompt="FAIL this request",
            category="test_error",
        ),
        PromptRow(
            id=3,
            prompt="What is QA?",
            category="factual",
        ),
    ]

    results = await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=2,
    )

    assert len(results) == 3

    assert results[0].error is None
    assert results[0].response == "Mock response for: What is Python?"

    assert results[1].error == "RuntimeError: Mock API error"

    assert results[2].error is None
    assert results[2].response == "Mock response for: What is QA?"


@pytest.mark.asyncio
async def test_concurrency_does_not_exceed_limit():
    client = TrackingClient()

    rows = [
        PromptRow(
            id=i,
            prompt=f"Prompt {i}",
            category="factual",
        )
        for i in range(1, 6)
    ]

    await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=2,
    )

    assert client.max_active_calls == 2


def test_output_columns(tmp_path):
    output_file = tmp_path / "results.csv"

    rows = [
        PromptRow(
            id=1,
            prompt="What is Python?",
            category="factual",
        )
    ]

    client = MockClient()

    results = asyncio.run(
        run_prompts(
            client=client,
            prompt_rows=rows,
            model="mock-model",
            temperature=None,
            max_concurrency=2,
        )
    )

    write_results(str(output_file), results)

    with open(output_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        data = list(reader)

        assert reader.fieldnames == [
            "id",
            "prompt",
            "category",
            "response",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "error",
        ]

        assert len(data) == 1
        assert data[0]["id"] == "1"
        assert data[0]["prompt"] == "What is Python?"
        assert data[0]["category"] == "factual"
        assert data[0]["response"] == "Mock response for: What is Python?"
        assert data[0]["error"] == ""


@pytest.mark.asyncio
async def test_retry_on_retryable_error():
    client = RetryClient()

    rows = [
        PromptRow(
            id=1,
            prompt="Test retry",
            category="test",
        )
    ]

    results = await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=1,
    )

    assert client.attempts == 2
    assert results[0].response == "Successful response for: Test retry"
    assert results[0].error is None


@pytest.mark.asyncio
async def test_retry_on_openai_rate_limit_error():
    client = RateLimitClient()

    rows = [
        PromptRow(
            id=1,
            prompt="Test OpenAI rate limit",
            category="test",
        )
    ]

    results = await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=1,
    )

    assert client.attempts == 2
    assert (
        results[0].response
        == "Successful response for: Test OpenAI rate limit"
    )
    assert results[0].error is None


@pytest.mark.asyncio
async def test_retry_on_timeout():
    client = TimeoutClient()

    rows = [
        PromptRow(
            id=1,
            prompt="Test timeout",
            category="test",
        )
    ]

    results = await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=1,
        timeout=0.05,
    )

    assert client.attempts == 2
    assert results[0].response == "Successful response for: Test timeout"
    assert results[0].error is None


@pytest.mark.asyncio
async def test_timeout_retries_are_exhausted():
    client = AlwaysTimeoutClient()

    rows = [
        PromptRow(
            id=1,
            prompt="Always timeout",
            category="test",
        )
    ]

    results = await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=1,
        timeout=0.01,
    )

    assert client.attempts == 3
    assert results[0].response == ""
    assert results[0].error != ""
    assert "TimeoutError" in results[0].error


@pytest.mark.asyncio
async def test_retries_are_exhausted():
    client = AlwaysFailRetryClient()

    rows = [
        PromptRow(
            id=1,
            prompt="Always fail",
            category="test",
        )
    ]

    results = await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=1,
    )

    assert client.attempts == 3
    assert results[0].response == ""
    assert results[0].error == "InternalServerError: Internal server error"


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried():
    client = NonRetryableClient()

    rows = [
        PromptRow(
            id=1,
            prompt="Do not retry",
            category="test",
        )
    ]

    results = await run_prompts(
        client=client,
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=1,
    )

    assert client.attempts == 1
    assert results[0].response == ""
    assert results[0].error == "RuntimeError: Invalid request"


def test_cli_mock_mode(tmp_path):
    input_file = tmp_path / "prompts.csv"
    output_file = tmp_path / "results.csv"

    input_file.write_text(
        "id,prompt,category\n"
        "1,What is Python?,factual\n"
        "2,What is QA?,factual\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--mock",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Results written to" in result.stdout
    assert "succeeded 2" in result.stdout
    assert "failed 0" in result.stdout

    with open(output_file, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["response"] == "Mock response for: What is Python?"


def test_temperature_none_is_allowed():
    validate_temperature(None)


def test_temperature_value_is_rejected():
    with pytest.raises(
        ValueError,
        match="Temperature is not supported",
    ):
        validate_temperature(0.7)


def test_cli_mock_failure_returns_exit_code_1(tmp_path):
    input_file = tmp_path / "prompts.csv"
    output_file = tmp_path / "results.csv"

    input_file.write_text(
        "id,prompt,category\n"
        "1,FAIL this request,test_error\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--mock",
        ],
    )

    assert result.exit_code == 1
    assert "failed 1" in result.stdout
    assert output_file.exists()


def test_cli_rejects_temperature_before_creating_client(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    input_file = tmp_path / "prompts.csv"
    output_file = tmp_path / "results.csv"

    input_file.write_text(
        "id,prompt,category\n"
        "1,What is Python?,factual\n",
        encoding="utf-8",
    )

    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--temperature",
            "0.7",
        ],
    )

    assert result.exit_code == 2
    assert "Temperature is not supported" in result.output
    assert not output_file.exists()


@pytest.mark.asyncio
async def test_real_openai_insufficient_quota_is_not_retried():
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1

        return httpx2.Response(
            status_code=429,
            request=request,
            json={
                "error": {
                    "message": "You exceeded your current quota.",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            },
        )

    http_client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    )

    client = AsyncOpenAI(
        api_key="test-key",
        http_client=http_client,
        max_retries=0,
    )

    class RealSDKClient:
        async def complete(
            self,
            prompt: str,
            model: str,
            temperature: float | None,
        ):
            response = await client.responses.create(
                model=model,
                input=prompt,
            )

            return response

    rows = [
        PromptRow(
            id=1,
            prompt="Test insufficient quota",
            category="test",
        )
    ]

    results = await run_prompts(
        client=RealSDKClient(),
        prompt_rows=rows,
        model="mock-model",
        temperature=None,
        max_concurrency=1,
    )

    await http_client.aclose()

    assert attempts == 1
    assert results[0].response == ""
    assert "RateLimitError" in results[0].error
    assert "insufficient_quota" in results[0].error