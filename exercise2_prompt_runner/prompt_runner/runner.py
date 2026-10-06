import asyncio
import csv
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError

from prompt_runner.client import LLMClient
from prompt_runner.models import PromptResult, PromptRow


def read_prompts(input_path: str) -> tuple[list[PromptRow], int]:
    valid_rows: list[PromptRow] = []
    skipped_rows = 0

    with open(input_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                print(
                    f"Invalid row {row_number}: "
                    "extra columns detected. "
                    "Make sure CSV fields containing commas are quoted."
                )
                skipped_rows += 1
                continue

            try:
                prompt_row = PromptRow.model_validate(row)
                valid_rows.append(prompt_row)
            except ValidationError as error:
                print(f"Invalid row {row_number}: {error}")
                skipped_rows += 1

    return valid_rows, skipped_rows


def write_results(
    output_path: str,
    results: list[PromptResult],
) -> None:
    fieldnames = [
        "id",
        "prompt",
        "category",
        "response",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for result in results:
            writer.writerow(result.model_dump())


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, RateLimitError):
        if getattr(error, "code", None) == "insufficient_quota":
            return False

        return True

    if isinstance(
        error,
        (
            asyncio.TimeoutError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
        ),
    ):
        return True

    status_code = getattr(error, "status_code", None)

    return status_code in (500, 502, 503, 504)


async def run_prompt(
    client: LLMClient,
    prompt_row: PromptRow,
    model: str,
    temperature: float | None,
    timeout: float,
) -> PromptResult:
    start_time = time.perf_counter()

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            completion = await asyncio.wait_for(
                client.complete(
                    prompt=prompt_row.prompt,
                    model=model,
                    temperature=temperature,
                ),
                timeout=timeout,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return PromptResult(
                id=prompt_row.id,
                prompt=prompt_row.prompt,
                category=prompt_row.category,
                response=completion.text,
                latency_ms=latency_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                error=None,
            )

        except Exception as error:
            if is_retryable_error(error) and attempt < max_attempts - 1:
                await asyncio.sleep(0.1 * (2**attempt))
                continue

            latency_ms = (time.perf_counter() - start_time) * 1000

            if isinstance(error, asyncio.TimeoutError):
                error_message = (
                    f"TimeoutError: no response within {timeout:g}s"
                )
            else:
                error_message = f"{type(error).__name__}: {error}"

            return PromptResult(
                id=prompt_row.id,
                prompt=prompt_row.prompt,
                category=prompt_row.category,
                response="",
                latency_ms=latency_ms,
                input_tokens=None,
                output_tokens=None,
                error=error_message,
            )

    raise RuntimeError("Unexpected retry loop termination")


async def run_prompt_with_limit(
    semaphore: asyncio.Semaphore,
    client: LLMClient,
    prompt_row: PromptRow,
    model: str,
    temperature: float | None,
    timeout: float,
) -> PromptResult:
    async with semaphore:
        return await run_prompt(
            client=client,
            prompt_row=prompt_row,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )


async def run_prompts(
    client: LLMClient,
    prompt_rows: list[PromptRow],
    model: str,
    temperature: float | None,
    max_concurrency: int,
    timeout: float = 30.0,
) -> list[PromptResult]:
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        run_prompt_with_limit(
            semaphore=semaphore,
            client=client,
            prompt_row=prompt_row,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
        for prompt_row in prompt_rows
    ]

    responses = await asyncio.gather(*tasks)

    return responses