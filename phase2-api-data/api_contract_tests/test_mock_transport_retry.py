"""
Flaky-network handling: the real run_prompts retry loop, driven through the
real OpenAI SDK and a mock transport. No network is used.
"""
import httpx2
import pytest
from openai import AsyncOpenAI

from prompt_runner.client import LLMResponse
from prompt_runner.models import PromptRow
from prompt_runner.runner import run_prompts

SERVER_ERROR = (
    503,
    {"error": {"message": "Service unavailable", "type": "server_error", "code": "server_error"}},
)
RATE_LIMITED = (
    429,
    {"error": {"message": "Too many requests", "type": "rate_limit", "code": "rate_limit_exceeded"}},
)
QUOTA_EXHAUSTED = (
    429,
    {"error": {"message": "quota", "type": "insufficient_quota", "code": "insufficient_quota"}},
)
BAD_REQUEST = (
    400,
    {"error": {"message": "bad input", "type": "invalid_request_error", "code": None}},
)
SUCCESS = (
    200,
    {
        "id": "resp_test",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "mock-model",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Success", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
)


async def run_with_script(script):
    """
    Reply to the n-th request with script[n] (the last reply repeats forever).
    Returns the single result and how many requests the server received.
    """
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        status, body = script[min(attempts, len(script) - 1)]
        attempts += 1
        return httpx2.Response(status, request=request, json=body)

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    sdk_client = AsyncOpenAI(
        api_key="test-key",
        http_client=http_client,
        max_retries=0,  # the SDK must not retry: only our code may
    )

    class MockTransportClient:
        async def complete(
            self, prompt: str, model: str, temperature: float | None
        ) -> LLMResponse:
            response = await sdk_client.responses.create(model=model, input=prompt)
            usage = response.usage
            return LLMResponse(
                text=response.output_text,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
            )

    rows = [PromptRow(id=1, prompt="Test retry", category="test")]

    try:
        results = await run_prompts(
            client=MockTransportClient(),
            prompt_rows=rows,
            model="mock-model",
            temperature=None,
            max_concurrency=1,
        )
    finally:
        await http_client.aclose()

    return results[0], attempts


@pytest.mark.asyncio
async def test_retries_after_503_and_then_succeeds():
    result, attempts = await run_with_script([SERVER_ERROR, SUCCESS])

    assert attempts == 2
    assert result.response == "Success"
    assert result.error is None


@pytest.mark.asyncio
async def test_retries_after_rate_limit_and_then_succeeds():
    result, attempts = await run_with_script([RATE_LIMITED, SUCCESS])

    assert attempts == 2
    assert result.error is None


@pytest.mark.asyncio
async def test_gives_up_after_three_attempts_and_records_the_error():
    result, attempts = await run_with_script([SERVER_ERROR])

    assert attempts == 3
    assert result.response == ""
    assert result.error.startswith("InternalServerError")


@pytest.mark.asyncio
async def test_permanent_error_is_not_retried():
    result, attempts = await run_with_script([BAD_REQUEST, SUCCESS])

    assert attempts == 1
    assert result.error.startswith("BadRequestError")


@pytest.mark.asyncio
async def test_exhausted_quota_is_not_retried():
    result, attempts = await run_with_script([QUOTA_EXHAUSTED, SUCCESS])

    assert attempts == 1
    assert result.error.startswith("RateLimitError")
