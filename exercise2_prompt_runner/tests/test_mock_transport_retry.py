import httpx2
import pytest
from openai import AsyncOpenAI

from prompt_runner.client import LLMResponse
from prompt_runner.models import PromptRow
from prompt_runner.runner import run_prompts


@pytest.mark.asyncio
async def test_mock_transport_retries_after_503():
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx2.Response(
                status_code=503,
                request=request,
                json={
                    "error": {
                        "message": "Service unavailable",
                        "type": "server_error",
                        "code": "server_error",
                    }
                },
            )

        return httpx2.Response(
            status_code=200,
            request=request,
            json={
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
                            {
                                "type": "output_text",
                                "text": "Success",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    http_client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    )

    sdk_client = AsyncOpenAI(
        api_key="test-key",
        http_client=http_client,
        max_retries=0,
    )

    class MockTransportClient:
        async def complete(
            self,
            prompt: str,
            model: str,
            temperature: float | None,
        ) -> LLMResponse:
            response = await sdk_client.responses.create(
                model=model,
                input=prompt,
            )

            usage = response.usage

            return LLMResponse(
                text=response.output_text,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
            )

    rows = [
        PromptRow(
            id=1,
            prompt="Test retry",
            category="test",
        )
    ]

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

    assert attempts == 2
    assert results[0].response == "Success"
    assert results[0].error is None
