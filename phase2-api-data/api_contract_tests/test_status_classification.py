"""
Item 2 of the exercise: what httpbin's /status/{code} really returns, passed
through the real OpenAI SDK, then classified by our own is_retryable_error.
"""
import httpx2
import pytest
from openai import AsyncOpenAI

from prompt_runner.runner import is_retryable_error

pytestmark = pytest.mark.network


class ForwardToHttpbinStatus(httpx2.AsyncBaseTransport):
    """Send whatever the SDK asks for to httpbin's /status/{code} instead."""

    def __init__(self, base_url: str, code: int):
        self.base_url = base_url
        self.code = code

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        async with httpx2.AsyncClient(timeout=10.0) as upstream_client:
            upstream = await upstream_client.get(f"{self.base_url}/status/{self.code}")
        return httpx2.Response(
            upstream.status_code, content=upstream.content, request=request
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_retryable"),
    [
        (400, False),
        (404, False),
        (429, True),
        (500, True),
    ],
)
async def test_httpbin_status_is_classified_by_our_client(
    base_url, status_code, expected_retryable
):
    """
    200 is not an error, so there is nothing to classify (it is covered by the
    success tests). The other four must be classified as the exercise expects.
    """
    http_client = httpx2.AsyncClient(
        transport=ForwardToHttpbinStatus(base_url, status_code)
    )
    sdk = AsyncOpenAI(api_key="test-key", http_client=http_client, max_retries=0)

    try:
        with pytest.raises(Exception) as exc_info:
            await sdk.responses.create(model="m", input="hi")
    finally:
        await http_client.aclose()

    assert exc_info.value.status_code == status_code
    assert is_retryable_error(exc_info.value) is expected_retryable
