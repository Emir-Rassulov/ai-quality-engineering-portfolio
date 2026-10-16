"""
How our client code classifies errors, using the errors the REAL OpenAI SDK
raises (built through a mock transport), not hand-made fakes.
"""
import asyncio

import httpx2
import pytest
from openai import AsyncOpenAI

from prompt_runner.runner import is_retryable_error


async def sdk_error(status=None, *, body=None, transport_error=None):
    """Make the real SDK raise the error it would raise for this response."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if transport_error is not None:
            raise transport_error
        return httpx2.Response(
            status,
            request=request,
            json=body or {"error": {"message": "boom", "type": "t", "code": None}},
        )

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    sdk = AsyncOpenAI(api_key="test-key", http_client=http_client, max_retries=0)
    try:
        await sdk.responses.create(model="m", input="hi")
    except Exception as error:
        return error
    finally:
        await http_client.aclose()
    raise AssertionError("the SDK did not raise")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, False),
        (404, False),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
async def test_sdk_status_errors_are_classified_correctly(status_code, expected):
    error = await sdk_error(status_code)

    assert is_retryable_error(error) is expected


@pytest.mark.asyncio
async def test_429_for_exhausted_quota_is_not_retryable():
    """A permanent 429: retrying can never help."""
    error = await sdk_error(
        429,
        body={
            "error": {
                "message": "quota",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }
        },
    )

    assert error.code == "insufficient_quota"
    assert is_retryable_error(error) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_error",
    [httpx2.ReadTimeout("slow"), httpx2.ConnectError("down")],
    ids=["timeout", "connection-error"],
)
async def test_network_failures_are_retryable(transport_error):
    error = await sdk_error(transport_error=transport_error)

    assert is_retryable_error(error) is True


def test_asyncio_timeout_is_retryable():
    assert is_retryable_error(asyncio.TimeoutError()) is True


def test_unrecognized_error_is_not_retryable():
    assert is_retryable_error(ValueError("Invalid input")) is False


# Fallback branch: errors that are not SDK errors but carry a status_code.
class FakeHTTPError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP error: {status_code}")


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, False),
        (404, False),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
def test_generic_status_errors_use_the_same_rule_as_sdk_errors(status_code, expected):
    assert is_retryable_error(FakeHTTPError(status_code)) is expected
