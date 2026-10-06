import asyncio

import httpx2
from openai import InternalServerError, RateLimitError

from prompt_runner.client import LLMResponse, MockClient


def create_internal_server_error() -> InternalServerError:
    response = httpx2.Response(
        status_code=500,
        request=httpx2.Request(
            "POST",
            "https://api.openai.com/v1/responses",
        ),
    )

    return InternalServerError(
        "Internal server error",
        response=response,
        body={
            "error": {
                "message": "Internal server error"
            }
        },
    )


class TrackingClient(MockClient):
    def __init__(self) -> None:
        self.active_calls = 0
        self.max_active_calls = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.active_calls += 1
        self.max_active_calls = max(
            self.max_active_calls,
            self.active_calls,
        )

        await asyncio.sleep(0.05)

        self.active_calls -= 1

        return LLMResponse(
            text=f"Response for: {prompt}"
        )


class RetryClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.attempts += 1

        if self.attempts == 1:
            raise create_internal_server_error()

        return LLMResponse(
            text=f"Successful response for: {prompt}"
        )


class RateLimitClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.attempts += 1

        if self.attempts == 1:
            response = httpx2.Response(
                status_code=429,
                request=httpx2.Request(
                    "POST",
                    "https://api.openai.com/v1/responses",
                ),
            )

            raise RateLimitError(
                "Too many requests",
                response=response,
                body={
                    "error": {
                        "message": "Too many requests"
                    }
                },
            )

        return LLMResponse(
            text=f"Successful response for: {prompt}"
        )


class AlwaysFailRetryClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.attempts += 1

        raise create_internal_server_error()


class NonRetryableClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.attempts += 1

        raise RuntimeError("Invalid request")


class TimeoutClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.attempts += 1

        if self.attempts == 1:
            await asyncio.sleep(10)

        return LLMResponse(
            text=f"Successful response for: {prompt}"
        )


class AlwaysTimeoutClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.attempts += 1

        await asyncio.sleep(10)

        return LLMResponse(
            text=f"Successful response for: {prompt}"
        )

class InsufficientQuotaClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        self.attempts += 1

        response = httpx2.Response(
            status_code=429,
            request=httpx2.Request(
                "POST",
                "https://api.openai.com/v1/responses",
            ),
        )

        error = RateLimitError(
            "Quota exhausted",
            response=response,
            body={
                "error": {
                    "message": "Quota exhausted",
                    "code": "insufficient_quota",
                }
            },
        )

        raise error