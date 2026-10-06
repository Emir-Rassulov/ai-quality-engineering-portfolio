from dataclasses import dataclass
import os
from typing import Protocol

from openai import AsyncOpenAI


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        ...


class MockClient:
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        if "FAIL" in prompt:
            raise RuntimeError("Mock API error")

        return LLMResponse(
            text=f"Mock response for: {prompt}"
        )


def validate_temperature(temperature: float | None) -> None:
    if temperature is not None:
        raise ValueError(
            "Temperature is not supported by the selected "
            "OpenAI model/API configuration."
        )


class OpenAIClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            max_retries=0,
        )

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
    ) -> LLMResponse:
        validate_temperature(temperature)

        response = await self.client.responses.create(
            model=model,
            input=prompt,
        )

        usage = response.usage

        return LLMResponse(
            text=response.output_text,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )