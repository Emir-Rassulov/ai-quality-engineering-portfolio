from pydantic import BaseModel


class PromptRow(BaseModel):
    id: int
    prompt: str
    category: str


class PromptResult(BaseModel):
    id: int
    prompt: str
    category: str
    response: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None