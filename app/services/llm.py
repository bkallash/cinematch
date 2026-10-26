"""Shared OpenRouter (OpenAI-compatible) client construction and JSON chat calls."""

from openai import AsyncOpenAI

from app.config import settings


def create_client() -> AsyncOpenAI:
    # The SDK refuses an empty key; callers check OPENROUTER_API_KEY before real calls.
    return AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY or "unset",
        base_url=settings.OPENROUTER_BASE_URL,
        timeout=30.0,
        max_retries=1,
    )


async def complete_json(client: AsyncOpenAI, system: str, prompt: str, temperature: float) -> str:
    """Run one JSON-mode chat completion and return the raw message content."""
    response = await client.chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
