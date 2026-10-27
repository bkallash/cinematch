"""Shared OpenRouter (OpenAI-compatible) client and JSON chat calls."""

from typing import Optional

from openai import AsyncOpenAI

from app.config import settings

_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        # The SDK refuses an empty key; callers check OPENROUTER_API_KEY before real calls.
        _client = AsyncOpenAI(api_key=settings.OPENROUTER_API_KEY or "unset",
                              base_url=settings.OPENROUTER_BASE_URL, timeout=30.0, max_retries=1)
    return _client


async def complete_json(system: str, prompt: str, temperature: float) -> str:
    """Run one JSON-mode chat completion and return the raw message content."""
    response = await get_client().chat.completions.create(
        model=settings.OPENROUTER_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
