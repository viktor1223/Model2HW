"""LLM client abstraction for dependency injection."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface for LLM API calls."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return the LLM's text response."""
        ...


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4, GPT-5, etc.).

    Requires the ``openai`` package (install with ``pip install model2hw[agents]``).
    """

    def __init__(self, model: str = "gpt-4", api_key: str | None = None) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "The openai package is required for OpenAIClient. "
                "Install it with: pip install model2hw[agents]"
            )

        import os

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key is required. Pass it via the api_key parameter "
                "or set the OPENAI_API_KEY environment variable."
            )

        self._model = model
        self._client = openai.OpenAI(api_key=resolved_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
