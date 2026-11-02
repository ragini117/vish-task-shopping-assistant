"""Create the chat model for the selected mode and translate API errors.

Free mode -> Groq, Paid mode -> OpenAI. Everything else in the pipeline
(dataset, FAISS index, retrieval, prompts) is shared between both modes.
"""

import logging
import re
from dataclasses import dataclass

from config import FREE_MODE, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS, PAID_MODE

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """User-presentable LLM failure (never contains the API key)."""


@dataclass
class LLMSettings:
    mode: str
    model: str
    api_key: str = ""

    @property
    def provider(self) -> str:
        return "OpenAI" if self.mode == PAID_MODE else "Groq"

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"

    @property
    def ready(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and self.model)

    def __repr__(self) -> str:  # keep the key out of logs and tracebacks
        return f"LLMSettings(mode={self.mode!r}, model={self.model!r}, api_key={'***' if self.api_key else ''!r})"


def _uses_fixed_temperature(model: str) -> bool:
    """OpenAI reasoning models reject a custom temperature."""
    return model.lower().startswith(("o1", "o3", "o4", "gpt-5"))


def create_llm(settings: LLMSettings):
    if not settings.ready:
        raise LLMError(f"No {settings.provider} API key provided. Add it in the sidebar to enable AI responses.")
    key = settings.api_key.strip()
    if settings.mode == PAID_MODE:
        from langchain_openai import ChatOpenAI

        kwargs = dict(model=settings.model, api_key=key, timeout=LLM_TIMEOUT_SECONDS, max_retries=1)
        if not _uses_fixed_temperature(settings.model):
            kwargs["temperature"] = LLM_TEMPERATURE
        return ChatOpenAI(**kwargs)
    if settings.mode == FREE_MODE:
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.model, api_key=key, temperature=LLM_TEMPERATURE,
                        timeout=LLM_TIMEOUT_SECONDS, max_retries=1)
    raise LLMError(f"Unknown LLM mode: {settings.mode}")


def _provider_detail(exc: Exception, api_key: str = "") -> str:
    """The provider's own error message (e.g. Groq's 'model_not_found' text), with any API key redacted."""
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        err = body.get("error", body)
        detail = err.get("message", "") if isinstance(err, dict) else str(err)
    detail = detail or getattr(exc, "message", "") or str(exc)
    if api_key and api_key.strip():
        detail = detail.replace(api_key.strip(), "***")
    detail = re.sub(r"\b(gsk_|sk-)[A-Za-z0-9_\-]{8,}", r"\1***", detail)
    return detail.strip()[:300]


def describe_llm_error(exc: Exception, provider: str, api_key: str = "") -> str:
    """Map provider exceptions to a clear message, followed by the provider's own (key-redacted) error text."""
    summary = _summarize_llm_error(exc, provider)
    detail = _provider_detail(exc, api_key)
    return f"{summary} {provider} said: \"{detail}\"" if detail else summary


def _summarize_llm_error(exc: Exception, provider: str) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "authentication" in name or "401" in text or "invalid api key" in text or "incorrect api key" in text \
            or "invalid_api_key" in text:
        return f"Your {provider} API key was rejected. Please check the key in the sidebar."
    if "insufficient_quota" in text or "exceeded your current quota" in text or "billing" in text \
            or "credit" in text:
        return f"Your {provider} account has insufficient credits or quota. Add credits or switch LLM mode."
    if "ratelimit" in name or "rate limit" in text or "429" in text or "rate_limit" in text:
        return f"{provider} rate limit reached. Please wait a few seconds and try again."
    if "notfound" in name or "model_not_found" in text or "does not exist" in text or "decommissioned" in text \
            or "404" in text:
        return f"The selected {provider} model is not available for your key. Choose another model in the sidebar."
    if "permission" in name or "403" in text:
        return f"Your {provider} key does not have permission to use this model."
    if "timeout" in name or "timed out" in text:
        return f"The {provider} request timed out. Please try again."
    if "connection" in name or "connect" in text:
        return f"Could not reach {provider}. Check your internet connection."
    if "badrequest" in name or "400" in text:
        return f"{provider} rejected the request (bad request). Try another model."
    return f"{provider} request failed ({type(exc).__name__})."


def generate(settings: LLMSettings, messages: list[tuple[str, str]]) -> str:
    llm = create_llm(settings)
    try:
        result = llm.invoke(messages)
    except Exception as exc:
        logger.warning("LLM call failed: %s via %s (%s)", type(exc).__name__, settings.provider, settings.model)
        raise LLMError(describe_llm_error(exc, settings.provider, settings.api_key)) from None
    content = result.content if hasattr(result, "content") else str(result)
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    if not content.strip():
        raise LLMError(f"{settings.label} returned an empty response. Try again or choose another model.")
    return content.strip()


# Groq also serves speech, TTS and safety-classifier models; they cannot answer chat messages.
_GROQ_NON_CHAT = ("whisper", "orpheus", "playai", "tts", "guard", "safeguard")


def list_groq_models(api_key: str) -> list[str]:
    """Chat model IDs currently available to this Groq key, fetched live from the Groq API."""
    from groq import Groq

    try:
        models = Groq(api_key=api_key.strip(), timeout=10, max_retries=1).models.list().data
    except Exception as exc:
        logger.warning("Could not list Groq models: %s", type(exc).__name__)
        raise LLMError(describe_llm_error(exc, "Groq", api_key)) from None
    return sorted(m.id for m in models
                  if getattr(m, "active", True) and not any(tag in m.id.lower() for tag in _GROQ_NON_CHAT))


def test_connection(settings: LLMSettings) -> str:
    """Tiny request used by the sidebar 'Test connection' button."""
    reply = generate(settings, [("human", "Reply with the single word OK.")])
    return reply[:40] or "OK"
