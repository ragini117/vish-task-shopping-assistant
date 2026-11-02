"""Chat history and shopping context stored in Streamlit session state.

The manager works on any dict-like object, so it can be unit-tested without
Streamlit by passing a plain dict.
"""

from dataclasses import asdict, dataclass, field
from typing import MutableMapping

from config import HISTORY_TURNS_FOR_LLM
from rag.retrieval.product_filter import SearchFilters


@dataclass
class ShoppingContext:
    """The last search and the products it showed.

    It is never merged into a new search: every message is searched on its own. It is only used to refer to the
    products on screen ("add the first one", "compare the top 3") and, while `pending` is True, to apply the
    shopper's answer to a question the assistant just asked ("Exactly ₹100", "Under ₹300") to that request.
    """
    filters: SearchFilters = field(default_factory=SearchFilters)
    semantic_query: str = ""
    last_product_ids: list[str] = field(default_factory=list)
    offset: int = 0
    total: int = 0
    browse_all: bool = False              # shopper explicitly asked for the whole catalog
    pending: bool = False                 # the assistant asked a price question or offered a suggestion


@dataclass
class AssistantResponse:
    text: str
    kind: str = "text"                    # text | products | categories | compare | cart | detail
    product_ids: list[str] = field(default_factory=list)
    category_options: list[dict] = field(default_factory=list)
    compare_ids: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    llm_label: str | None = None          # which model wrote the text (None = template)
    llm_hint: str | None = None           # e.g. "add an API key"

    def to_message(self) -> dict:
        return {"role": "assistant", "content": self.text, **{k: v for k, v in asdict(self).items() if k != "text"}}


class ConversationManager:
    KEYS = ("messages", "shopping_context")

    def __init__(self, state: MutableMapping):
        self.state = state
        state.setdefault("messages", [])
        state.setdefault("shopping_context", None)

    @property
    def messages(self) -> list[dict]:
        return self.state["messages"]

    @property
    def context(self) -> ShoppingContext | None:
        return self.state["shopping_context"]

    @context.setter
    def context(self, value: ShoppingContext | None):
        self.state["shopping_context"] = value

    def add_user(self, text: str) -> int:
        self.messages.append({"role": "user", "content": text})
        return len(self.messages) - 1

    def add_assistant(self, response: AssistantResponse) -> int:
        self.messages.append(response.to_message())
        return len(self.messages) - 1

    def history_for_llm(self, exclude_last: int = 0) -> list[dict]:
        msgs = self.messages[: len(self.messages) - exclude_last] if exclude_last else self.messages
        return [{"role": m["role"], "content": m.get("content", "")} for m in msgs[-HISTORY_TURNS_FOR_LLM:]]

    def clear(self):
        self.state["messages"] = []
        self.state["shopping_context"] = None
