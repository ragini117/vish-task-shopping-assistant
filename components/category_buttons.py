"""Dynamic category / subcategory buttons rendered from dataset-derived options."""

import streamlit as st

from utils.helpers import slugify


def _select(option: dict):
    st.session_state["pending_action"] = {"type": "category", "option": option}
    st.session_state["selected_category"] = option["category"]
    st.session_state["selected_subcategory"] = option.get("subcategory")


def render_category_buttons(options: list[dict], message_index: int, columns: int = 3):
    """Responsive grid of buttons; each click stores the selection in session state."""
    if not options:
        return
    for start in range(0, len(options), columns):
        cols = st.columns(columns)
        for col, option in zip(cols, options[start: start + columns]):
            key = f"cat_{message_index}_{start}_{slugify(option['category'])}_{slugify(option.get('subcategory') or 'all')}"
            col.button(option["label"], key=key, on_click=_select, args=(option,), use_container_width=True)


def _ask(text: str):
    st.session_state["pending_action"] = {"type": "message", "text": text}


def render_followups(suggestions: list[str], message_index: int):
    if not suggestions:
        return
    st.markdown("<div class='followup-label'>Try asking:</div>", unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for i, (col, text) in enumerate(zip(cols, suggestions)):
        col.button(text, key=f"fu_{message_index}_{i}", on_click=_ask, args=(text,), use_container_width=True)
