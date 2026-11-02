"""AI Shopping Assistant — Streamlit entry point.

Run with:  streamlit run app.py
"""

import logging
import math
import sys

import streamlit as st

from chatbot.conversation_manager import AssistantResponse, ConversationManager
from chatbot.intent_detector import IntentDetector
from chatbot.shopping_assistant import ShoppingAssistant
from components.category_buttons import render_category_buttons, render_followups
from components.product_cards import render_comparison, render_product_grid
from components.shopping_cart import Cart, render_cart
from config import (DATA_PATH, FREE_MODE, GROQ_API_KEY_ENV, GROQ_DEFAULT_MODEL, GROQ_MODELS, INDEX_DIR,
                    OPENAI_API_KEY_ENV, OPENAI_MODELS, PAID_MODE, PRICE_SLIDER_STEP)
from rag.embeddings.embedding_model import EmbeddingError, get_embeddings
from rag.ingestion.product_ingestion import DatasetError, load_products
from rag.llm.llm_factory import LLMError, LLMSettings, list_groq_models, test_connection
from rag.retrieval.product_filter import SearchFilters
from rag.retrieval.product_retriever import ProductRetriever
from utils.helpers import file_fingerprint
from rag.vectorstore.faiss_manager import FaissManager, VectorStoreError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛍️", layout="wide")

CSS = """
<style>
:root { --accent: #6c4ce0; --accent-soft: rgba(108, 76, 224, 0.12); --ok: #1a9c5b; --warn: #d98a00; --bad: #d64545; }
.block-container { padding-top: 4.5rem; max-width: 1280px; }
.app-title { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
.app-title span { background: linear-gradient(90deg, #6c4ce0, #e0569b); -webkit-background-clip: text; background-clip: text; color: transparent; }
.app-sub { opacity: 0.7; margin-top: 0.1rem; }
.mode-pill { display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px; font-weight: 600; font-size: 0.85rem;
             background: var(--accent-soft); border: 1px solid rgba(108, 76, 224, 0.35); }
.mode-pill.paid { background: rgba(224, 86, 155, 0.12); border-color: rgba(224, 86, 155, 0.4); }
.mode-pill .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: var(--warn); }
.mode-pill .dot.ready { background: var(--ok); }
/* Product card = keyed container holding the HTML details AND its Add to Cart / Compare buttons */
[class*="st-key-pcard_"] { border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 14px; padding: 0.9rem 1rem 0.8rem;
                           background: rgba(128, 128, 128, 0.04); gap: 0.4rem; min-height: 100%;
                           transition: box-shadow .15s ease, transform .15s ease; }
[class*="st-key-pcard_"]:hover { box-shadow: 0 6px 22px rgba(0, 0, 0, 0.18); transform: translateY(-1px); }
[class*="st-key-pcard_"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }  /* Streamlit default is -1rem */
.product-card { min-height: 372px; }
[class*="st-key-pcard_"] button { white-space: nowrap; padding-left: 0.4rem; padding-right: 0.4rem; }
[class*="st-key-pcard_"] button p { font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (max-width: 640px) { .product-card { min-height: 0; } }
/* Carousel controls */
[class*="st-key-cnav_"] button { border-radius: 999px; font-size: 1.1rem; font-weight: 700; }
.carousel-status { text-align: center; font-size: 0.85rem; opacity: 0.75; }
.cdots { display: inline-flex; gap: 5px; margin-left: 8px; vertical-align: middle; }
.cdot { width: 7px; height: 7px; border-radius: 50%; background: rgba(128, 128, 128, 0.4); }
.cdot.on { background: var(--accent); width: 18px; border-radius: 4px; }
/* Tablet: 2 cards visible, mobile: 1 card visible; swipe within the page, arrows change page */
@media (max-width: 1024px) {
  [class*="st-key-carousel_"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-pcard_"]) {
    flex-wrap: nowrap !important; overflow-x: auto; scroll-snap-type: x mandatory; padding-bottom: 0.4rem; }
  [class*="st-key-carousel_"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-pcard_"]) > [data-testid="stColumn"] {
    flex: 0 0 calc(50% - 0.5rem) !important; min-width: calc(50% - 0.5rem) !important; scroll-snap-align: start; }
  [class*="st-key-carousel_"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-pcard_"]) > [data-testid="stColumn"]:not(:has([class*="st-key-pcard_"])) { display: none; }
  [class*="st-key-pcard_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
  [class*="st-key-pcard_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 0 !important; }
}
@media (max-width: 640px) {
  [class*="st-key-carousel_"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-pcard_"]) > [data-testid="stColumn"] {
    flex: 0 0 100% !important; min-width: 100% !important; }
  [class*="st-key-cnav_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
  [class*="st-key-cnav_"] [data-testid="stColumn"] { min-width: 0 !important; }
}
.card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; }
.pid { font-family: ui-monospace, monospace; font-size: 0.72rem; opacity: 0.65; }
.pname { font-weight: 700; font-size: 1.02rem; line-height: 1.25; }
.pmeta { font-size: 0.8rem; opacity: 0.7; margin: 0.15rem 0 0.45rem; }
.price-row { display: flex; align-items: baseline; gap: 0.45rem; flex-wrap: wrap; }
.sale { font-size: 1.3rem; font-weight: 800; }
.mrp { text-decoration: line-through; opacity: 0.55; font-size: 0.88rem; }
.rating { font-size: 0.85rem; margin: 0.25rem 0 0.4rem; }
.stars { color: #f5a623; letter-spacing: 1px; }
.muted { opacity: 0.65; }
.desc { font-size: 0.82rem; opacity: 0.85; margin-bottom: 0.45rem; display: -webkit-box; -webkit-line-clamp: 3;
        -webkit-box-orient: vertical; overflow: hidden; }
.attr { font-size: 0.78rem; margin: 0.18rem 0; }
.attr b { display: inline-block; min-width: 58px; opacity: 0.8; }
.chip { display: inline-block; padding: 0.05rem 0.45rem; margin: 0.1rem 0.15rem 0.1rem 0; border-radius: 6px;
        font-size: 0.72rem; background: rgba(128, 128, 128, 0.14); }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }
.badge-in { background: rgba(26, 156, 91, 0.14); color: var(--ok); }
.badge-low { background: rgba(217, 138, 0, 0.15); color: var(--warn); }
.badge-out { background: rgba(214, 69, 69, 0.14); color: var(--bad); }
.badge-off { background: rgba(214, 69, 69, 0.12); color: var(--bad); }
.followup-label { font-size: 0.8rem; opacity: 0.65; margin-top: 0.6rem; }
.cart-meta { font-size: 0.78rem; opacity: 0.7; }
div[data-testid="stChatMessage"] { border-radius: 14px; }
@media (max-width: 640px) { .app-title { font-size: 1.5rem; } }
</style>
"""


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def build_pipeline(fingerprint: str, rebuild_token: int) -> dict:
    """Load data, FAISS index and helpers once per dataset version (not on every rerun)."""
    df = load_products(DATA_PATH)
    faiss_manager, warning = None, None
    try:
        faiss_manager = FaissManager(INDEX_DIR, get_embedding_model())
        faiss_manager.load_or_build(df, fingerprint, force_rebuild=rebuild_token > 0)
    except (EmbeddingError, VectorStoreError, ImportError) as exc:
        logging.getLogger(__name__).exception("Semantic search disabled")  # full traceback in the terminal
        faiss_manager = None
        warning = (f"Semantic search is unavailable ({type(exc).__name__}: {exc}). Product search is disabled "
                   "until this is fixed — there is no keyword fallback. See the terminal for the full traceback "
                   f"(Python: {sys.executable}).")
    retriever = ProductRetriever(df, faiss_manager)
    detector = IntentDetector(df)
    return {
        "df": df,
        "faiss": faiss_manager,
        "assistant": ShoppingAssistant(df, retriever, detector),
        "detector": detector,
        "warning": warning,
    }


def load_pipeline() -> dict:
    try:
        fingerprint = file_fingerprint(DATA_PATH)
    except FileNotFoundError:
        raise DatasetError(f"Dataset not found at '{DATA_PATH}'. Place {DATA_PATH.name} in the data/ folder.")
    token = st.session_state.get("rebuild_token", 0)
    with st.spinner("⏳ Loading product catalog, embedding model and FAISS index… "
                    "(takes 20–40 seconds after the server starts; later page loads are instant)"):
        return build_pipeline(fingerprint, token)


# --------------------------------------------------------------------------- sidebar
def persistent_widget(widget, label: str, key: str, default, **kwargs):
    """Keep a widget's value even while it is hidden (e.g. the other mode's API key)."""
    store = f"_keep_{key}"
    st.session_state.setdefault(store, default)
    st.session_state[key] = st.session_state[store]

    def save():
        st.session_state[store] = st.session_state[key]

    return widget(label, key=key, on_change=save, **kwargs)


@st.cache_data(ttl=600, show_spinner="Loading Groq models…")
def fetch_groq_models(api_key: str) -> tuple[list[str], str]:
    """Live Groq chat models for this key (cached 10 min); returns (models, error message)."""
    try:
        return list_groq_models(api_key), ""
    except LLMError as exc:
        return [], str(exc)


def sidebar_llm() -> LLMSettings:
    st.sidebar.markdown("### 🤖 LLM settings")
    mode = persistent_widget(st.sidebar.radio, "LLM mode", "llm_mode", FREE_MODE, options=[FREE_MODE, PAID_MODE],
                             help="Both modes use the same dataset, FAISS index and retrieval pipeline. "
                                  "Only the language model provider changes.")
    if mode == FREE_MODE:
        provider, env_key, models, placeholder = "groq", GROQ_API_KEY_ENV, GROQ_MODELS, "gsk_…"
    else:
        provider, env_key, models, placeholder = "openai", OPENAI_API_KEY_ENV, OPENAI_MODELS, "sk-…"
    label = "Groq API key" if provider == "groq" else "OpenAI API key"
    typed = persistent_widget(st.sidebar.text_input, label, f"{provider}_api_key", "", type="password",
                              placeholder=("Using key from .env" if env_key else placeholder),
                              help="Stored only in this browser session. Never logged or shown in chat.")
    api_key = (typed or "").strip() or env_key
    default_model, live_models = models[0], None
    if provider == "groq":
        default_model = GROQ_DEFAULT_MODEL
        if api_key:
            live_models, list_error = fetch_groq_models(api_key)
            if live_models:
                models = sorted(live_models, key=lambda m: m != GROQ_DEFAULT_MODEL)
            else:
                st.sidebar.error(f"Could not load your Groq model list. {list_error or 'No chat models returned.'} "
                                 "Showing the built-in list instead.")
    if default_model not in models:
        default_model = models[0]
    if st.session_state.get(f"_keep_{provider}_model") not in models:  # e.g. a decommissioned model was saved
        st.session_state[f"_keep_{provider}_model"] = default_model
    model = persistent_widget(st.sidebar.selectbox, "Model", f"{provider}_model", default_model, options=models)
    custom = persistent_widget(st.sidebar.text_input, "Custom model ID (optional)", f"{provider}_model_custom", "",
                               placeholder="overrides the selection above")
    if live_models and (custom or "").strip() and custom.strip() not in live_models:
        st.sidebar.warning(f"`{custom.strip()}` is not in your Groq key's model list, so requests will likely fail.")
    settings = LLMSettings(mode=mode, model=(custom or "").strip() or model, api_key=api_key)

    if not api_key:
        st.sidebar.info(f"No {settings.provider} key yet — product search still works; AI-written replies are off.")
    elif not typed and env_key:
        st.sidebar.caption(f"Using {provider.upper()}_API_KEY from .env")
    if st.sidebar.button("Test connection", use_container_width=True, disabled=not api_key):
        try:
            with st.sidebar:
                with st.spinner("Contacting provider…"):
                    test_connection(settings)
            st.sidebar.success(f"Connected to {settings.label}")
        except LLMError as exc:
            st.sidebar.error(str(exc))
    return settings


def sidebar_filters(df) -> SearchFilters | None:
    with st.sidebar.expander("🔎 Product filters (optional)"):
        st.caption("Applied to every search unless your message says otherwise.")
        categories = ["Any"] + sorted(df["category"].unique())
        category = st.selectbox("Category", categories, key="f_category")
        scope = df if category == "Any" else df[df["category"] == category]
        subcategory = st.selectbox("Subcategory", ["Any"] + sorted(scope["subcategory"].unique()), key="f_subcategory",
                                   disabled=category == "Any")
        brands = st.multiselect("Brand", sorted(scope["brand"].unique()), key="f_brands")
        # Round the slider's top up to a whole step: a ₹73,919.50 maximum with ₹100 steps used to snap to
        # ₹73,900, which then looked like a deliberate "under ₹73,900" filter on every search.
        max_price = float(math.ceil(df["sale_price"].max() / PRICE_SLIDER_STEP) * PRICE_SLIDER_STEP)
        price = st.slider("Sale price (₹)", 0.0, max_price, (0.0, max_price), step=float(PRICE_SLIDER_STEP), key="f_price")
        min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, step=0.5, key="f_rating")
        colors = st.multiselect("Colour", sorted({c for v in df["_colors"] for c in v}), key="f_colors",
                                format_func=str.title)
        sizes = st.multiselect("Size", sorted({s for v in df["sizes"] for s in v.split(", ")}), key="f_sizes")
        in_stock = st.checkbox("In stock only", key="f_stock")
        c1, c2 = st.columns(2)
        if c1.button("Browse", use_container_width=True, type="primary"):
            st.session_state["pending_action"] = {"type": "browse"}
        if c2.button("Reset", use_container_width=True):
            for k in ("f_category", "f_subcategory", "f_brands", "f_price", "f_rating", "f_colors", "f_sizes", "f_stock"):
                st.session_state.pop(k, None)
            st.rerun()

    f = SearchFilters(
        category=None if category == "Any" else category,
        subcategory=None if subcategory == "Any" or category == "Any" else subcategory,
        brands=brands,
        min_price=price[0] if price[0] > 0 else None,
        max_price=price[1] if price[1] < df["sale_price"].max() else None,   # only a real cap counts
        min_rating=min_rating or None,
        colors=[c.title() for c in colors],
        sizes=sizes,
        in_stock_only=in_stock,
    )
    return None if f.is_empty() else f


def sidebar_data(pipeline: dict):
    df = pipeline["df"]
    with st.sidebar.expander("🗂️ Data & index"):
        st.markdown(f"**{len(df):,} products** · {df['category'].nunique()} categories · "
                    f"{df['subcategory'].nunique()} subcategories · {df['brand'].nunique()} brands")
        st.caption(pipeline["faiss"].status if pipeline["faiss"] else "FAISS index unavailable")
        if st.button("Rebuild FAISS index", use_container_width=True,
                     help="Re-embed all products. The index also rebuilds automatically when the CSV changes."):
            st.session_state["rebuild_token"] = st.session_state.get("rebuild_token", 0) + 1
            st.rerun()


# --------------------------------------------------------------------------- chat rendering
def render_message(index: int, msg: dict, df, cart: Cart):
    is_user = msg["role"] == "user"
    with st.chat_message(msg["role"], avatar="🧑" if is_user else "🛍️"):
        st.markdown(msg.get("content", "").replace("$", "\\$"))
        if is_user:
            return
        for note in msg.get("notes", []):
            st.info(note, icon="ℹ️")
        if msg.get("error"):
            st.warning(f"{msg['error']} Showing a verified catalog summary instead.", icon="⚠️")
        if msg.get("category_options"):
            render_category_buttons(msg["category_options"], index)
        if msg.get("product_ids"):
            render_product_grid(df, msg["product_ids"], cart, key_prefix=f"m{index}")
        if msg.get("compare_ids"):
            render_comparison(df, msg["compare_ids"])
        if msg.get("followups"):
            render_followups(msg["followups"], index)
        if msg.get("llm_label"):
            st.caption(f"✨ Written by {msg['llm_label']} · grounded in catalog data")
        elif msg.get("llm_hint"):
            st.caption(msg["llm_hint"])


def action_label(action: dict) -> str:
    if action["type"] == "message":
        return action["text"]
    if action["type"] == "category":
        option = action["option"]
        return option.get("subcategory") or option["category"]
    return "Browse products with my sidebar filters"


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    if "toast" in st.session_state:
        st.toast(st.session_state.pop("toast"))

    try:
        pipeline = load_pipeline()
    except DatasetError as exc:
        st.error(f"📂 {exc}")
        st.stop()
    except Exception as exc:  # unexpected startup failure
        logging.exception("Startup failed")
        st.error(f"Could not start the assistant: {exc}")
        st.stop()

    df, assistant = pipeline["df"], pipeline["assistant"]
    convo = ConversationManager(st.session_state)
    cart = Cart(st.session_state, df)
    st.session_state.setdefault("compare_ids", [])

    st.sidebar.markdown("## 🛍️ Shopping Assistant")
    llm = sidebar_llm()
    filters = sidebar_filters(df)
    st.sidebar.markdown("---")
    render_cart(cart)
    sidebar_data(pipeline)

    # Header
    head, mode_col, clear_col = st.columns([6, 2.2, 1.3], vertical_alignment="center")
    head.markdown("<p class='app-title'>🛍️ <span>AI Shopping Assistant</span></p>"
                  f"<div class='app-sub'>Grounded answers from {len(df):,} real catalog products · prices in ₹</div>",
                  unsafe_allow_html=True)
    pill_class = "paid" if llm.mode == PAID_MODE else ""
    dot = "ready" if llm.ready else ""
    status = llm.label if llm.ready else f"{llm.provider} · no API key"
    mode_col.markdown(f"<div class='mode-pill {pill_class}'><span class='dot {dot}'></span>"
                      f"{'Paid' if llm.mode == PAID_MODE else 'Free'} mode · {status}</div>", unsafe_allow_html=True)
    if clear_col.button("🧹 Clear Chat", use_container_width=True):
        convo.clear()
        st.session_state["compare_ids"] = []
        st.rerun()
    if pipeline["warning"]:
        st.warning(pipeline["warning"])
    retriever = assistant.retriever
    st.caption(("🧠 " if retriever.semantic_available else "🔤 ") + f"Search mode: {retriever.mode_label}")

    # Compare tray (products picked with the Compare button)
    selected = [p for p in st.session_state["compare_ids"] if p in df.index]
    if selected:
        with st.expander(f"⚖️ Compare selected products ({len(selected)})", expanded=len(selected) >= 2):
            render_comparison(df, selected)
            c1, c2, _ = st.columns([1.2, 1, 3])
            if c1.button("Ask assistant to compare", disabled=len(selected) < 2, use_container_width=True):
                st.session_state["pending_action"] = {"type": "message", "text": "Compare the selected products"}
            if c2.button("Clear selection", use_container_width=True):
                st.session_state["compare_ids"] = []
                st.rerun()

    if not convo.messages:
        convo.add_assistant(assistant.welcome())
    for i, msg in enumerate(convo.messages):
        render_message(i, msg, df, cart)

    prompt = st.chat_input("Ask about products, e.g. “Nike running shoes under ₹2,000” or “laptops rated above 4”")
    action = {"type": "message", "text": prompt} if prompt else st.session_state.pop("pending_action", None)
    if not action:
        return
    st.session_state.pop("pending_action", None)

    index = convo.add_user(action_label(action))
    render_message(index, convo.messages[index], df, cart)
    with st.chat_message("assistant", avatar="🛍️"):
        with st.spinner("Searching the catalog…"):
            try:
                if action["type"] == "category":
                    response = assistant.handle_category_selection(action["option"], convo, llm, filters)
                elif action["type"] == "browse":
                    response = assistant.browse(filters or SearchFilters(), convo, llm)
                else:
                    response = assistant.handle_message(action["text"], convo, cart, llm, filters)
            except Exception:
                logging.exception("Failed to handle message")
                response = AssistantResponse(text="Sorry, something went wrong while handling that request. "
                                                  "Please try rephrasing it.")
    convo.add_assistant(response)
    st.rerun()


main()
