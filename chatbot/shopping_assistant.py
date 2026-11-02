"""The shopping assistant: routes a message to the right action and builds a grounded reply.

Product facts shown to the user (cards, tables, notes, cart totals) always come
straight from the DataFrame. The LLM only writes the short conversational text,
and that text is checked against the retrieved rows before it is shown.
"""

import logging
import math
import re

import pandas as pd

from chatbot.conversation_manager import AssistantResponse, ConversationManager, ShoppingContext
from chatbot.intent_detector import IntentDetector, ParsedQuery
from components.shopping_cart import Cart
from config import CAROUSEL_CARDS_PER_PAGE
from rag.llm.llm_factory import LLMError, LLMSettings, generate
from rag.llm.system_prompt import build_messages, find_ungrounded_claims, llm_context_rows
from rag.retrieval.product_filter import (CONSTRAINT_GROUPS, SORT_OPTIONS, SearchFilters, active_groups, apply_filters,
                                      without_group)
from rag.retrieval.product_retriever import ProductRetriever, RetrievalResult, SemanticSearchError
from utils.helpers import format_price

logger = logging.getLogger(__name__)


# What the shopper can say (or click) to drop one filter group; see clear_rules in the intent detector.
CLEAR_PHRASES = {"brand": "Any brand", "colour": "Any colour", "size": "Any size", "rating": "Any rating",
                 "material": "Any material", "discount": "Any discount", "stock": "Include out-of-stock",
                 "price": "Any price"}


def category_option(label: str, category: str, subcategory: str | None, action: str) -> dict:
    return {"label": label, "category": category, "subcategory": subcategory, "action": action}


class ShoppingAssistant:
    def __init__(self, df: pd.DataFrame, retriever: ProductRetriever, detector: IntentDetector):
        self.df = df
        self.retriever = retriever
        self.detector = detector

    # ================================================================ public
    def welcome(self) -> AssistantResponse:
        return AssistantResponse(
            text=(f"👋 Hi! I'm your AI Shopping Assistant. I can search our catalog of **{len(self.df):,} products**, "
                  "filter by price, brand, size, colour and rating, compare items and manage your cart.\n\n"
                  "Pick a category to start, or just tell me what you need — e.g. *“running shoes under ₹2,000”*."),
            kind="categories",
            category_options=self.main_category_options(),
        )

    def main_category_options(self) -> list[dict]:
        counts = self.df["category"].value_counts()
        return [category_option(f"{cat} ({counts[cat]})", cat, None, "subcategories")
                for cat in self.detector.categories]

    def subcategory_options(self, category: str) -> list[dict]:
        counts = self.df[self.df["category"] == category]["subcategory"].value_counts()
        options = [category_option(f"{sub} ({counts[sub]})", category, sub, "products")
                   for sub in self.detector.subcats_by_category.get(category, [])]
        options.append(category_option(f"All {category} ({int(counts.sum())})", category, None, "products"))
        return options

    @staticmethod
    def _search_unavailable(exc: SemanticSearchError) -> AssistantResponse:
        return AssistantResponse(
            text=(f"⚠️ **Product search is unavailable:** semantic search (FAISS) failed — {exc}. "
                  "I won't switch to keyword search, so no products are shown. Check the terminal log, then use "
                  "**Sidebar → Data & index → Rebuild FAISS index** or restart the app with the project's .venv."),
            error="Semantic search unavailable.")

    def handle_message(self, text: str, convo: ConversationManager, cart: Cart, llm: LLMSettings,
                       sidebar_filters: SearchFilters | None = None) -> AssistantResponse:
        try:
            return self._handle_message(text, convo, cart, llm, sidebar_filters)
        except SemanticSearchError as exc:
            return self._search_unavailable(exc)

    def _handle_message(self, text: str, convo: ConversationManager, cart: Cart, llm: LLMSettings,
                        sidebar_filters: SearchFilters | None = None) -> AssistantResponse:
        text = (text or "").strip()
        if not text:
            return AssistantResponse("Please type what you're looking for, e.g. *“black casual shoes”*.")
        if len(text) > 500:
            return AssistantResponse("That message is quite long — please describe what you're looking for in a sentence or two.")

        pq = self.detector.parse(text)
        ctx = convo.context
        if ctx is not None and ctx.pending and not self._is_clarification_reply(pq):
            ctx.pending = False   # a question left unanswered expires with the next message
        purchase = cart.state.get("pending_purchase")
        if purchase:
            if self._is_variant_reply(pq):
                return self._choose_variant(purchase["pid"], pq, cart)
            if pq.intent != "cart_add":
                cart.state.pop("pending_purchase", None)   # the shopper moved on

        if pq.intent == "greeting":
            response = self.welcome()
            response.text = "Hello! 👋 What are you shopping for today? Choose a category or describe the product you need."
            return response
        if pq.intent == "thanks":
            return AssistantResponse("You're welcome! 😊 Let me know if you'd like to look for anything else — "
                                     "your cart and current search are still here.")
        if pq.intent == "help":
            response = self.welcome()
            response.text = ("I can help you **find products** (try *“laptops with rating above 4”*). Each message is a "
                             "new search, so put everything in one message, e.g. *“black running shoes under ₹2,000, "
                             "cheapest first”*. I can also **compare** items (*“compare the first two”*) and manage "
                             "your **cart** (*“add the first one to cart”*, *“show my cart”*). We carry these categories:")
            return response
        if pq.intent == "browse_all":
            convo.context = None
            return self._run_search(text, pq.filters, "", convo, llm, sidebar_filters, pq=pq, browse_all=True)
        if pq.intent.startswith("cart_"):
            return self._handle_cart(pq, ctx, cart)
        if pq.intent == "compare":
            return self._handle_compare(pq, ctx, convo, llm)
        if pq.intent == "detail":
            return self._handle_detail(pq, ctx, convo, llm)
        if pq.reset_context and not pq.has_scope and not pq.filters.has_constraints():
            convo.context = None
            response = self.welcome()
            response.text = "Done — I've cleared your filters. What would you like to look for?"
            return response

        if pq.price_unreadable and not pq.filters.has_price():
            return AssistantResponse(
                "I couldn't read the price in your message, so I haven't applied a price filter. Please write it like "
                "*“under ₹2,000”*, *“between ₹500 and ₹1,000”* or *“around ₹300”*.",
                followups=["Under ₹500", "Under ₹2,000", "Between ₹500 and ₹1,000"])
        if pq.price_question is not None:
            return self._ask_price_meaning(pq, ctx, convo)

        if pq.category_options:
            return self._ambiguous_scope(pq)

        if ctx and pq.relative == "more" and not pq.has_scope:
            n = len(ctx.last_product_ids)
            return AssistantResponse(
                f"All **{n}** matching product{'s are' if n != 1 else ' is'} already in the carousel above — "
                "use the ← → arrows to browse every one of them." if n else
                "There are no results to page through yet — tell me what you're looking for.")
        if ctx and pq.relative == "other_colors" and not pq.has_scope and not pq.filters.colors:
            return self._color_options(ctx)

        resolved = self._resolve_search(pq, ctx)
        if resolved is None:
            if ctx and pq.refers_to_results and ctx.last_product_ids:
                return self._answer_about_results(text, ctx, convo, llm)
            return self._semantic_or_clarify(text, pq, convo, llm, sidebar_filters)
        filters, semantic_query, is_new = resolved

        # Broad request ("show me shoes") -> ask which subcategory with dynamic buttons.
        if (is_new and filters.category and not filters.subcategory and not filters.has_constraints()
                and not pq.leftover_terms and not pq.unknown_brands and not pq.unsupported_sizes):
            convo.context = ShoppingContext(filters=filters.copy(), semantic_query=semantic_query)
            noun = pq.category_term if pq.category_term and pq.category_term != filters.category.lower() else filters.category
            return AssistantResponse(
                text=f"Sure! What type of **{noun}** are you looking for?",
                kind="categories",
                category_options=self.subcategory_options(filters.category),
            )
        return self._run_search(text, filters, semantic_query, convo, llm, sidebar_filters, pq=pq)

    def handle_category_selection(self, option: dict, convo: ConversationManager, llm: LLMSettings,
                                  sidebar_filters: SearchFilters | None = None) -> AssistantResponse:
        try:
            return self._handle_category_selection(option, convo, llm, sidebar_filters)
        except SemanticSearchError as exc:
            return self._search_unavailable(exc)

    def _handle_category_selection(self, option: dict, convo: ConversationManager, llm: LLMSettings,
                                   sidebar_filters: SearchFilters | None = None) -> AssistantResponse:
        category, subcategory = option["category"], option.get("subcategory")
        if option.get("action") == "subcategories":
            return AssistantResponse(
                text=f"Great choice! Which kind of **{category}** would you like?",
                kind="categories",
                category_options=self.subcategory_options(category),
            )
        # A fresh search for exactly the clicked category: nothing from earlier messages is carried over.
        label = " ".join(filter(None, [subcategory, category]))
        return self._run_search(label, SearchFilters(category=category, subcategory=subcategory), "", convo, llm,
                                sidebar_filters)

    def browse(self, filters: SearchFilters, convo: ConversationManager, llm: LLMSettings) -> AssistantResponse:
        try:
            return self._run_search("Browse with sidebar filters", filters, "", convo, llm, None, browse_all=True)
        except SemanticSearchError as exc:
            return self._search_unavailable(exc)

    # ============================================================ resolution
    @staticmethod
    def _intent_text(pq: ParsedQuery) -> str:
        """What the product request is about, without prices, filters or chit-chat (used for semantic search)."""
        return pq.filters.product_name or " ".join(pq.leftover_terms)

    @staticmethod
    def _is_clarification_reply(pq: ParsedQuery) -> bool:
        """A bare answer to the assistant's question or suggestion: only a price ("Exactly ₹100", "Under ₹300")
        or a "drop this filter" phrase ("Any colour", "Any price") — no product, category or other filter."""
        f = pq.filters
        only_price = set(f.constraint_fields()) <= {"min_price", "max_price", "target_price", "budget"}
        return (only_price and not pq.has_scope and not pq.leftover_terms and not pq.unknown_brands
                and not pq.unsupported_sizes and (f.has_price() or bool(pq.cleared_fields)))

    def _resolve_search(self, pq: ParsedQuery, ctx: ShoppingContext | None):
        """Every message is a fresh, independent search: only the filters stated in this message are applied.

        The one exception: while the assistant is waiting for an answer to its own question (ctx.pending, e.g.
        "books worth ₹100?" or "no shirts under ₹200 — under ₹300?"), a bare answer such as "Exactly ₹100" is
        applied to that pending request only — never to anything searched before it.
        Returns (filters, semantic_query, is_new) or None when nothing searchable was found.
        """
        new = pq.filters
        if ctx is not None and ctx.pending and self._is_clarification_reply(pq):
            pq.context_used = True
            return self._apply_relative(ctx.filters.merged_with(new), pq), ctx.semantic_query, False
        if pq.has_scope:
            return new, self._intent_text(pq), True
        if pq.unknown_brands and not new.has_constraints():
            return None  # e.g. just "Nike": nothing in the catalog to show
        if new.has_constraints() or pq.unsupported_sizes:
            # "pen under ₹500": the words are searched semantically, so the filters alone never decide relevance.
            return new, self._intent_text(pq), True
        return None

    def _ask_price_meaning(self, pq: ParsedQuery, ctx: ShoppingContext | None,
                           convo: ConversationManager) -> AssistantResponse:
        """"worth ₹300" / "shoes 2000": confirm whether it's a cap, an exact price or a rough target."""
        amount = format_price(pq.price_question)
        # Only this message's request is held for the answer ("Exactly ₹300"); earlier searches are not.
        resolved = self._resolve_search(pq, None)
        filters, semantic_query = (resolved[0], resolved[1]) if resolved else (SearchFilters(), self._intent_text(pq))
        convo.context = ShoppingContext(filters=filters, semantic_query=semantic_query, pending=True)
        what = f" for **{filters.describe(include_sort=False)}**" if not filters.is_empty() else ""
        return AssistantResponse(
            f"Just to be sure{what}: do you mean **up to {amount}**, **exactly {amount}** or **around {amount}**?",
            followups=[f"Up to {amount}", f"Exactly {amount}", f"Around {amount}"])   # "up to" is <=, as asked

    def _apply_relative(self, f: SearchFilters, pq: ParsedQuery) -> SearchFilters:
        """Apply "any colour"-style clearing and "cheaper"/"higher rated" ordering to a pending request.

        Prices or ratings are never derived from products shown earlier: that would carry old results forward.
        """
        for name in pq.cleared_fields:
            if name == "price":
                f = f.clear_price()
            elif name in ("colors", "sizes", "brands", "materials"):
                setattr(f, name, [])
            elif name == "in_stock_only":
                f.in_stock_only = False
            else:
                setattr(f, name, None)
        if pq.relative == "cheaper":
            f.sort_by = "price_asc"
        elif pq.relative == "higher_rated":
            f.sort_by = "rating_desc"
        return f

    @staticmethod
    def _with_sidebar(filters: SearchFilters, sidebar: SearchFilters | None) -> tuple[SearchFilters, list[str]]:
        """Sidebar filters act as defaults for anything the conversation didn't specify.

        Returns the effective filters and a description of what the sidebar added, so the reply can say so.
        """
        if sidebar is None:
            return filters, []
        f = filters.copy()
        added = SearchFilters()
        if sidebar.category and not f.category:
            f.category = added.category = sidebar.category
            if sidebar.subcategory and not f.subcategory:
                f.subcategory = added.subcategory = sidebar.subcategory
        for name in ("brands", "colors", "sizes"):
            if getattr(sidebar, name) and not getattr(f, name):
                setattr(f, name, list(getattr(sidebar, name)))
                setattr(added, name, list(getattr(sidebar, name)))
        if sidebar.has_price() and not f.has_price():   # price is one unit: the conversation's price wins outright
            f.min_price = added.min_price = sidebar.min_price
            f.max_price = added.max_price = sidebar.max_price
        if sidebar.min_rating is not None and f.min_rating is None:
            f.min_rating = added.min_rating = sidebar.min_rating
        if sidebar.in_stock_only and not f.in_stock_only:
            f.in_stock_only = added.in_stock_only = True
        return f, ([added.describe(include_sort=False)] if not added.is_empty() else [])

    # ================================================================ search
    def _run_search(self, user_text: str, filters: SearchFilters, semantic_query: str, convo: ConversationManager,
                    llm: LLMSettings, sidebar_filters: SearchFilters | None, pq: ParsedQuery | None = None,
                    browse_all: bool = False) -> AssistantResponse:
        """Run one search and describe it from ONE final list (result.products).

        That list feeds the count, the summary, the cheapest/highest-rated lines, the carousel and the
        conversation context. Filters are never relaxed: no matches means zero products plus, at most, a
        clearly labelled suggestion the shopper can choose.
        """
        effective, sidebar_added = self._with_sidebar(filters, sidebar_filters)
        notes = self._notes(pq, effective)
        if sidebar_added:
            notes.append(f"Sidebar filters are also applied: **{sidebar_added[0]}**. Change them in the sidebar.")

        result = self.retriever.retrieve(semantic_query, effective, browse_all=browse_all)
        if result.relevant.empty and semantic_query.strip() and pq is not None and not pq.has_scope:
            return self._not_carried(pq)

        # Remember what was asked (plus a semantically inferred subcategory), never the sidebar's values.
        remembered = filters.copy(category=result.filters.category, subcategory=result.filters.subcategory) \
            if result.inferred_subcategory else filters.copy()
        unsupported = bool(pq and (pq.unsupported_sizes or pq.unknown_brands))
        if result.total == 0 or unsupported:
            if not browse_all:  # the suggestion offered below ("Under ₹300", "Any colour") applies to this request
                convo.context = ShoppingContext(filters=remembered, semantic_query=semantic_query, pending=True)
            if effective.price_type == "exact" and not unsupported:
                amount = format_price(effective.exact_price)
                what = effective.product_name or effective.copy(target_price=None).describe(include_sort=False)
                return AssistantResponse(
                    text=(f"**No exact-price match:** nothing for **{what}** costs exactly **{amount}**. "
                          f"Would you like products at nearby prices (around {amount}) or anything under {amount}?"),
                    followups=[f"Around {amount}", f"Under {amount}", "Any price"], notes=notes)
            return self._no_matches(result, effective, semantic_query, notes)

        convo.context = ShoppingContext(filters=remembered, semantic_query=semantic_query,
                                        last_product_ids=list(result.products.index), total=result.total,
                                        browse_all=browse_all)
        template = self._search_summary(result)
        summary_for_llm = template + ("\nNOTES: " + " ".join(notes) if notes else "")
        task = ("Write a short, friendly reply to the customer's message that recommends the best options among these "
                "products, explaining briefly why they fit the request. Product cards are shown below your reply.")
        response = AssistantResponse(text=template, kind="products", product_ids=list(result.products.index),
                                     notes=notes, followups=self._followups(result))
        lo, hi = effective.price_window()
        self._llm_text(response, llm, convo, user_text, summary_for_llm, result.products, task,
                       extra_numbers=[v for v in (lo, hi, effective.target_price, effective.min_rating,
                                                     effective.min_discount) if v],
                       counts=[result.total, result.name_matches])
        return response

    def _no_matches(self, result: RetrievalResult, requested: SearchFilters, semantic_query: str,
                    notes: list[str]) -> AssistantResponse:
        """Zero matching products. Offer (never apply) the single change that would give results."""
        what = requested.describe(include_sort=False)
        if result.inferred_subcategory:
            what = result.filters.describe(include_sort=False)
        noun = (result.filters.subcategory or requested.category or semantic_query or "products").lower()
        text = f"**No matching products found** for **{what}**."
        followups: list[str] = []
        base = result.relevant
        applied = result.filters
        if not base.empty and applied.max_price is not None and applied.target_price is None:
            others = apply_filters(base, applied.copy(max_price=None, max_inclusive=True, budget=False))
            if not others.empty:
                cheapest = float(others["sale_price"].min())
                cap = (math.floor(cheapest / 100) + 1) * 100
                text += (f" The cheapest relevant option ({noun}, other filters kept) costs **{format_price(cheapest)}**. "
                         f"*Suggestion:* would you like to see {noun} under {format_price(cap)}?")
                followups.append(f"Under {format_price(cap)}")
        elif not base.empty and applied.min_price is not None and applied.target_price is None:
            others = apply_filters(base, applied.copy(min_price=None, min_inclusive=True))
            if not others.empty:
                priciest = float(others["sale_price"].max())
                floor_ = math.floor(priciest / 100) * 100
                floor_ = floor_ - 100 if floor_ >= priciest else floor_
                text += (f" The most expensive relevant option ({noun}, other filters kept) costs **{format_price(priciest)}**. "
                         f"*Suggestion:* would you like to see {noun} above {format_price(floor_)}?")
                followups.append(f"Above {format_price(floor_)}")
        if not followups and not base.empty:
            for group in active_groups(applied):
                if group in ("price",):
                    continue
                n = len(apply_filters(base, without_group(applied, group)))
                if n:
                    dropped = SearchFilters(**{k: getattr(applied, k) for k in CONSTRAINT_GROUPS[group]})
                    text += (f" *Suggestion:* {n} product{'s' if n != 1 else ''} match if you drop the "
                             f"**{dropped.describe(include_sort=False)}** filter — want to see them?")
                    followups.append(CLEAR_PHRASES[group])
                    break
        if not followups:
            text += " Try a different search or pick a category:"
            return AssistantResponse(text=text, kind="categories", category_options=self.main_category_options(),
                                     notes=notes)
        return AssistantResponse(text=text, followups=followups + ["Start a new search"], notes=notes)

    def _semantic_or_clarify(self, text: str, pq: ParsedQuery, convo: ConversationManager, llm: LLMSettings,
                             sidebar_filters: SearchFilters | None) -> AssistantResponse:
        """No category or filters were recognised: search on the words themselves, never the whole catalog."""
        if self._off_topic(pq):
            return self._clarify()
        if not pq.leftover_terms:
            # Nothing describes a product; an unknown brand alone ("Nike") must not drive semantic search.
            if pq.unknown_brands:
                return self._not_carried(pq)
            return self._clarify()
        return self._run_search(text, SearchFilters(), self._intent_text(pq), convo, llm, sidebar_filters, pq=pq)

    def _clarify(self) -> AssistantResponse:
        return AssistantResponse(
            text=("I'm a shopping assistant for this store, so I can only help with our products. "
                  "Could you tell me what you're looking for? You can also pick a category:"),
            kind="categories", category_options=self.main_category_options())

    @staticmethod
    def _off_topic(pq: ParsedQuery) -> bool:
        """A general question ("what is the capital of France?") with nothing shopping-related in it."""
        text = pq.text.lower()
        return (re.match(r"^\s*(what|who|when|where|why|how|tell me|can you|are you|is it|do you know|explain|write)\b", text)
                is not None
                and not re.search(r"\b(buy|sell|carry|have|stock|looking for|need|want|find|show|price|cost|cheap)\b", text)
                and not pq.has_scope and not pq.filters.has_constraints() and not pq.unknown_brands)

    def _not_carried(self, pq: ParsedQuery) -> AssistantResponse:
        """Specific request with zero matches: say so plainly, show no products."""
        if pq.unknown_brands:
            what = f"{', '.join(pq.unknown_brands)} products"
        elif self._off_topic(pq):
            return self._clarify()
        else:
            what = " and ".join(pq.leftover_terms)
        return AssistantResponse(
            text=(f"Sorry, we don't currently carry **{what}**. "
                  "Here's what we do have — pick a category to browse:"),
            kind="categories", category_options=self.main_category_options())

    def _ambiguous_scope(self, pq: ParsedQuery) -> AssistantResponse:
        options = []
        for cat, sub in pq.category_options:
            if sub:
                options.append(category_option(f"{sub} · {cat}", cat, sub, "products"))
            else:
                options.append(category_option(f"{cat} (browse)", cat, None, "subcategories"))
        return AssistantResponse(
            text=f"“{pq.category_term}” could mean a few things in our catalog. Which one did you mean?",
            kind="categories", category_options=options)

    def _notes(self, pq: ParsedQuery | None, f: SearchFilters) -> list[str]:
        notes = []
        if pq and pq.unknown_brands:
            carried = sorted(self.df[self.df["category"] == f.category]["brand"].unique()) if f.category else []
            brands = ", ".join(pq.unknown_brands)
            msg = f"We don't carry **{brands}** products."
            if carried:
                msg += f" Brands available in {f.category}: {', '.join(carried)}."
            notes.append(msg)
        if pq and pq.unsupported_sizes:
            order = ("XS", "S", "M", "L", "XL", "XXL", "Standard")
            sizes = ", ".join(sorted(self.detector.sizes, key=lambda s: order.index(s) if s in order else 99))
            notes.append(f"Size **{', '.join(pq.unsupported_sizes)}** isn't stocked — our products come in sizes {sizes}.")
        return notes

    @staticmethod
    def _search_summary(result: RetrievalResult) -> str:
        """Reply text computed from the one final result list (the same list the carousel shows)."""
        rows = result.products
        n = len(rows)
        f = result.filters
        what = f.describe(include_sort=False)
        lines = []
        if f.product_name:
            k = result.name_matches
            price = f.describe_price()
            lines.append(f"Found **{k}** exact product match{'es' if k != 1 else ''} for **{f.product_name}**"
                         f"{' ' + price if price else ''}.")
            if n > k:
                lines.append(f"**{n} matching products** in total for **{what}**.")
        else:
            lines.append(f"**{n} matching product{'s' if n != 1 else ''}** for **{what}**.")
        order = SORT_OPTIONS[f.sort_by][2] if f.sort_by in SORT_OPTIONS else result.ordering
        if n > 1:
            lines.append(f"Exact matches first, then {order}." if f.product_name and result.name_matches
                         and result.name_matches < n
                         else f"Sorted by {order}.")
        cheapest = rows.loc[rows["sale_price"].idxmin()]
        best = rows.loc[rows["rating"].idxmax()]
        if n == 1:
            lines.append(f"**{cheapest['name']}** costs **{format_price(cheapest['sale_price'])}** "
                         f"(⭐ {cheapest['rating']:.1f}).")
        else:
            lines.append(f"Cheapest: **{cheapest['name']}** at **{format_price(cheapest['sale_price'])}**. "
                         f"Highest rated: **{best['name']}** (⭐ {best['rating']:.1f}).")
        if n > CAROUSEL_CARDS_PER_PAGE:
            lines.append(f"Use ← → to browse all {n}.")
        return " ".join(lines)

    @staticmethod
    def _followups(result: RetrievalResult) -> list[str]:
        suggestions = []
        if result.filters.exact_product:
            ids = list(result.products.index)
            return ["Buy this one"] if len(ids) == 1 else [f"Buy {p}" for p in ids[:6]]
        if len(result.products) >= 2:
            suggestions.append("Compare the top 3")
        suggestions.append("Available in other colors?")
        return suggestions

    # ======================================================= other intents
    def _resolve_ids(self, pq: ParsedQuery, ctx: ShoppingContext | None) -> list[str]:
        ids = [p for p in pq.product_ids if p in self.df.index]
        if not ids and pq.ordinals and ctx and ctx.last_product_ids:
            shown = ctx.last_product_ids
            for n in pq.ordinals:
                idx = len(shown) - 1 if n == -1 else n - 1
                if 0 <= idx < len(shown):
                    ids.append(shown[idx])
        return ids

    def _handle_cart(self, pq: ParsedQuery, ctx: ShoppingContext | None, cart: Cart) -> AssistantResponse:
        if pq.intent == "cart_view":
            return AssistantResponse(text="🛒 **Your cart**\n\n" + cart.summary_text(), kind="cart")
        if pq.intent == "cart_add":
            return self._select_for_purchase(pq, ctx, cart)
        ids = self._resolve_ids(pq, ctx)
        if not ids:
            missing = [p for p in pq.product_ids if p not in self.df.index]
            if missing:
                return AssistantResponse(text=f"I couldn't find product ID {', '.join(missing)} in our catalog.")
            return AssistantResponse(text="Which product? Say e.g. *“remove PROD0001 from cart”*.")
        messages = []
        for pid in ids:
            ok, msg = cart.remove(pid)
            messages.append(("✅ " if ok else "⚠️ ") + msg)
        t = cart.totals()
        messages.append(f"\nCart: **{t['items']} item(s)**, total **{format_price(t['total'])}**.")
        return AssistantResponse(text="\n\n".join(messages), kind="cart")

    def _select_for_purchase(self, pq: ParsedQuery, ctx: ShoppingContext | None, cart: Cart) -> AssistantResponse:
        """"I want to buy this product" / "buy PROD0001" / "add the first one": settle on exactly one product."""
        missing = [p for p in pq.product_ids if p not in self.df.index]
        if missing:
            return AssistantResponse(text=f"I couldn't find product ID {', '.join(missing)} in our catalog.")
        ids = list(dict.fromkeys(self._resolve_ids(pq, ctx)))
        shown = list(dict.fromkeys(ctx.last_product_ids)) if ctx else []
        if not ids and len(shown) == 1:          # "this one" with exactly one product on screen
            ids = shown
        if len(ids) == 1:
            return self._choose_variant(ids[0], pq, cart)
        candidates = ids or shown
        if not candidates:
            return AssistantResponse(text="Which product would you like? Search for it first, or say e.g. "
                                          "*“buy PROD0001”*.")
        listed = "\n".join(f"- **{self.df.loc[p, 'name']}** ({p}) — {format_price(self.df.loc[p, 'sale_price'])}"
                           for p in candidates[:6])
        return AssistantResponse(
            text=f"Which one would you like to buy? Please pick one product:\n\n{listed}",
            followups=[f"Buy {p}" for p in candidates[:6]])

    @staticmethod
    def _is_variant_reply(pq: ParsedQuery) -> bool:
        """A bare answer to "which size / colour?": only sizes and/or colours, nothing else."""
        f = pq.filters
        return (pq.intent == "search" and bool(f.sizes or f.colors or pq.unsupported_sizes)
                and set(f.constraint_fields()) <= {"sizes", "colors"} and not pq.has_scope
                and not pq.leftover_terms and not pq.unknown_brands)

    def _choose_variant(self, pid: str, pq: ParsedQuery, cart: Cart) -> AssistantResponse:
        """Ask for size, then colour (only the options this product has), then add it to the cart."""
        row = self.df.loc[pid]
        name = row["name"]
        if not row["in_stock"] or int(row["stock_quantity"]) <= 0:
            cart.state.pop("pending_purchase", None)
            return AssistantResponse(text=f"Sorry, **{name}** ({pid}) is out of stock.", kind="detail", product_ids=[pid])
        sizes, colours = cart.options(pid)
        pending = cart.state.get("pending_purchase") or {}
        choice = {"pid": pid, "size": None, "colour": None}
        if pending.get("pid") == pid:
            choice.update(size=pending.get("size"), colour=pending.get("colour"))
        problems, refused = [], set()
        for wanted, options, key, label in ((pq.filters.sizes + pq.unsupported_sizes, sizes, "size", "Size"),
                                            (pq.filters.colors, colours, "colour", "Colour")):
            for value in wanted:
                match = next((o for o in options if o.lower() == str(value).lower()), None)
                if match:
                    choice[key] = match
                else:
                    problems.append(f"{label} **{value}** isn't available for {name}.")
                    refused.add(key)
        if len(sizes) == 1:
            choice["size"] = sizes[0]
        if len(colours) == 1:
            choice["colour"] = colours[0]
        for key in refused:   # never swap an unavailable choice for another one silently: ask again
            if choice[key] is not None and choice[key].lower() not in {str(v).lower() for v in
                                                                          pq.filters.sizes + pq.filters.colors}:
                choice[key] = None

        prefix = (" ".join(problems) + " ") if problems else ""
        for key, options, ask in (("size", sizes, "size"), ("colour", colours, "colour")):
            if choice[key] is None:
                cart.state["pending_purchase"] = choice
                buttons = [f"Size {o}" for o in options] if key == "size" else list(options)
                return AssistantResponse(
                    text=f"{prefix}Which **{ask}** would you like for **{name}** ({pid})? Available: {', '.join(options)}.",
                    kind="detail", product_ids=[pid], followups=buttons)

        cart.state.pop("pending_purchase", None)
        ok, msg = cart.add(pid, size=choice["size"], colour=choice["colour"])
        t = cart.totals()
        text = f"{prefix}{'✅ ' if ok else '⚠️ '}{msg}\n\nCart: **{t['items']} item(s)**, total **{format_price(t['total'])}**."
        return AssistantResponse(text=text, kind="cart")

    def _handle_compare(self, pq: ParsedQuery, ctx: ShoppingContext | None, convo: ConversationManager,
                        llm: LLMSettings) -> AssistantResponse:
        ids = self._resolve_ids(pq, ctx)
        selected = convo.state.get("compare_ids", [])
        if len(ids) < 2 and len(selected) >= 2 and not pq.count_hint and not pq.ordinals:
            ids = list(selected)
        if len(ids) < 2 and ctx and ctx.last_product_ids:
            ids = ctx.last_product_ids[: pq.count_hint or min(3, len(ctx.last_product_ids))]
        ids = list(dict.fromkeys(ids))
        if len(ids) < 2:
            return AssistantResponse(text="Please pick at least two products to compare — click **Compare** on the "
                                          "product cards or say e.g. *“compare PROD0001 and PROD0002”*.")
        rows = self.df.loc[ids]
        cheapest = rows.loc[rows["sale_price"].idxmin()]
        best = rows.loc[rows["rating"].idxmax()]
        deal = rows.loc[rows["discount_percent"].idxmax()]
        template = (f"Here's a side-by-side comparison of {len(rows)} products. "
                    f"**{cheapest['name']}** is the cheapest at {format_price(cheapest['sale_price'])}, "
                    f"**{best['name']}** has the highest rating (⭐ {best['rating']:.1f}), and "
                    f"**{deal['name']}** has the biggest discount ({int(deal['discount_percent'])}% off).")
        response = AssistantResponse(text=template, kind="compare", compare_ids=ids,
                                     followups=["Add the first one to cart"])
        task = ("Compare these products for the customer: summarise the key trade-offs (price, rating, discount, "
                "material, sizes) and say which suits which kind of buyer. A comparison table is shown below.")
        self._llm_text(response, llm, convo, pq.text, "Products selected for comparison.", rows, task)
        return response

    def _handle_detail(self, pq: ParsedQuery, ctx: ShoppingContext | None, convo: ConversationManager,
                       llm: LLMSettings) -> AssistantResponse:
        ids = self._resolve_ids(pq, ctx)
        if not ids:
            missing = ", ".join(pq.product_ids) or "that product"
            return AssistantResponse(text=f"I couldn't find {missing} in our catalog. Please check the product ID.")
        rows = self.df.loc[ids]
        convo.context = ShoppingContext(
            filters=SearchFilters(category=rows.iloc[0]["category"], subcategory=rows.iloc[0]["subcategory"]),
            semantic_query=rows.iloc[0]["subcategory"], last_product_ids=ids, total=len(ids))
        names = ", ".join(f"**{n}**" for n in rows["name"])
        response = AssistantResponse(text=f"Here are the details for {names}.", kind="detail", product_ids=ids,
                                     followups=["Add the first one to cart", "Available in other colors?"])
        task = "Describe this product for the customer using only the context: who it suits and its key details."
        self._llm_text(response, llm, convo, pq.text, "Product details requested.", rows, task)
        return response

    def _color_options(self, ctx: ShoppingContext) -> AssistantResponse:
        rows = self.df.loc[[p for p in ctx.last_product_ids if p in self.df.index]]
        if rows.empty:
            return AssistantResponse(text="Search for a product first and I'll list its colour options.")
        lines = [f"- **{r['name']}**: {r['colors']}" for _, r in rows.head(CAROUSEL_CARDS_PER_PAGE * 2).iterrows()]
        all_colors = sorted(set(c for v in self.df[self.df["category"] == rows.iloc[0]["category"]]["_colors"] for c in v))
        text = ("Here are the colour options for the first products in your results:\n\n" + "\n".join(lines) +
                f"\n\nAcross **{rows.iloc[0]['category']}** we have: {', '.join(c.title() for c in all_colors)}. "
                "Say e.g. *“only blue ones”* to filter.")
        return AssistantResponse(text=text, followups=["Only black ones", "Only blue ones", "Any colour"])

    def _answer_about_results(self, text: str, ctx: ShoppingContext, convo: ConversationManager,
                              llm: LLMSettings) -> AssistantResponse:
        rows = self.df.loc[[p for p in ctx.last_product_ids if p in self.df.index]]
        best = rows.loc[rows["rating"].idxmax()]
        cheapest = rows.loc[rows["sale_price"].idxmin()]
        template = (f"Among the products shown, **{best['name']}** has the best rating (⭐ {best['rating']:.1f}) and "
                    f"**{cheapest['name']}** is the most affordable at {format_price(cheapest['sale_price'])}.")
        response = AssistantResponse(text=template, followups=["Compare the top 3"])
        task = "Answer the customer's question about the products listed in the context (the ones shown earlier)."
        self._llm_text(response, llm, convo, text, f"Products currently shown: {ctx.filters.describe()}.", rows, task)
        return response

    # =================================================================== LLM
    def _llm_text(self, response: AssistantResponse, llm: LLMSettings, convo: ConversationManager, query: str,
                  summary: str, rows: pd.DataFrame, task: str, extra_numbers: list[float] | None = None,
                  counts: list[int] | None = None):
        """Replace the template text with an LLM reply when possible; never block on failures.

        `rows` is the complete result list. The prompt carries a bounded sample of it (llm_context_rows), but the
        reply is checked against all of it: any product in the final results is a legitimate thing to mention.
        Both providers get the same summary, rows and grounding rules; only `llm` differs.
        """
        if not llm.ready:
            response.llm_hint = (f"No {llm.provider} API key, so no AI model was called — this reply is generated "
                                 "directly from the catalog. Add a key in the sidebar for AI-written answers.")
            return
        context_rows = llm_context_rows(rows)
        messages = build_messages(query, summary, context_rows, task, convo.history_for_llm(exclude_last=1),
                                  total=len(rows))
        try:
            reply = generate(llm, messages)
        except LLMError as exc:
            response.error = str(exc)
            return
        # Brands the customer or the search summary named (e.g. the unmet "brand PageTurner") may be mentioned.
        named = (query + " " + summary).lower()
        other_brands = [b for b in self.detector.brands if b.lower() not in named]
        problems = find_ungrounded_claims(reply, rows, extra_numbers, catalog_brands=other_brands,
                                          allowed_counts=counts, catalog=self.df)
        if not reply or problems:
            logger.info("Discarded LLM reply with ungrounded claims: %s", problems)
            response.llm_hint = "The AI reply mentioned details not found in the catalog, so a verified summary is shown instead."
            return
        response.text = reply
        response.llm_label = llm.label
