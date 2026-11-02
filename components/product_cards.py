"""Product cards and the comparison table, rendered from dataset rows only."""

import hashlib
import math
from html import escape

import pandas as pd
import streamlit as st

from config import CAROUSEL_CARDS_PER_PAGE, LOW_STOCK_THRESHOLD
from utils.helpers import format_price, split_list


def _stock_badge(row) -> str:
    qty = int(row["stock_quantity"])
    if not row["in_stock"] or qty <= 0:
        return "<span class='badge badge-out'>Out of stock</span>"
    if qty <= LOW_STOCK_THRESHOLD:
        return f"<span class='badge badge-low'>Only {qty} left</span>"
    return f"<span class='badge badge-in'>In stock · {qty}</span>"


def _chips(values: str) -> str:
    return "".join(f"<span class='chip'>{escape(v)}</span>" for v in split_list(values))


def card_html(row) -> str:
    discount = int(row["discount_percent"])
    price_html = f"<span class='sale'>{format_price(row['sale_price'])}</span>"
    if discount > 0:
        price_html += (f"<span class='mrp'>{format_price(row['price'])}</span>"
                       f"<span class='badge badge-off'>{discount}% off</span>")
    stars = "★" * int(round(row["rating"])) + "☆" * (5 - int(round(row["rating"])))
    return f"""
<div class="product-card">
  <div class="card-top">
    <span class="pid">{escape(row['product_id'])}</span>
    {_stock_badge(row)}
  </div>
  <div class="pname">{escape(row['name'])}</div>
  <div class="pmeta">{escape(row['brand'])} · {escape(row['category'])} › {escape(row['subcategory'])}</div>
  <div class="price-row">{price_html}</div>
  <div class="rating"><span class="stars">{stars}</span> {row['rating']:.1f} <span class="muted">({int(row['review_count']):,} reviews)</span></div>
  <div class="desc">{escape(row['description'])}</div>
  <div class="attr"><b>Sizes</b> {_chips(row['sizes'])}</div>
  <div class="attr"><b>Colours</b> {_chips(row['colors'])}</div>
  <div class="attr"><b>Material</b> <span class="muted">{escape(row['material'])}</span></div>
</div>"""


def _add_to_cart(cart, product_id: str):
    """One click adds a product sold in a single size and colour; otherwise the assistant asks which ones first."""
    if cart.needs_choice(product_id):
        st.session_state["pending_action"] = {"type": "message", "text": f"Add {product_id} to cart"}
        return
    ok, msg = cart.add(product_id)
    st.session_state["toast"] = ("✅ " if ok else "⚠️ ") + msg


def _toggle_compare(product_id: str):
    selected = st.session_state.setdefault("compare_ids", [])
    if product_id in selected:
        selected.remove(product_id)
    else:
        selected.append(product_id)


def carousel_state_key(key_prefix: str, product_ids: list[str]) -> str:
    """Page index per message *and* result set, so new results start at page 1 and old ones keep their place."""
    digest = hashlib.md5("|".join(product_ids).encode()).hexdigest()[:8]
    return f"_carousel_page_{key_prefix}_{digest}"


def _shift_page(state_key: str, delta: int, pages: int):
    st.session_state[state_key] = max(0, min(pages - 1, st.session_state.get(state_key, 0) + delta))


def render_product_grid(df: pd.DataFrame, product_ids: list[str], cart, key_prefix: str,
                        columns: int = CAROUSEL_CARDS_PER_PAGE):
    """Horizontal carousel: `columns` cards per page with ← / → controls above the cards.

    Only the visible page is rendered. On tablet/mobile the CSS in app.py shows 2 / 1 card(s) at a time,
    with the rest of the page reachable by swiping.
    """
    rows = [df.loc[pid] for pid in product_ids if pid in df.index]
    if not rows:
        return
    pages = math.ceil(len(rows) / columns)
    state_key = carousel_state_key(key_prefix, [r["product_id"] for r in rows])
    page = min(st.session_state.get(state_key, 0), pages - 1)
    start = page * columns
    window = rows[start: start + columns]

    n = len(rows)
    status = "1 matching product" if n == 1 else f"{n} matching products · Showing {start + 1}–{start + len(window)} of {n}"
    if pages > 1:
        with st.container(key=f"cnav_{key_prefix}"):
            left, mid, right = st.columns([1, 6, 1], vertical_alignment="center")
            left.button("←", key=f"{key_prefix}_prev", on_click=_shift_page, args=(state_key, -1, pages),
                        disabled=page == 0, use_container_width=True, help="Previous products")
            dots = "".join(f"<span class='cdot{' on' if i == page else ''}'></span>" for i in range(pages)) \
                if pages <= 12 else ""   # long lists (e.g. the whole catalog) show the position text only
            mid.markdown(f"<div class='carousel-status'>{status} <span class='cdots'>{dots}</span></div>",
                         unsafe_allow_html=True)
            right.button("→", key=f"{key_prefix}_next", on_click=_shift_page, args=(state_key, 1, pages),
                         disabled=page >= pages - 1, use_container_width=True, help="More products")
    else:
        st.markdown(f"<div class='carousel-status'>{status}</div>", unsafe_allow_html=True)

    selected = st.session_state.get("compare_ids", [])
    with st.container(key=f"carousel_{key_prefix}"):
        cols = st.columns(columns)
        for col, row in zip(cols, window):
            pid = row["product_id"]
            with col, st.container(key=f"pcard_{key_prefix}_{pid}"):
                st.markdown(card_html(row), unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                out = not row["in_stock"] or int(row["stock_quantity"]) <= 0
                b1.button("🛒 Add to Cart", key=f"{key_prefix}_add_{pid}", on_click=_add_to_cart, args=(cart, pid),
                          disabled=out, use_container_width=True, type="primary")
                b2.button("✓ Comparing" if pid in selected else "⚖️ Compare", key=f"{key_prefix}_cmp_{pid}",
                          on_click=_toggle_compare, args=(pid,), use_container_width=True)


def comparison_frame(df: pd.DataFrame, product_ids: list[str]) -> pd.DataFrame:
    rows = df.loc[[p for p in product_ids if p in df.index]]
    best_price, best_rating, best_disc = rows["sale_price"].min(), rows["rating"].max(), rows["discount_percent"].max()
    table = {}
    for _, r in rows.iterrows():
        table[f"{r['name']} ({r['product_id']})"] = {
            "Brand": r["brand"],
            "Category": f"{r['category']} › {r['subcategory']}",
            "Sale price": format_price(r["sale_price"]) + (" 🏆" if r["sale_price"] == best_price else ""),
            "Original price": format_price(r["price"]),
            "Discount": f"{int(r['discount_percent'])}%" + (" 🏆" if r["discount_percent"] == best_disc else ""),
            "Rating": f"{r['rating']:.1f} ({int(r['review_count']):,} reviews)" + (" 🏆" if r["rating"] == best_rating else ""),
            "Material": r["material"],
            "Sizes": r["sizes"],
            "Colours": r["colors"],
            "Features": r["tags"],
            "Stock": f"{int(r['stock_quantity'])} units" if r["in_stock"] and r["stock_quantity"] > 0 else "Out of stock",
        }
    return pd.DataFrame(table)


def render_comparison(df: pd.DataFrame, product_ids: list[str]):
    if len(product_ids) < 2:
        st.info("Select at least two products to compare.")
        return
    st.dataframe(comparison_frame(df, product_ids), use_container_width=True)
    st.caption("🏆 marks the best value in each row.")
