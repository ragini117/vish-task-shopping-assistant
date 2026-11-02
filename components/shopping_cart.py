"""Dummy shopping cart: state logic plus the sidebar cart view."""

from typing import MutableMapping

import pandas as pd
import streamlit as st

from utils.helpers import format_price, split_list


class Cart:
    """Cart stored as {product_id: quantity} in session state, validated against stock.

    The chosen size and colour of each line are kept in {product_id: {"size", "colour"}}. A product with
    more than one size or colour can only be added once both are chosen.
    """

    def __init__(self, state: MutableMapping, products: pd.DataFrame):
        self.state = state
        self.products = products
        state.setdefault("cart", {})
        state.setdefault("cart_variants", {})

    @property
    def items(self) -> dict[str, int]:
        return self.state["cart"]

    @property
    def variants(self) -> dict[str, dict]:
        return self.state["cart_variants"]

    def options(self, product_id: str) -> tuple[list[str], list[str]]:
        """(sizes, colours) this product is sold in."""
        row = self.products.loc[product_id]
        return split_list(row["sizes"]), split_list(row["colors"])

    def needs_choice(self, product_id: str) -> bool:
        sizes, colours = self.options(product_id)
        return len(sizes) > 1 or len(colours) > 1

    @staticmethod
    def variant_label(variant: dict | None) -> str:
        if not variant:
            return ""
        return " · ".join(p for p in (f"Size {variant['size']}" if variant.get("size") else "",
                                      variant.get("colour") or "") if p)

    def _stock(self, product_id: str) -> int:
        row = self.products.loc[product_id]
        return int(row["stock_quantity"]) if bool(row["in_stock"]) else 0

    def add(self, product_id: str, quantity: int = 1, size: str | None = None,
            colour: str | None = None) -> tuple[bool, str]:
        if product_id not in self.products.index:
            return False, f"Product {product_id} does not exist."
        name = self.products.loc[product_id, "name"]
        stock = self._stock(product_id)
        if stock <= 0:
            return False, f"{name} is out of stock."
        sizes, colours = self.options(product_id)
        size = size or (sizes[0] if len(sizes) == 1 else None)
        colour = colour or (colours[0] if len(colours) == 1 else None)
        if (sizes and size not in sizes) or (colours and colour not in colours):
            return False, (f"Please choose a size ({', '.join(sizes)}) and colour ({', '.join(colours)}) "
                           f"for {name} before adding it to your cart.")
        variant = {"size": size, "colour": colour}
        current = self.items.get(product_id, 0)
        if current and self.variants.get(product_id, variant) != variant:
            return False, (f"{name} is already in your cart in {self.variant_label(self.variants[product_id])}. "
                           "Remove it first to choose a different size or colour.")
        if current + quantity > stock:
            return False, f"Only {stock} unit(s) of {name} in stock; you already have {current} in your cart."
        self.items[product_id] = current + quantity
        self.variants[product_id] = variant
        label = self.variant_label(variant)
        return True, f"Added {name}{f' ({label})' if label else ''} to your cart (qty {self.items[product_id]})."

    def set_quantity(self, product_id: str, quantity: int) -> tuple[bool, str]:
        if product_id not in self.items:
            return False, "Item is not in the cart."
        stock = self._stock(product_id)
        quantity = max(1, int(quantity))
        if quantity > stock:
            self.items[product_id] = stock
            return False, f"Only {stock} unit(s) available; quantity set to {stock}."
        self.items[product_id] = quantity
        return True, "Quantity updated."

    def remove(self, product_id: str) -> tuple[bool, str]:
        if self.items.pop(product_id, None) is None:
            return False, f"{product_id} is not in your cart."
        self.variants.pop(product_id, None)
        return True, f"Removed {self.products.loc[product_id, 'name']} from your cart."

    def clear(self):
        self.items.clear()
        self.variants.clear()

    def lines(self) -> pd.DataFrame:
        if not self.items:
            return pd.DataFrame(columns=["product_id", "name", "quantity", "sale_price", "line_total"])
        rows = self.products.loc[list(self.items)].copy()
        rows["quantity"] = [self.items[pid] for pid in rows.index]
        rows["variant"] = [self.variant_label(self.variants.get(pid)) for pid in rows.index]
        rows["line_total"] = rows["sale_price"] * rows["quantity"]
        rows["line_mrp"] = rows["price"] * rows["quantity"]
        return rows

    def totals(self) -> dict:
        lines = self.lines()
        if lines.empty:
            return {"items": 0, "total": 0.0, "mrp": 0.0, "savings": 0.0}
        total, mrp = float(lines["line_total"].sum()), float(lines["line_mrp"].sum())
        return {"items": int(lines["quantity"].sum()), "total": total, "mrp": mrp, "savings": mrp - total}

    def summary_text(self) -> str:
        lines = self.lines()
        if lines.empty:
            return "Your cart is empty."
        parts = [f"- **{r['name']}** ({r['product_id']}{', ' + r['variant'] if r['variant'] else ''}) "
                 f"× {r['quantity']} = {format_price(r['line_total'])}" for _, r in lines.iterrows()]
        t = self.totals()
        parts.append(f"\n**Total: {format_price(t['total'])}** (you save {format_price(t['savings'])})")
        return "\n".join(parts)


def _on_qty_change(cart: Cart, product_id: str, key: str):
    ok, msg = cart.set_quantity(product_id, st.session_state[key])
    if not ok:
        st.session_state[key] = cart.items.get(product_id, 1)
        st.session_state["toast"] = msg


def _on_remove(cart: Cart, product_id: str):
    _, msg = cart.remove(product_id)
    st.session_state["toast"] = msg


def render_cart(cart: Cart):
    """Sidebar cart with quantity selection, removal and totals."""
    totals = cart.totals()
    with st.expander(f"🛒 Cart ({totals['items']} item{'s' if totals['items'] != 1 else ''})", expanded=bool(totals["items"])):
        lines = cart.lines()
        if lines.empty:
            st.caption("Your cart is empty. Use **Add to Cart** on any product.")
            return
        for pid, row in lines.iterrows():
            variant = f" · {row['variant']}" if row["variant"] else ""
            st.markdown(f"**{row['name']}**  \n<span class='cart-meta'>{pid}{variant} · "
                        f"{format_price(row['sale_price'])} each</span>", unsafe_allow_html=True)
            key = f"cart_qty_{pid}"
            if key not in st.session_state or st.session_state[key] != row["quantity"]:
                st.session_state[key] = int(row["quantity"])
            c1, c2 = st.columns([3, 2])
            c1.number_input("Qty", min_value=1, max_value=max(1, int(row["stock_quantity"])), step=1, key=key,
                            label_visibility="collapsed", on_change=_on_qty_change, args=(cart, pid, key))
            c2.button("Remove", key=f"cart_rm_{pid}", on_click=_on_remove, args=(cart, pid), use_container_width=True)
        st.divider()
        st.markdown(f"Subtotal (MRP): ~~{format_price(totals['mrp'])}~~  \n"
                    f"You save: **{format_price(totals['savings'])}**  \n"
                    f"### Total: {format_price(totals['total'])}")
        c1, c2 = st.columns(2)
        if c1.button("Clear cart", key="cart_clear", use_container_width=True):
            cart.clear()
            st.rerun()
        if c2.button("Checkout", key="cart_checkout", type="primary", use_container_width=True):
            cart.clear()
            st.session_state["toast"] = "Demo order placed! (No real payment was made.)"
            st.rerun()
