"""Exact metadata filtering and sorting with Pandas (no LLM involved).

`SearchFilters` is the single search state: the retriever applies it, the reply
describes it and the conversation context stores it, so all three always agree.
"""

from dataclasses import asdict, dataclass, field, fields

import pandas as pd

from config import EXACT_PRICE_SLACK
from utils.helpers import format_price

# Fields that together express "what the shopper wants to pay". A new price replaces all of them.
PRICE_FIELDS = ("min_price", "max_price", "min_inclusive", "max_inclusive", "target_price", "target_tolerance", "budget")

SORT_OPTIONS = {
    "price_asc": (["sale_price", "rating"], [True, False], "lowest price first"),
    "price_desc": (["sale_price", "rating"], [False, False], "highest price first"),
    "rating_desc": (["rating", "review_count"], [False, False], "highest rated first"),
    "discount_desc": (["discount_percent", "rating"], [False, False], "biggest discount first"),
    "popularity_desc": (["review_count", "rating"], [False, False], "most reviewed first"),
}


@dataclass
class SearchFilters:
    category: str | None = None
    subcategory: str | None = None
    brands: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    min_inclusive: bool = True            # "at least ₹200" (>=) vs "above/over ₹200" (>)
    max_inclusive: bool = True            # "up to ₹200" (<=) vs "under/below ₹200" (<)
    target_price: float | None = None     # "exactly ₹300" (tolerance 0) or "around ₹300" (tolerance > 0)
    target_tolerance: float = 0.0         # fraction of target_price allowed either side
    min_rating: float | None = None
    max_rating: float | None = None
    min_discount: float | None = None
    colors: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    in_stock_only: bool = False
    budget: bool = False            # "affordable"/"cheap": lower-priced part of the matches
    sort_by: str | None = None
    # A catalog product name the shopper named ("KitchenCraft Urban Storage", or "Active Fiction").
    # Products with this name are ranked first and reported as exact product matches.
    product_name: str | None = None
    # The shopper named one specific product by its full catalog name: show only products with exactly that
    # name (names are not unique, so this can be more than one card). Off when they asked for alternatives.
    exact_product: bool = False

    @property
    def price_type(self) -> str:
        """exact | around | range | minimum | maximum | none — how the price was requested."""
        if self.target_price is not None:
            return "around" if self.target_tolerance else "exact"
        if self.min_price is not None and self.max_price is not None:
            return "range"
        if self.max_price is not None:
            return "maximum"
        return "minimum" if self.min_price is not None else "none"

    @property
    def exact_price(self) -> float | None:
        return self.target_price if self.price_type == "exact" else None

    def name_mask(self, df: pd.DataFrame) -> pd.Series:
        """True for products whose name contains the requested product name."""
        if not self.product_name:
            return pd.Series(False, index=df.index)
        return df["name"].str.lower().str.contains(self.product_name.lower(), regex=False)

    def constraint_fields(self) -> dict:
        """Fields other than category/subcategory that differ from their defaults."""
        skip = {"category", "subcategory", "min_inclusive", "max_inclusive", "target_tolerance"}
        defaults = asdict(SearchFilters())
        return {k: v for k, v in asdict(self).items() if k not in skip and v != defaults[k]}

    def has_constraints(self) -> bool:
        return bool(self.constraint_fields())

    def is_empty(self) -> bool:
        return not self.category and not self.subcategory and not self.has_constraints()

    def has_price(self) -> bool:
        return self.min_price is not None or self.max_price is not None or self.target_price is not None

    def clear_price(self) -> "SearchFilters":
        return self.copy(min_price=None, max_price=None, target_price=None, target_tolerance=0.0, budget=False)

    def price_window(self) -> tuple[float | None, float | None]:
        """The (low, high) sale-price bounds actually applied, combining min/max and target price."""
        lo, hi = self.min_price, self.max_price
        if self.target_price is not None:
            slack = max(self.target_price * self.target_tolerance, EXACT_PRICE_SLACK)
            t_lo, t_hi = self.target_price - slack, self.target_price + slack
            lo = t_lo if lo is None else max(lo, t_lo)
            hi = t_hi if hi is None else min(hi, t_hi)
        return lo, hi

    def merged_with(self, newer: "SearchFilters") -> "SearchFilters":
        """Apply a follow-up's filters on top of these.

        Only the fields the follow-up mentions change. Price is one unit: "under ₹500" after
        "between ₹300 and ₹800" replaces the whole range instead of leaving a stale minimum.
        """
        merged = SearchFilters(**asdict(self))
        price_given = newer.has_price()
        for f in fields(SearchFilters):
            if f.name in PRICE_FIELDS:
                if price_given:
                    setattr(merged, f.name, getattr(newer, f.name))
                elif f.name == "budget" and newer.budget:   # "cheap ones" without a number
                    merged.budget = True
                continue   # no new price: keep the old one exactly, including whether its bounds are strict
            value = getattr(newer, f.name)
            if value not in (None, [], False):
                setattr(merged, f.name, value)
        if newer.category and newer.category != self.category and not newer.subcategory:
            merged.subcategory = None
        # A named product only survives follow-ups that stay within its brand and subcategory.
        scope_changed = (newer.subcategory and newer.subcategory != self.subcategory) or \
            (newer.brands and newer.brands != self.brands) or (newer.category and newer.category != self.category)
        if scope_changed and not newer.product_name:
            merged.product_name = None
        return merged

    def copy(self, **changes) -> "SearchFilters":
        data = asdict(self)
        data.update(changes)
        return SearchFilters(**data)

    def describe_price(self) -> str | None:
        if self.target_price is not None:
            if not self.target_tolerance:
                return f"exactly {format_price(self.target_price)}"
            lo, hi = self.price_window()
            return f"around {format_price(self.target_price)} ({format_price(lo)}–{format_price(hi)})"
        low = high = None
        if self.min_price is not None:
            low = f"{'at least' if self.min_inclusive else 'above'} {format_price(self.min_price)}"
        if self.max_price is not None:
            high = f"{'up to' if self.max_inclusive else 'under'} {format_price(self.max_price)}"
        if low and high and self.min_inclusive and self.max_inclusive:
            return f"{format_price(self.min_price)}–{format_price(self.max_price)}"
        return ", ".join(p for p in (low, high) if p) or None

    def describe(self, include_sort: bool = True) -> str:
        parts = []
        if self.subcategory:
            parts.append(self.subcategory)
        elif self.category:
            parts.append(self.category)
        if self.brands:
            parts.append("brand " + " / ".join(self.brands))
        price = self.describe_price()
        if price:
            parts.append(price)
        if self.budget:
            parts.append("budget-friendly")
        if self.colors:
            parts.append("colour " + " / ".join(self.colors))
        if self.sizes:
            parts.append("size " + " / ".join(self.sizes))
        if self.materials:
            parts.append(" / ".join(self.materials))
        if self.min_rating is not None:
            parts.append(f"rating ≥ {self.min_rating:g}")
        if self.max_rating is not None:
            parts.append(f"rating ≤ {self.max_rating:g}")
        if self.min_discount is not None:
            parts.append(f"discount ≥ {self.min_discount:g}%")
        if self.in_stock_only:
            parts.append("in stock")
        if include_sort and self.sort_by in SORT_OPTIONS:
            parts.append(SORT_OPTIONS[self.sort_by][2])
        return " · ".join(parts) if parts else "all products"


# Hard-constraint groups, used to explain exactly which requested filters alternatives do not meet.
CONSTRAINT_GROUPS = {
    "category": ("category",),
    "subcategory": ("subcategory",),
    "brand": ("brands",),
    "price": ("min_price", "max_price", "min_inclusive", "max_inclusive", "target_price", "target_tolerance"),
    "rating": ("min_rating", "max_rating"),
    "discount": ("min_discount",),
    "colour": ("colors",),
    "size": ("sizes",),
    "material": ("materials",),
    "stock": ("in_stock_only",),
}


def without_group(f: SearchFilters, group: str) -> SearchFilters:
    """The same filters with one constraint group (e.g. "colour") removed."""
    defaults = SearchFilters()
    return f.copy(**{n: getattr(defaults, n) for n in CONSTRAINT_GROUPS[group]})


def active_groups(f: SearchFilters) -> list[str]:
    defaults = SearchFilters()
    return [g for g, names in CONSTRAINT_GROUPS.items()
            if g not in ("category", "subcategory") and any(getattr(f, n) != getattr(defaults, n) for n in names)]


def apply_filters(df: pd.DataFrame, f: SearchFilters) -> pd.DataFrame:
    """Return rows matching every constraint exactly. Prices use sale_price."""
    mask = pd.Series(True, index=df.index)
    if f.category:
        mask &= df["category"].str.lower() == f.category.lower()
    if f.subcategory:
        mask &= df["subcategory"].str.lower() == f.subcategory.lower()
    if f.brands:
        mask &= df["brand"].str.lower().isin([b.lower() for b in f.brands])
    if f.exact_product and f.product_name:
        mask &= df["name"].str.lower() == f.product_name.lower()
    # Strict price bounds: "under ₹200" is < 200, "up to ₹200" is <= 200, "above" is >, "at least" is >=.
    if f.min_price is not None:
        mask &= (df["sale_price"] >= f.min_price) if f.min_inclusive else (df["sale_price"] > f.min_price)
    if f.max_price is not None:
        mask &= (df["sale_price"] <= f.max_price) if f.max_inclusive else (df["sale_price"] < f.max_price)
    if f.target_price is not None:
        lo, hi = f.copy(min_price=None, max_price=None).price_window()
        mask &= df["sale_price"].between(lo, hi)
    if f.min_rating is not None:
        mask &= df["rating"] >= f.min_rating
    if f.max_rating is not None:
        mask &= df["rating"] <= f.max_rating
    if f.min_discount is not None:
        mask &= df["discount_percent"] >= f.min_discount
    if f.colors:
        wanted = {c.lower() for c in f.colors}
        mask &= df["_colors"].map(lambda s: bool(s & wanted))
    if f.sizes:
        wanted = {s.lower() for s in f.sizes}
        mask &= df["_sizes"].map(lambda s: bool(s & wanted))
    if f.materials:
        mask &= df["material"].str.lower().isin([m.lower() for m in f.materials])
    if f.in_stock_only:
        mask &= df["in_stock"] & (df["stock_quantity"] > 0)
    result = df[mask]
    if f.budget and len(result) > 3:
        cutoff = result["sale_price"].quantile(0.4)
        result = result[result["sale_price"] <= cutoff]
    return result


def sort_products(df: pd.DataFrame, sort_by: str | None) -> pd.DataFrame:
    if sort_by not in SORT_OPTIONS or df.empty:
        return df
    cols, ascending, _ = SORT_OPTIONS[sort_by]
    return df.sort_values(cols, ascending=ascending, kind="mergesort")
