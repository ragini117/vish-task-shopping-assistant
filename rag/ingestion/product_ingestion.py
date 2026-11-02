"""Load, validate and clean the product CSV, and turn rows into documents."""

import logging
import re
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from config import REQUIRED_COLUMNS
from utils.helpers import singularize, split_list

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ["name", "category", "subcategory", "brand", "description", "sizes", "colors", "material", "tags"]
NUMERIC_COLUMNS = ["price", "discount_percent", "sale_price", "rating", "review_count", "stock_quantity"]


class DatasetError(Exception):
    """Raised when the product dataset is missing or unusable."""


def _parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def load_products(path: Path) -> pd.DataFrame:
    """Read the CSV, validate its schema and return a cleaned DataFrame.

    Cleaning rules (all values still come from the dataset):
    - rows without product_id or name are dropped; duplicate IDs keep the first row
    - missing text becomes "" (sizes default to "Standard")
    - numbers are coerced; a missing sale_price is derived from price and discount
    - in_stock is parsed from text, falling back to stock_quantity > 0
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found at '{path}'. Place {Path(path).name} in the data/ folder.")
    try:
        df = pd.read_csv(path, dtype={"product_id": str})
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Dataset '{path.name}' is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Dataset '{path.name}' is not a valid CSV file: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {', '.join(missing)}")

    raw_count = len(df)
    df = df.dropna(subset=["product_id", "name"])
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df = df[df["product_id"] != ""]
    df = df.drop_duplicates(subset="product_id", keep="first")

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df.loc[df["sizes"] == "", "sizes"] = "Standard"
    df.loc[df["category"] == "", "category"] = "Other"
    df.loc[df["subcategory"] == "", "subcategory"] = "General"

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["discount_percent"] = df["discount_percent"].fillna(0).clip(0, 100)
    df = df.dropna(subset=["price"])
    missing_sale = df["sale_price"].isna()
    df.loc[missing_sale, "sale_price"] = (df["price"] * (1 - df["discount_percent"] / 100)).round(2)
    df["rating"] = df["rating"].fillna(0).clip(0, 5)
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["stock_quantity"] = df["stock_quantity"].fillna(0).clip(lower=0).astype(int)

    parsed_stock = df["in_stock"].map(_parse_bool)
    df["in_stock"] = parsed_stock.where(parsed_stock.notna(), df["stock_quantity"] > 0).astype(bool)

    if df.empty:
        raise DatasetError("Dataset contains no valid product rows.")
    dropped = raw_count - len(df)
    if dropped:
        logger.warning("Dropped %d invalid or duplicate rows from the dataset", dropped)

    # Pre-parsed lowercase sets for fast exact filtering.
    df["_sizes"] = df["sizes"].map(lambda v: {s.lower() for s in split_list(v)})
    df["_colors"] = df["colors"].map(lambda v: {s.lower() for s in split_list(v)})
    df["_tags"] = df["tags"].map(lambda v: {s.lower() for s in split_list(v)})
    df["_search_text"] = (
        df["name"] + " " + df["brand"] + " " + df["category"] + " " + df["subcategory"] + " " + df["tags"]
    ).str.lower()
    # Whole-word tokens (plus singular forms) so "pen" never matches "OpenShelf" or "open-toe".
    df["_tokens"] = df["_search_text"].map(tokenize)
    return df.set_index("product_id", drop=False)


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return set(words) | {singularize(w) for w in words}


def product_to_text(row) -> str:
    """Text that gets embedded for a product (name, category, brand, description, features)."""
    return (
        f"{row['name']}. Brand: {row['brand']}. Category: {row['category']} > {row['subcategory']}. "
        f"Material: {row['material']}. Colors: {row['colors']}. Sizes: {row['sizes']}. "
        f"Tags: {row['tags']}. {row['description']}"
    )


def build_documents(df: pd.DataFrame) -> list[Document]:
    return [
        Document(
            page_content=product_to_text(row),
            metadata={"product_id": row["product_id"], "category": row["category"], "subcategory": row["subcategory"]},
        )
        for _, row in df.iterrows()
    ]
