"""Small, dependency-free helpers shared across modules."""

import hashlib
import re
from pathlib import Path

from config import CURRENCY


def split_list(value) -> list[str]:
    """Split a comma-separated cell ("Black, White") into a clean list."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def format_price(value) -> str:
    """Format a number in Indian rupee style: 123456.5 -> ₹1,23,456.50."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return f"{CURRENCY}-"
    rupees = int(round(amount * 100)) // 100
    paise = int(round(amount * 100)) % 100
    digits = str(abs(rupees))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    sign = "-" if rupees < 0 else ""
    return f"{sign}{CURRENCY}{digits}" + (f".{paise:02d}" if paise else "")


def file_fingerprint(path: Path) -> str:
    """MD5 of the file contents; used to detect dataset changes."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def singularize(word: str) -> str:
    """Very small English singularizer, good enough for product nouns."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def phrase_variants(phrase: str) -> set[str]:
    """Spelling variants for matching a phrase: plural/singular, hyphen/space."""
    base = phrase.lower().strip()
    words = base.split()
    variants = {base}
    if words:
        variants.add(" ".join(words[:-1] + [singularize(words[-1])]))
    for v in list(variants):
        if "-" in v:
            variants.add(v.replace("-", " "))
            variants.add(v.replace("-", ""))
    return {v for v in variants if v}


def phrase_regex(variants) -> re.Pattern:
    """Whole-word regex matching any of the variants (longest first)."""
    ordered = sorted(variants, key=len, reverse=True)
    body = "|".join(re.escape(v) for v in ordered)
    return re.compile(rf"(?<![\w-])(?:{body})(?![\w-])", re.IGNORECASE)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
