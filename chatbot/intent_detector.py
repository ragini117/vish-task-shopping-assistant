"""Rule-based intent and filter extraction driven by the dataset's own vocabulary.

Categories, subcategories, brands, colours, sizes and materials are all read
from the loaded DataFrame, so new values in the CSV are understood without
code changes. Config synonyms only map everyday words onto those values.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd

from config import AROUND_PRICE_TOLERANCE, CATEGORY_SYNONYMS, EXTERNAL_BRANDS, GENERIC_TAGS, SUBCATEGORY_SYNONYMS
from rag.retrieval.product_filter import SearchFilters
from utils.helpers import phrase_regex, phrase_variants, split_list

CURRENCY_PREFIX = r"(?:₹\s*|\brs\.?\s*|\binr\s*)"
CURRENCY_SUFFIX = r"(?:\s*(?:rupees?|rs\b\.?|inr\b|/-))"
UNIT = r"(k\b|thousand\b|lakhs?\b)"       # "2k", "1.5 lakh"; "2kg" is not "2k"
AMOUNT = rf"{CURRENCY_PREFIX}?(\d[\d,]*(?:\.\d+)?)\s*{UNIT}?{CURRENCY_SUFFIX}?"
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "last": -1}
RELATIVE_CUES = [
    ("cheaper", r"\b(cheaper|less expensive|lower price[d]?|more affordable)\b"),
    ("higher_rated", r"\b(higher|better)[- ]rat(?:ed|ing|ings)\b"),
    ("other_colors", r"\b(other|different|another|more|which|what)\s+colou?rs?\b|\bcolou?r options\b"),
    ("more", r"\b(show more|more options|see more|load more|next page|more results|show me more|any more|more products)\b|^\s*(more|next)\s*[.!?]?\s*$"),
]
SIZE_WORDS = {"extra small": "XS", "extra large": "XL", "double xl": "XXL", "small": "S", "medium": "M", "large": "L"}
COLOR_ALIASES = {"gray": "grey", "navy blue": "navy", "cream": "beige", "violet": "purple"}
MATERIAL_ALIASES = {"aluminium": "aluminum", "wooden": "wood", "steel": "stainless steel", "synthetic": "synthetic blend"}

STOP_TOKENS = {
    "i", "me", "my", "a", "an", "the", "to", "for", "of", "in", "on", "with", "and", "or", "is", "are", "some", "any",
    "want", "wanna", "need", "buy", "show", "find", "get", "give", "suggest", "recommend", "looking", "look", "see",
    "please", "pls", "can", "you", "could", "would", "like", "also", "only", "just", "ones", "one", "that", "those",
    "these", "them", "it", "what", "about", "how", "which", "do", "have", "has", "there", "under", "below", "above",
    "over", "than", "less", "more", "between", "within", "up", "upto", "products", "product", "items", "item", "stuff",
    "options", "option", "available", "now", "all", "category", "type", "types", "kind", "kinds", "good", "nice",
    "best", "top", "new", "something", "things", "thing", "list", "price", "priced", "budget", "rs", "inr", "rupees",
    "size", "sizes", "colour", "color", "colours", "colors", "rating", "rated", "stars", "star", "discount", "off",
    "sale", "cheap", "cheaper", "cheapest", "affordable", "expensive", "costly", "in-stock", "stock", "me.", "hey",
    "hi", "hello", "let", "lets", "let's", "go", "too", "very", "really", "much", "many", "most", "least", "low",
    "high", "higher", "lower", "range", "from", "at", "by", "be", "we", "our", "your", "im", "i'm", "am", "will",
    "should", "shall", "may", "might", "must", "then", "so", "but", "if", "not", "no", "yes", "ok", "okay", "thanks",
    "gear", "equipment", "collection", "section", "department", "range", "variety", "selection", "shopping", "shop",
    "filter", "filters", "remove", "clear", "reset", "start", "search", "again", "instead", "exactly", "around",
    "worth", "cost", "costing", "costs", "priced", "rupee", "alternative", "alternatives", "similar", "sort", "sorted", "order", "ordered", "actually", "rather", "make", "switch", "else", "different", "another", "other", "same", "similar", "change",
}


@dataclass
class ParsedQuery:
    text: str
    intent: str = "search"   # greeting | thanks | help | browse_all | cart_view | cart_add | cart_remove | compare | detail | search
    filters: SearchFilters = field(default_factory=SearchFilters)
    category_options: list[tuple[str, str | None]] = field(default_factory=list)  # ambiguous scope
    category_term: str | None = None        # the user's own word for the category, e.g. "shoes"
    product_ids: list[str] = field(default_factory=list)
    ordinals: list[int] = field(default_factory=list)
    count_hint: int | None = None           # "these two" -> 2
    unknown_brands: list[str] = field(default_factory=list)
    unsupported_sizes: list[str] = field(default_factory=list)
    relative: str | None = None             # cheaper | higher_rated | other_colors | more
    refinement_cue: bool = False
    refers_to_results: bool = False
    catalog_wide: bool = False              # "products with ...", "anything under ..."
    reset_context: bool = False             # "start over", "clear filters"
    cleared_fields: list[str] = field(default_factory=list)  # "any colour" -> ["colors"]
    leftover_terms: list[str] = field(default_factory=list)
    price_question: float | None = None     # "worth ₹300" / "shoes 2000": amount whose meaning must be confirmed
    price_unreadable: bool = False          # "under ₹" / "under ₹abc": a price was meant but could not be read
    context_used: bool = False              # set by the assistant when the previous search was refined
    wants_alternatives: bool = False        # "alternatives to X", "similar to X": don't restrict to X

    @property
    def has_scope(self) -> bool:
        return bool(self.filters.category or self.filters.subcategory)

    @property
    def product_name(self) -> str | None:
        return self.filters.product_name


class IntentDetector:
    def __init__(self, df: pd.DataFrame):
        self.categories = sorted(df["category"].unique())
        self.subcats_by_category = {c: sorted(g["subcategory"].unique()) for c, g in df.groupby("category")}
        self.brands = sorted(df["brand"].unique())
        self.colors = sorted({c for v in df["colors"] for c in split_list(v)})
        self.sizes = sorted({s for v in df["sizes"] for s in split_list(v)})
        self.materials = sorted({m for m in df["material"].unique() if m})
        self.brand_categories = {b: set(g["category"]) for b, g in df.groupby("brand")}
        self._build_scope_patterns(df)
        self._build_attribute_patterns()
        self._build_name_patterns(df)

    # ----------------------------------------------------------- vocabulary
    def _build_scope_patterns(self, df: pd.DataFrame):
        """Ordered (regex, [(category, subcategory|None)]) list; earlier = higher priority."""
        term_targets: dict[str, set] = defaultdict(set)
        tiers: dict[str, int] = {}

        def add(term: str, target: tuple, tier: int):
            for v in phrase_variants(term):
                term_targets[v].add(target)
                tiers[v] = min(tiers.get(v, 9), tier)

        pairs = df[["category", "subcategory"]].drop_duplicates().itertuples(index=False)
        pairs = [(p.category, p.subcategory) for p in pairs]
        valid_subs = defaultdict(list)
        for cat, sub in pairs:
            valid_subs[sub].append(cat)
            add(sub, (cat, sub), 0)
        for sub, words in SUBCATEGORY_SYNONYMS.items():
            for cat in valid_subs.get(sub, []):
                for w in words:
                    add(w, (cat, sub), 1)
        for cat in self.categories:
            add(cat, (cat, None), 2)
            for part in re.split(r"\s*(?:&|,| and )\s*", cat):
                if part:
                    add(part, (cat, None), 2)
            for w in CATEGORY_SYNONYMS.get(cat, []):
                add(w, (cat, None), 2)
            # head nouns shared by several subcategories, e.g. "shoes" for Footwear
            heads = Counter(s.split()[-1].lower() for s in self.subcats_by_category[cat] if " " in s)
            for head, n in heads.items():
                if n >= 2:
                    add(head, (cat, None), 2)

        # Tags that occur in exactly one subcategory are product-type words.
        excluded = GENERIC_TAGS | {b.lower() for b in self.brands} | {m.lower() for m in self.materials} \
            | {c.lower() for c in self.colors}
        tag_targets: dict[str, Counter] = defaultdict(Counter)
        for cat, sub, tags in df[["category", "subcategory", "tags"]].itertuples(index=False):
            for t in split_list(tags):
                tag_targets[t.lower()][(cat, sub)] += 1
        for tag, counts in tag_targets.items():
            if tag in excluded or len(counts) != 1 or sum(counts.values()) < 10:
                continue
            target = next(iter(counts))
            if all(v not in term_targets for v in phrase_variants(tag)):
                add(tag, target, 3)

        self.scope_patterns = []
        for term in sorted(term_targets, key=lambda t: (tiers[t], -len(t))):
            self.scope_patterns.append((phrase_regex([term]), term, sorted(term_targets[term], key=str)))
        # Longest phrases first within the whole list so "sports shoes" beats "sports".
        self.scope_patterns.sort(key=lambda p: (-len(p[1]), tiers[p[1]]))

    def _build_attribute_patterns(self):
        def mapping(values, aliases=None):
            table = {v.lower(): v for v in values}
            for alias, target in (aliases or {}).items():
                if target in table:
                    table[alias] = table[target]
            return [(phrase_regex([k]), v) for k, v in sorted(table.items(), key=lambda kv: -len(kv[0]))]

        self.color_patterns = mapping(self.colors, COLOR_ALIASES)
        self.material_patterns = mapping(self.materials, MATERIAL_ALIASES)
        brand_table = {}
        for b in self.brands:
            brand_table[b.lower()] = b
            brand_table[b.lower().replace(" ", "")] = b
        self.brand_patterns = [(phrase_regex([k]), v) for k, v in sorted(brand_table.items(), key=lambda kv: -len(kv[0]))]
        own = {b.lower() for b in self.brands}
        self.external_brand_patterns = [(phrase_regex([b]), b) for b in EXTERNAL_BRANDS if b not in own]
        self.size_lookup = {s.lower(): s for s in self.sizes}

    def _build_name_patterns(self, df: pd.DataFrame):
        """Product names from the CSV ("KitchenCraft Urban Storage") and their model part ("Urban Storage").

        Names follow "<brand> <model word> <subcategory>"; the model word is what remains after removing the
        brand and subcategory, and only words shared by several products count, so free text never matches.
        """
        models: Counter = Counter()
        full_names: dict[str, tuple[str, str]] = {}
        for name, brand, sub in df[["name", "brand", "subcategory"]].itertuples(index=False):
            model = name.replace(brand, "", 1).replace(sub, "", 1).strip()
            if model and name == f"{brand} {model} {sub}":
                models[model] += 1
                full_names[name] = (model, f"{model} {sub}")
        self.model_words = {m for m, n in models.items() if n >= 5}
        self.full_names = {name for name, (model, _) in full_names.items() if model in self.model_words}
        scopes = df.drop_duplicates("name").set_index("name")[["category", "subcategory"]]
        self.name_scope = {n: tuple(scopes.loc[n]) for n in self.full_names}
        forms: dict[str, tuple[str, str]] = {}   # spelling -> (canonical name, model word)
        for name, (model, short) in full_names.items():
            if model in self.model_words:
                for v in phrase_variants(name):
                    forms[v] = (name, model)
                for v in phrase_variants(short):
                    forms.setdefault(v, (short, model))
        self.name_forms = forms
        self.name_pattern = phrase_regex(forms) if forms else None

    def _detect_product_name(self, pq: ParsedQuery, text: str) -> str:
        """Record a named catalog product and blank its model word ("Urban"); brand and subcategory stay
        in the text so they are still parsed as filters. Runs before sort words, so "Premium Storage" is a
        product name and not a "most expensive first" request."""
        m = self.name_pattern.search(text) if self.name_pattern else None
        if not m:
            return text
        canonical, model = self.name_forms[m.group(0).lower()]
        pq.filters.product_name = canonical
        # A full catalog name picks that specific product; a short form ("Active Fiction") names several.
        pq.filters.exact_product = canonical in self.full_names and not pq.wants_alternatives
        span = re.sub(rf"\b{re.escape(model.lower())}\b", " " * len(model), m.group(0), count=1)
        return text[: m.start()] + span + text[m.end():]

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _amount(number: str, unit: str | None) -> float:
        value = float(number.replace(",", ""))
        if unit in ("k", "thousand"):
            value *= 1000
        elif unit and unit.startswith("lakh"):
            value *= 100000
        return value

    @staticmethod
    def _blank(text: str, match: re.Match) -> str:
        return text[: match.start()] + " " * (match.end() - match.start()) + text[match.end():]

    # ----------------------------------------------------------------- parse
    def parse(self, query: str) -> ParsedQuery:
        raw = query.strip()
        text = " " + re.sub(r"\s+", " ", raw.lower()) + " "
        text = text.replace("–", "-").replace("’", "'")
        pq = ParsedQuery(text=raw)
        f = pq.filters

        pq.product_ids = [m.upper() for m in re.findall(r"\bprod\d{3,}\b", text)]
        text = re.sub(r"\bprod\d{3,}\b", " ", text)
        pq.ordinals = [ORDINALS[m] for m in re.findall(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|last)\b", text)]
        count = re.search(r"\b(?:these|those|the|top|first)\s+(two|three|four|2|3|4)\b", text)
        if count:
            pq.count_hint = {"two": 2, "three": 3, "four": 4}.get(count.group(1)) or int(count.group(1))

        pq.wants_alternatives = bool(re.search(
            r"\b(alternatives?|similar|other options|something like|products like|instead of|compared? to)\b", text))
        self._detect_intent(pq, text)
        text = self._detect_product_name(pq, text)

        # Ratings first so "above 4" is not read as a price.
        rating_patterns = [
            r"\b(?:rating|ratings|rated|stars?)\s*(?:of|is|:)?\s*(above|over|greater than|more than|at least|minimum|min|>=|>|below|under|less than|<=|<)?\s*(\d(?:\.\d)?)\s*(?:\+|stars?|or (?:more|above|higher|better)|and (?:above|up))?",
            r"\b(above|over|greater than|more than|at least|minimum|below|under|less than)?\s*(\d(?:\.\d)?)\s*\+?\s*(?:stars?|star rating|rating|rated)\b(?:\s*(?:and above|or above|or more|or higher|and up|\+))?",
        ]
        for pattern in rating_patterns:
            for m in list(re.finditer(pattern, text)):
                value = float(m.group(2))
                if value > 5:
                    continue
                if m.group(1) in ("below", "under", "less than", "<=", "<"):
                    f.max_rating = value
                else:
                    f.min_rating = value
                text = self._blank(text, m)
        found, text = self._take(r"\b(highly|top|well|best|highest|high)[- ]rated\b|\bbest reviewed\b"
                                 r"|\b(good|great|high|top) ratings?\b", text)
        if found:
            f.min_rating = f.min_rating or 4.0
            f.sort_by = "rating_desc"

        # Discounts
        m = re.search(r"(?:discount|off)\s*(?:of\s*)?(?:at least|above|over|more than|minimum)?\s*(\d{1,2})\s*%|(\d{1,2})\s*%\s*(?:off|discount|or more|and above|\+)?", text)
        if m:
            f.min_discount = float(m.group(1) or m.group(2))
            text = self._blank(text, m)
        found, text = self._take(r"\b(on sale|discounted|deals?|offers?|on offer|biggest discount|best discount|"
                                 r"highest discount|max(?:imum)? discount)\b", text)
        if found:
            f.sort_by = f.sort_by or "discount_desc"
            if f.min_discount is None:
                f.min_discount = 1

        text = self._parse_price(pq, text)

        # Sizes: one or several ("size M or L", "sizes M, L", "size M/L", "M/L hoodies") — every size is kept.
        size = r"(?:extra large|extra small|xxl|xl|xs|s|m|l|small|medium|large|standard|\d+(?:\.\d)?)"
        size_list = rf"{size}(?:\s*(?:,|/|\bor\b|\band\b|&)\s*{size})*"
        for pattern in (rf"\b(?:sizes?|sz)\s*(?:of|is|are|:|-)?\s*({size_list})\b",
                        r"\b((?:xxl|xl|xs|s|m|l)(?:\s*/\s*(?:xxl|xl|xs|s|m|l))+)\b"):
            for m in list(re.finditer(pattern, text)):
                for token in re.split(r"\s*(?:,|/|\bor\b|\band\b|&)\s*", m.group(1)):
                    if token:
                        self._add_size(pq, token)
                text = self._blank(text, m)
        for m in list(re.finditer(r"\b(?:uk|us|eu|india|ind)\s*(\d{1,2}(?:\.\d)?)\b", text)):
            pq.unsupported_sizes.append(m.group(0).strip().upper())
            text = self._blank(text, m)
        for m in list(re.finditer(r"\b(xxl|xl|xs)\b|\b(extra large|extra small|small|medium|large)\s+size\b", text)):
            self._add_size(pq, m.group(1) or m.group(2))
            text = self._blank(text, m)

        # Colours, materials, brands
        for pattern, value in self.color_patterns:
            if pattern.search(text):
                if value not in f.colors:
                    f.colors.append(value)
                text = pattern.sub(" ", text)
        for pattern, value in self.material_patterns:
            if pattern.search(text):
                f.materials.append(value)
                text = pattern.sub(" ", text)
        for pattern, value in self.brand_patterns:
            if pattern.search(text):
                if value not in f.brands:
                    f.brands.append(value)
                text = pattern.sub(" ", text)
        for pattern, value in self.external_brand_patterns:
            if pattern.search(text):
                pq.unknown_brands.append(value.title())
                text = pattern.sub(" ", text)

        # Explicit sort instructions ("sort by price high to low", "order by rating") set the order and are
        # blanked, so "sort" is never mistaken for a product word.
        m = re.search(r"\b(?:sort|sorted|order|ordered|arrange|arranged|rank|ranked)\s+(?:(?:them|it|these|results?)\s+)?"
                      r"by\s+(price|cost|ratings?|reviews?|popularity|discounts?)\b"
                      r"(?:\s*[:,-]?\s*(high to low|highest first|descending|desc|low to high|lowest first|ascending|asc)\b)?",
                      text)
        if m:
            key, direction = m.group(1), m.group(2) or ""
            if key in ("price", "cost"):
                f.sort_by = "price_desc" if direction in ("high to low", "highest first", "descending", "desc") \
                    else "price_asc"
            elif key.startswith("rating"):
                f.sort_by = "rating_desc"
            elif key.startswith("review") or key == "popularity":
                f.sort_by = "popularity_desc"
            else:
                f.sort_by = "discount_desc"
            text = self._blank(text, m)

        # Sorting / budget / stock
        cheapest, text = self._take(r"\b(cheapest|lowest price[d]?|least expensive|low to high|lowest cost)\b", text)
        cheap, text = self._take(r"\b(cheap|affordable|inexpensive|economical|budget|value for money|low cost|low price[d]?)\b", text)
        if cheapest:
            f.sort_by = "price_asc"
        elif cheap:
            f.sort_by = "price_asc"
            f.budget = not f.has_price()   # an explicit price beats the vague "cheap"
        found, text = self._take(r"\b(most expensive|costliest|priciest|high to low|premium|luxury|high[- ]end|top[- ]end)\b", text)
        if found:
            f.sort_by, f.budget = "price_desc", False
        found, text = self._take(r"\b(most popular|popular|bestsell(?:er|ers|ing)|trending|most reviewed)\b", text)
        if found:
            f.sort_by = f.sort_by or "popularity_desc"
        if not f.sort_by and re.search(r"\bbest\b", text):
            f.sort_by = "rating_desc"
        found, text = self._take(r"\b(in stock|in-stock|available now|currently available|ready to ship)\b", text)
        if found:
            f.in_stock_only = True

        # Follow-up cues
        for name, pattern in RELATIVE_CUES:
            if re.search(pattern, text):
                pq.relative = pq.relative or name
                text = re.sub(pattern, " ", text)  # control phrases are not product words
        pq.refinement_cue = bool(
            pq.relative
            or re.match(r"^\s*(only|just|and|also|but|what about|how about|any|now|then|in|with|under|below|above|over|show me the|show the|sort|filter|actually|instead|rather|change|make it|switch)\b", text)
            or re.search(r"\b(instead|actually|change it to|make it)\b", text)
            or re.search(r"\b(ones|those|these|them|the same)\b", text)
        )
        pq.refers_to_results = bool(re.search(
            r"\b(these|those|them|they|it|this one|that one|which one|which of|the first|the second|the third|above products?)\b", text))
        pq.catalog_wide = bool(re.search(r"\b(all products|any products?|products|items|everything|anything|whole store|entire catalog)\b", text)) \
            and not re.search(r"\b(these|those|them|ones)\b", text)
        pq.reset_context = bool(re.search(r"\b(start over|reset|clear (all )?filters|remove (all )?filters|new search)\b", text))
        clear_rules = {
            "colors": r"\b(any|all|other) colou?rs?\b|\bremove (the )?colou?r\b",
            "sizes": r"\b(any|all) sizes?\b|\bremove (the )?size\b",
            "price": r"\b(any|no) (price|budget)\b|\b(remove|clear|drop|reset|forget|ignore) (the |my )?(price|budget)( filter| limit| range)?\b",
            "brands": r"\b(any|all|other) brands?\b|\bremove (the )?brand\b",
            "min_rating": r"\b(any|all) ratings?\b|\bremove (the )?rating\b",
            "materials": r"\b(any|all) materials?\b|\bremove (the )?material\b",
            "min_discount": r"\b(any|no) discount\b|\bremove (the )?discount\b",
            "in_stock_only": r"\binclude out[- ]of[- ]stock\b|\bremove (the )?stock\b",
        }
        pq.cleared_fields = [name for name, pattern in clear_rules.items() if re.search(pattern, text)]
        if "colors" in pq.cleared_fields and pq.relative == "other_colors" and not re.search(r"\b(any|all) colou?rs?\b", text):
            pq.cleared_fields.remove("colors")
        for pattern in list(clear_rules.values()) + [r"\b(start over|reset|clear (all )?filters|remove (all )?filters|new search)\b"]:
            text = re.sub(pattern, " ", text)

        # A number nobody claimed ("shoes 2000") is probably a price, but which kind? Ask instead of guessing.
        if pq.price_question is None and not f.has_price():
            m = re.search(rf"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*{UNIT}?(?![\w%])", text)
            if m and self._amount(m.group(1), m.group(2)) >= 50:
                value = self._amount(m.group(1), m.group(2))
                if f.product_name:   # "KitchenCraft Urban Storage 10884": that product's price
                    f.target_price, f.target_tolerance = value, 0.0
                else:
                    pq.price_question = value
                text = self._blank(text, m)

        self._detect_scope(pq, text)
        if f.product_name in self.name_scope:
            # A full catalog name pins its own category/subcategory ("PixelWave Max Accessories" is Electronics ›
            # Accessories), so words like "accessories" are not ambiguous here.
            f.category, f.subcategory = self.name_scope[f.product_name]
            pq.category_options = []
        return pq

    @staticmethod
    def _take(pattern: str, text: str) -> tuple[bool, str]:
        """Detect a control phrase and blank it so it is never mistaken for a product word."""
        new = re.sub(pattern, " ", text)
        return new != text, new

    def _detect_intent(self, pq: ParsedQuery, text: str):
        if re.search(r"\b(compare|comparison|versus|vs\.?|difference between|differences)\b", text):
            pq.intent = "compare"
        elif re.search(r"\b(remove|delete|take out)\b.*\b(cart|basket)\b|\b(remove|delete)\s+prod", text) or \
                (re.search(r"\b(remove|delete)\b", text) and pq.product_ids):
            pq.intent = "cart_remove"
        elif re.search(r"\badd\b.*\b(cart|basket|bag)\b|\badd\s+(it|this|that|the)\b", text) or (
                not pq.wants_alternatives and (
                    # "I want to buy this product", "buy it", "get that one", "I'll take it"
                    re.search(r"\b(buy|purchase|order|take|want|need|get)\s+(?:\w+\s+){0,2}?(this|that|it)\b"
                              r"(?:\s+(one|product|item))?(?!\s+(?:is|are|was|has|have|in stock)\b)", text)
                    or re.match(r"^\s*(i'?ll take|i want|i'?d like|give me)?\s*(this|that)\s+one\s*(please)?\s*[.!]*\s*$", text)
                    or (re.search(r"\b(buy|purchase|order)\b", text) and pq.product_ids))):
            pq.intent = "cart_add"
        elif re.search(r"\b(view|show|see|open|check|display)\s+(me\s+)?(my\s+|the\s+)?(cart|basket)\b|^\s*(my\s+)?(cart|basket)\s*[?.!]?\s*$|what'?s in my (cart|basket)|\bcart total\b", text):
            pq.intent = "cart_view"
        elif re.match(r"^\s*(please\s+)?((show|browse|see|list|view|display|give)\s+(me\s+)?)?(all|every|the entire|the whole|entire|whole|the complete|complete|the full|full)\s+(the\s+|your\s+)?(products?|items?|catalog(?:ue)?|store|inventory|range)\b", text) \
                or re.match(r"^\s*(all products|everything|browse|browse (all|everything|the catalog(?:ue)?|all products|the complete catalog(?:ue)?))\s*[.!?]*\s*$", text):
            pq.intent = "browse_all"
        elif re.match(r"^\s*(hi+|hello|hey+|namaste|hola|good (morning|afternoon|evening)|yo|greetings)\b[\s!.,?]*(there)?[\s!.,?]*$", text):
            pq.intent = "greeting"
        elif re.match(r"^\s*((ok(ay)?|great|cool|awesome|nice|perfect)[\s,!.]*)?(thanks|thank you|thank u|thanku|thx|ty|cheers|bye|goodbye)"
                      r"\b[\w\s!.,]{0,20}$", text) and not re.search(r"\b(but|show|find|want|need|under|cart)\b", text):
            pq.intent = "thanks"
        elif re.search(r"\b(help|what can you do|how does this work|how do i use|what do you sell|what categories)\b", text):
            pq.intent = "help"
        elif pq.product_ids or (pq.ordinals and re.search(r"\b(detail|details|more about|tell me|describe|info|information|specs)\b", text)):
            pq.intent = "detail"

    def _parse_price(self, pq: ParsedQuery, text: str) -> str:
        """Extract price constraints on sale_price and return text with them blanked.

        Semantics are documented next to AROUND_PRICE_TOLERANCE in config.py. A number becomes a price only
        when a price word or currency marker says so; "worth ₹X" is flagged for a clarifying question.
        """
        f = pq.filters

        def has_currency(m: re.Match) -> bool:
            return bool(re.search(rf"{CURRENCY_PREFIX}|{CURRENCY_SUFFIX}|\d\s*{UNIT}", m.group(0)))

        def plausible(value: float, m: re.Match) -> bool:
            return value > 0 and (has_currency(m) or value >= 50)

        for pattern in (rf"\b(?:between|from)\s+{AMOUNT}\s*(?:and|to|-)\s*{AMOUNT}",
                        rf"(?<![\w.]){AMOUNT}\s*(?:-|to)\s*{AMOUNT}"):
            m = re.search(pattern, text)
            if m:
                lo, hi = self._amount(m.group(1), m.group(2)), self._amount(m.group(3), m.group(4))
                if plausible(min(lo, hi), m):
                    f.min_price, f.max_price = min(lo, hi), max(lo, hi)
                    text = self._blank(text, m)
                    break

        # Strict vs inclusive bounds: "under/below/less than ₹X" -> < X, "up to/maximum/budget ₹X" -> <= X,
        # "above/over/more than ₹X" -> > X, "at least/minimum/starting from ₹X" -> >= X.
        inclusive_max = (r"within|upto|up to|at most|not more than|no more than|"
                         r"max(?:imum)?(?:\s+(?:price|budget))?(?:\s+(?:of|is))?|(?:my\s+)?budget(?:\s+(?:of|is))?")
        rules = [
            (r"\b(?:exactly|exact|precisely|priced at|costing|costs?|that costs|price of|price is)\s*(?:at|for|of)?\s*", "exact"),
            (r"(?:\b(?:around|about|approximately|approx\.?|roughly|nearly|close to)|~)\s*", "around"),
            (r"\bworth\s*(?:of\s*)?", "worth"),
            (r"(?:\b(?:under|below|less than|cheaper than|lower than)|<(?!=))\s*", "max_strict"),
            (rf"(?:\b(?:{inclusive_max})|<=)\s*", "max"),
            (r"\bfor\s+", "max_weak"),                           # "shoes for ₹2,000" / "for 2000": a budget (<=)
            (r"(?:\b(?:above|over|more than|greater than|costlier than|pricier than)|>(?!=))\s*", "min_strict"),
            (r"(?:\b(?:at least|min(?:imum)?(?:\s+price)?(?:\s+(?:of|is))?|starting(?:\s+(?:from|at))?)|>=)\s*", "min"),
            (r"\b(?:of|at)\s+", "exact_weak"),   # last, so "budget of ₹300" stays a cap; "Urban Storage of 10884 rs"
        ]
        for prefix, kind in rules:
            for m in list(re.finditer(prefix + AMOUNT + r"(?![\w%])", text)):
                value = self._amount(m.group(1), m.group(2))
                if not plausible(value, m):
                    if kind not in ("max_weak", "exact_weak"):   # "under 5": a price word with an implausible amount
                        pq.price_unreadable = True
                    continue
                if kind in ("exact", "exact_weak"):
                    if f.has_price():
                        continue   # "under ₹500 at ..." — keep the explicit price
                    f.target_price, f.target_tolerance = value, 0.0
                elif kind == "around":
                    f.target_price, f.target_tolerance = value, AROUND_PRICE_TOLERANCE
                elif kind == "worth":
                    pq.price_question = value
                elif kind in ("max", "max_weak", "max_strict"):
                    f.max_price, f.max_inclusive = value, kind != "max_strict"
                else:
                    f.min_price, f.min_inclusive = value, kind != "min_strict"
                text = self._blank(text, m)

        # A bare amount with a currency marker ("Urban Storage 10884 rs", "shoes ₹1500") is never silently a
        # budget cap: next to a named product it is that product's price, otherwise we ask what it means.
        if not f.has_price() and pq.price_question is None:
            for pattern in (rf"{CURRENCY_PREFIX}(\d[\d,]*(?:\.\d+)?)\s*{UNIT}?",
                            rf"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*{UNIT}?\s*(?:rupees?|rs\b\.?|inr\b|/-)"):
                m = re.search(pattern, text)
                if m:
                    value = self._amount(m.group(1), m.group(2))
                    if f.product_name:
                        f.target_price, f.target_tolerance = value, 0.0
                    else:
                        pq.price_question = value
                    text = self._blank(text, m)
                    break
        if f.min_price is not None and f.max_price is not None and f.min_price > f.max_price:
            f.min_price, f.max_price = f.max_price, f.min_price
        # A currency marker left over ("under ₹", "rs abc") means a price we could not read: never drop it silently.
        leftover_currency = r"₹|\brs\b\.?|\binr\b|\brupees?\b"
        if re.search(leftover_currency, text):
            pq.price_unreadable = True
            text = re.sub(leftover_currency, " ", text)
        return text

    def _add_size(self, pq: ParsedQuery, token: str):
        token = token.lower().strip()
        token = SIZE_WORDS.get(token, token)
        if token.lower() in self.size_lookup:
            size = self.size_lookup[token.lower()]
            if size not in pq.filters.sizes:
                pq.filters.sizes.append(size)
        else:
            pq.unsupported_sizes.append(token.upper())

    def _detect_scope(self, pq: ParsedQuery, text: str):
        """Find category/subcategory mentions; record ambiguity instead of guessing."""
        found = []  # (position, term, targets)
        for pattern, term, targets in self.scope_patterns:
            m = pattern.search(text)
            if m:
                found.append((m.start(), term, targets))
                text = self._blank(text, m)
        pq.leftover_terms = [t for t in re.findall(r"[a-z][a-z'-]+", text) if t not in STOP_TOKENS and len(t) > 2]
        if not found:
            return
        found.sort(key=lambda x: x[0])
        # Prefer the first subcategory-level mention; otherwise the first category mention.
        sub_hits = [h for h in found if any(t[1] for t in h[2])]
        _, term, targets = (sub_hits or found)[0]
        # Other words that pin down a single category ("electronics accessories") narrow it down.
        others = [h for h in found if h[1] != term and len({t[0] for t in h[2]}) == 1]
        cats_mentioned = {h[2][0][0] for h in others}
        narrowed = [t for t in targets if t[0] in cats_mentioned] or targets
        if len({t[0] for t in narrowed}) > 1 or len(narrowed) > 1 and not any(t[1] for t in narrowed):
            pq.category_options = narrowed
        elif any(t[1] for t in narrowed):
            pq.filters.category, pq.filters.subcategory = next(t for t in narrowed if t[1])
        else:
            pq.filters.category = narrowed[0][0]
        pq.category_term = term
