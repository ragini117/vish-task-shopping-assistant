"""Product retrieval: semantic relevance (FAISS) first, then strict metadata filters, then ranking.

1. Relevant set — which products are about what the shopper asked for:
   * a catalog type named in the query ("boots", "Running Shoes") -> every product of that type;
   * free text ("something to listen to music") -> products whose cosine similarity to the text is
     >= SEMANTIC_MIN_SCORE and within SEMANTIC_TOP_MARGIN of the best score. When at least
     SEMANTIC_SUBCATEGORY_SHARE of those hits share one subcategory, the whole subcategory is relevant
     (otherwise a threshold alone keeps e.g. 30 of the 34 boots). If a category was named and the
     describing words match nothing ("comfortable shoes"), the whole named category is relevant;
   * filters only ("under ₹300", a brand) -> the whole catalog, then the filters narrow it;
   * nothing at all -> nothing, unless the shopper explicitly asked to browse everything.
2. Every explicit filter is applied strictly. Nothing is relaxed automatically.
3. The complete ranked list is returned (no top-k cut); the UI pages through all of it.

There is no keyword or substring fallback: if FAISS is unavailable or fails, SemanticSearchError is raised
and the caller shows the error.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

from config import SEMANTIC_MIN_SCORE, SEMANTIC_SUBCATEGORY_SHARE, SEMANTIC_TOP_MARGIN
from rag.retrieval.product_filter import SearchFilters, apply_filters, sort_products

logger = logging.getLogger(__name__)


class SemanticSearchError(Exception):
    """Semantic retrieval is unavailable or failed; product search cannot run."""


@dataclass
class RetrievalResult:
    products: pd.DataFrame          # THE final list: relevant, filtered, ranked, complete
    filters: SearchFilters          # filters actually applied (including a semantically inferred subcategory)
    relevant: pd.DataFrame          # relevant products before filters (for "nothing under ₹200" suggestions)
    ordering: str                   # human label of the order products are listed in
    name_matches: int = 0           # results whose name is the product the shopper named
    inferred_subcategory: str | None = None

    @property
    def total(self) -> int:
        return len(self.products)


class ProductRetriever:
    def __init__(self, df: pd.DataFrame, faiss_manager=None):
        self.df = df
        self.faiss = faiss_manager
        self._cache: OrderedDict[str, pd.Series] = OrderedDict()
        self.semantic_error: str | None = None   # last runtime failure of semantic search, if any

    @property
    def semantic_available(self) -> bool:
        return self.faiss is not None and self.semantic_error is None

    @property
    def mode_label(self) -> str:
        if self.faiss is None:
            return "Semantic search unavailable — product search is disabled"
        if self.semantic_error:
            return f"Semantic search failed ({self.semantic_error}) — product search is disabled"
        return "Semantic (FAISS) search + strict filters"

    # ------------------------------------------------------------------ scores
    def semantic_scores(self, query: str) -> pd.Series:
        """Cosine similarity of every product to the query. Raises SemanticSearchError on any failure."""
        if self.faiss is None:
            raise SemanticSearchError("the FAISS index is not loaded")
        key = query.strip().lower()
        if key in self._cache:
            return self._cache[key]
        try:
            sims = self.faiss.similarity_scores(query)
        except Exception as exc:
            logger.exception("Semantic search failed for query %r", query)
            self.semantic_error = type(exc).__name__
            raise SemanticSearchError(f"{type(exc).__name__}: {exc}") from exc
        self.semantic_error = None
        scores = pd.Series(sims, dtype=float).reindex(self.df.index).fillna(0.0)
        self._cache[key] = scores
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return scores

    def semantic_hits(self, query: str, within: pd.DataFrame | None = None) -> pd.DataFrame:
        """Products the query is semantically about (threshold + margin from the best score)."""
        frame = self.df if within is None else within
        if not query.strip() or frame.empty:
            return frame.iloc[0:0]
        sem = self.semantic_scores(query).reindex(frame.index)
        top = float(sem.max())
        return frame[(sem >= SEMANTIC_MIN_SCORE) & (sem >= top - SEMANTIC_TOP_MARGIN)]

    def carries(self, query: str) -> bool:
        """Does the catalog contain anything semantically matching `query`? ("pen" -> False)"""
        return not self.semantic_hits(query).empty

    # --------------------------------------------------------------- retrieval
    def relevant_products(self, query: str, filters: SearchFilters,
                          browse_all: bool = False) -> tuple[pd.DataFrame, str | None]:
        """Step 1: the relevant set, before price/brand/size/... filters. Returns (products, inferred subcategory)."""
        scope = apply_filters(self.df, SearchFilters(category=filters.category, subcategory=filters.subcategory))
        if filters.subcategory:
            return scope, None
        if query.strip():
            hits = self.semantic_hits(query, scope)
            if hits.empty:
                # The shopper named a category ("comfortable shoes"): describing words that match no product
                # strongly still leave every product of that category relevant. Without a named category
                # ("pen") nothing is relevant.
                return (scope, None) if filters.category else (hits, None)
            counts = hits["subcategory"].value_counts()
            if counts.iloc[0] / len(hits) >= SEMANTIC_SUBCATEGORY_SHARE:
                sub = counts.index[0]
                return scope[scope["subcategory"] == sub], sub
            return hits, None
        if filters.category or filters.has_constraints() or browse_all:
            return scope, None
        return self.df.iloc[0:0], None   # an empty request is never "all products"

    def retrieve(self, query: str, filters: SearchFilters, browse_all: bool = False) -> RetrievalResult:
        """Relevant set -> strict filters -> ranking. Returns every matching product."""
        if self.faiss is None:
            raise SemanticSearchError("the FAISS index is not loaded")
        relevant, inferred = self.relevant_products(query, filters, browse_all)
        applied = filters
        if inferred:
            category = relevant["category"].iloc[0]
            applied = filters.copy(category=category, subcategory=inferred)
        matches = apply_filters(relevant, applied)

        if query.strip() and not matches.empty:
            sem = self.semantic_scores(query).reindex(matches.index)
            ordering = "relevance"
        else:
            sem = pd.Series(0.0, index=matches.index)
            ordering = "highest rated first"
        relevance = sem + 0.01 * matches["rating"]          # rating only breaks ties
        if applied.target_price is not None:
            closeness = 1 - ((matches["sale_price"] - applied.target_price).abs() / applied.target_price).clip(upper=1)
            relevance = relevance + closeness

        ranked = matches.assign(_relevance=relevance, _semantic=sem, _name_match=applied.name_mask(matches))
        ranked = ranked[~ranked.index.duplicated()]
        ranked = ranked.sort_values("_relevance", ascending=False, kind="mergesort")
        if applied.sort_by:
            ranked = sort_products(ranked, applied.sort_by)
        # The product the shopper named always comes first; rating or sort order never pushes it down.
        ranked = ranked.sort_values("_name_match", ascending=False, kind="mergesort")
        ranked = self._validated(ranked, applied)
        return RetrievalResult(ranked, applied, relevant, ordering, int(ranked["_name_match"].sum()), inferred)

    @staticmethod
    def _validated(ranked: pd.DataFrame, used: SearchFilters) -> pd.DataFrame:
        """Last line of defence: every returned product must satisfy the filters reported as applied."""
        valid = apply_filters(ranked, used.copy(budget=False))
        if len(valid) != len(ranked):
            dropped = sorted(set(ranked.index) - set(valid.index))
            logger.error("Dropped %d products that violate the applied filters: %s", len(dropped), dropped[:10])
            ranked = ranked.loc[valid.index]
        return ranked
