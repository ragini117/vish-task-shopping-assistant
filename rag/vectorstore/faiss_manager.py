"""Build, persist and load the FAISS product index."""

import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL
from rag.ingestion.product_ingestion import build_documents

logger = logging.getLogger(__name__)
META_FILE = "index_meta.json"


class VectorStoreError(Exception):
    """Raised when the FAISS index cannot be built or loaded."""


def check_faiss_import() -> None:
    """Import faiss up front so a missing/broken install is reported as such (with its traceback),
    instead of surfacing later as a misleading 'corrupted index' or 'embedding failed' error."""
    try:
        import faiss  # noqa: F401
    except Exception as exc:
        logger.exception("Could not import faiss (python: %s)", sys.executable)
        raise VectorStoreError(
            f"FAISS could not be imported in this Python environment ({sys.executable}): "
            f"{type(exc).__name__}: {exc}. Install it with: pip install faiss-cpu"
        ) from exc


class FaissManager:
    def __init__(self, index_dir: Path, embeddings):
        self.index_dir = Path(index_dir)
        self.embeddings = embeddings
        self.store: FAISS | None = None
        self.status = ""

    def _read_meta(self) -> dict:
        try:
            return json.loads((self.index_dir / META_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def load_or_build(self, df: pd.DataFrame, fingerprint: str, force_rebuild: bool = False) -> FAISS:
        """Load the saved index if it matches the dataset, otherwise (re)build it."""
        check_faiss_import()
        meta = self._read_meta()
        mismatches = [
            name for name, saved, current in (
                ("dataset fingerprint", meta.get("fingerprint"), fingerprint),
                ("embedding model", meta.get("model"), EMBEDDING_MODEL),
                ("product count", meta.get("count"), len(df)),
            ) if saved != current
        ]
        has_files = (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
        reason = None
        if force_rebuild:
            reason = "forced rebuild"
        elif not meta or not has_files:
            reason = "no saved index"
        elif mismatches:
            reason = "changed " + ", ".join(mismatches)
        else:
            try:
                self.store = FAISS.load_local(
                    str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True
                )
                if self.store.index.ntotal != len(df):
                    raise VectorStoreError(f"index has {self.store.index.ntotal} vectors, dataset has {len(df)}")
                self.status = f"Loaded saved FAISS index ({self.store.index.ntotal} products)"
                logger.info("%s from %s", self.status, self.index_dir)
                return self.store
            except (ImportError, ModuleNotFoundError):
                raise  # environment problem, rebuilding would not help
            except Exception as exc:  # corrupted or partial index files
                logger.exception("Saved FAISS index at %s could not be loaded", self.index_dir)
                reason = f"saved index unreadable ({type(exc).__name__}: {exc})"

        logger.info("Building FAISS index: %s", reason)
        self.store = self._build(df, fingerprint)
        self.status = f"Built new FAISS index ({len(df)} products, {reason})"
        return self.store

    def _build(self, df: pd.DataFrame, fingerprint: str) -> FAISS:
        docs = build_documents(df)
        try:
            store = FAISS.from_documents(docs, self.embeddings)
        except Exception as exc:
            logger.exception("Building the FAISS index failed")
            raise VectorStoreError(f"Building the FAISS index failed: {type(exc).__name__}: {exc}") from exc
        tmp_dir = self.index_dir.with_name(self.index_dir.name + "_tmp")
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            store.save_local(str(tmp_dir))
            (tmp_dir / META_FILE).write_text(
                json.dumps({"fingerprint": fingerprint, "model": EMBEDDING_MODEL, "count": len(df)}),
                encoding="utf-8",
            )
            shutil.rmtree(self.index_dir, ignore_errors=True)
            tmp_dir.rename(self.index_dir)
        except OSError as exc:  # index still usable in memory
            logger.warning("Could not save FAISS index to %s: %s", self.index_dir, exc)
        return store

    def similarity_scores(self, query: str) -> dict[str, float]:
        """Cosine similarity (0..1) of the query against every product."""
        if self.store is None or not query.strip():
            return {}
        k = self.store.index.ntotal
        results = self.store.similarity_search_with_score(query, k=k)
        # Normalized vectors: squared L2 distance d = 2 - 2*cos  ->  cos = 1 - d/2
        return {doc.metadata["product_id"]: max(0.0, 1.0 - float(dist) / 2.0) for doc, dist in results}
