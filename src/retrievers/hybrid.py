"""Hybrid retriever combining BM25 and vector search using Reciprocal Rank Fusion."""

from typing import List, Dict, Any, Optional, Tuple
from .bm25 import BM25Retriever
from .vector import VectorRetriever


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search using Reciprocal Rank Fusion (RRF).

    RRF combines results from multiple retrievers by ranking them based on their
    position in each result list, rather than their raw scores.
    
    The class exposes several adjustable parameters so you can experiment:
      - bm25_top_k: how many results to request from the BM25 retriever
      - vector_top_k: how many results to request from the vector retriever
      - bm25_weight, vector_weight: multiplicative weights for each retriever's contribution
      - rrf_k: the RRF constant (standard formula uses 1 / (k + rank))
      - final_k: number of results to return after fusion (overrides search `k` if provided)
      - use_scores: whether to include the original retriever scores in output
    """

    def __init__(
            self,
            bm25_top_k: int = 100,
            vector_top_k: int = 100,
            bm25_weight: float = 1.0,
            vector_weight: float = 1.0,
            rrf_k: float = 60.0,
            ):
        """
        Initialize the hybrid retriever with configurable parameters.

        Args:
            bm25_top_k: number of candidate docs to fetch from BM25 (default 100)
            vector_top_k: number of candidate docs to fetch from vector search (default 100)
            bm25_weight: multiplier applied to the BM25 RRF contribution (default 1.0)
            vector_weight: multiplier applied to the vector RRF contribution (default 1.0)
            rrf_k: the RRF constant added to rank in denominator (default 60.0)
        """
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()

        # tunable parameters
        self.bm25_top_k = int(bm25_top_k)
        self.vector_top_k = int(vector_top_k)
        self.bm25_weight = float(bm25_weight)
        self.vector_weight = float(vector_weight)
        self.rrf_k = float(rrf_k)

    def _rank_list_to_positions(self, results: List[Dict[str, Any]]) -> Dict[Any, int]:
        """
        Convert retriever results (list ordered by relevance) into a dict mapping id -> rank (1-based).
        If duplicates occur inside the same list, keep the smallest rank (best position).
        """
        positions: Dict[Any, int] = {}
        for idx, item in enumerate(results, start=1):
            doc_id = item.get("id")
            if doc_id is None:
                continue
            # keep the best (smallest) rank if doc appears multiple times
            if doc_id not in positions or idx < positions[doc_id]:
                positions[doc_id] = idx
        return positions
    
    def _merge_doc_info(
        self,
        combined: Dict[Any, Dict[str, Any]],
        source_results: List[Dict[str, Any]],
        source_name: str,
    ) -> None:
        """
        Merge the content/metadata/score fields from a source result list into `combined`.
        Only sets fields if they are missing; does not overwrite an existing content/metadata
        (we preserve first-seen content).
        """
        for item in source_results:
            doc_id = item.get("id")
            if doc_id is None:
                continue
            entry = combined.setdefault(doc_id, {})
            # Preserve first-seen content/metadata; store source-specific values under names
            if "content" not in entry and "content" in item:
                entry["content"] = item["content"]
            if "metadata" not in entry and "metadata" in item:
                entry["metadata"] = item["metadata"]
            # store raw scores under prefixed keys so downstream code can inspect them
            # e.g. bm25_score or vector_similarity
            if source_name == "bm25" and "score" in item:
                entry.setdefault("bm25_score", item["score"])
            if source_name == "vector" and "similarity" in item:
                entry.setdefault("vector_similarity", item["similarity"])


    def search(
        self,
        query: str,
        k: int = 5,
        bm25_top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        rrf_k: Optional[float] = None,
        final_k: Optional[int] = None,
        use_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).

        Args:
            query: search query text
            k: fallback number of results to return if final_k is not provided (default 5)
            bm25_top_k: override instance bm25_top_k for this search
            vector_top_k: override instance vector_top_k for this search
            bm25_weight: override instance bm25_weight for this search
            vector_weight: override instance vector_weight for this search
            rrf_k: override instance rrf_k for this search
            final_k: explicit number of results to return after fusion (overrides `k`)
            use_scores: include raw retriever scores in output (default True)

        Returns:
            List of result dicts containing at least: id, content (if available),
            metadata (if available), and rrf_score. If use_scores=True the dict
            will also contain bm25_score / vector_similarity when available.
        """
        # Resolve effective parameters (search-level overrides take precedence)
        bm25_top_k = int(bm25_top_k) if bm25_top_k is not None else self.bm25_top_k
        vector_top_k = int(vector_top_k) if vector_top_k is not None else self.vector_top_k
        bm25_weight = float(bm25_weight) if bm25_weight is not None else self.bm25_weight
        vector_weight = float(vector_weight) if vector_weight is not None else self.vector_weight
        rrf_k = float(rrf_k) if rrf_k is not None else self.rrf_k
        final_k = int(final_k) if final_k is not None else int(k)

        # Get results from both retrievers
        try:
            bm25_results = self.bm25_retriever.search(query, k=bm25_top_k) or []
        except Exception as e:
            # fail-safe: empty list if retriever errors
            print(f"[HybridRetriever] BM25 search error: {e}")
            bm25_results = []

        try:
            vector_results = self.vector_retriever.search(query, k=vector_top_k) or []
        except Exception as e:
            print(f"[HybridRetriever] Vector search error: {e}")
            vector_results = []

        # Convert the ranked lists into position maps (doc_id -> rank)
        bm25_positions = self._rank_list_to_positions(bm25_results)
        vector_positions = self._rank_list_to_positions(vector_results)

        # Build the set of candidate document ids (union of both result sets)
        candidate_ids = set(bm25_positions.keys()).union(set(vector_positions.keys()))

        # Compute RRF score for each candidate and sum contributions
        # RRF contribution from a retriever for a doc = weight * (1 / (rrf_k + rank))
        # docs not present in a retriever simply receive 0 contribution from it.
        fused: Dict[Any, Dict[str, Any]] = {}
        # Merge basic doc info (content/metadata/score fields)
        self._merge_doc_info(fused, bm25_results, "bm25")
        self._merge_doc_info(fused, vector_results, "vector")

        for doc_id in candidate_ids:
            total_score = 0.0
            bm25_rank = bm25_positions.get(doc_id)
            vector_rank = vector_positions.get(doc_id)

            if bm25_rank is not None:
                total_score += bm25_weight * (1.0 / (rrf_k + bm25_rank))
            if vector_rank is not None:
                total_score += vector_weight * (1.0 / (rrf_k + vector_rank))

            # Save computed fields
            entry = fused.setdefault(doc_id, {})
            entry["id"] = doc_id
            entry["rrf_score"] = total_score
            if bm25_rank is not None:
                entry["bm25_rank"] = bm25_rank
            if vector_rank is not None:
                entry["vector_rank"] = vector_rank

        # Convert fused dict to a sorted list (descending by rrf_score)
        # If multiple docs have same rrf_score, we tie-break by:
        #  - presence in both lists (prefer those), then by best min(rank), then by id
        def sort_key(item: Tuple[Any, Dict[str, Any]]):
            doc_id, data = item
            in_bm25 = 1 if "bm25_rank" in data else 0
            in_vector = 1 if "vector_rank" in data else 0
            presence = in_bm25 + in_vector  # 2 means present in both
            best_rank = min(
                [r for r in (data.get("bm25_rank"), data.get("vector_rank")) if r is not None],
                default=999999,
            )
            # Primary: rrf_score desc, Secondary: presence desc, Tertiary: best_rank asc
            return (-data.get("rrf_score", 0.0), -presence, best_rank, str(doc_id))

        sorted_items = sorted(fused.items(), key=sort_key)

        # Build final return list of dicts and include raw scores optionally
        results: List[Dict[str, Any]] = []
        for doc_id, data in sorted_items[:final_k]:
            out = {
                "id": doc_id,
                "rrf_score": data.get("rrf_score", 0.0),
            }
            # include content/metadata if present
            if "content" in data:
                out["content"] = data["content"]
            if "metadata" in data:
                out["metadata"] = data["metadata"]
            # include ranks (useful for debugging/tuning)
            if "bm25_rank" in data:
                out["bm25_rank"] = data["bm25_rank"]
            if "vector_rank" in data:
                out["vector_rank"] = data["vector_rank"]

            if use_scores:
                if "bm25_score" in data:
                    out["bm25_score"] = data["bm25_score"]
                if "vector_similarity" in data:
                    out["vector_similarity"] = data["vector_similarity"]

            results.append(out)

        return results

    def __repr__(self) -> str:
        return "HybridRetriever()"
