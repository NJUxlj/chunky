"""Hybrid search: combines vector search + BM25."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from chunky.search.bm25_engine import BM25Engine, BM25Result
from chunky.utils.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Hybrid search result with combined scores."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    labels: list[str]
    topics: list[str]
    vector_score: float
    bm25_score: float
    combined_score: float
    rank: int


class HybridSearcher:
    """Hybrid search combining vector similarity and BM25 keyword matching.
    
    Supports multiple fusion strategies:
    - RRF (Reciprocal Rank Fusion): Best for combining ranked lists
    - weighted_sum: Simple weighted combination of scores
    - relative_score: Score normalization before combination
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid searcher.
        
        Args:
            vector_weight: Weight for vector search scores (0-1).
            bm25_weight: Weight for BM25 scores (0-1).
            fusion_method: "rrf", "weighted_sum", or "relative_score".
            rrf_k: Constant for RRF fusion (default 60, from paper).
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        vector_results: list[dict[str, Any]],
        bm25_results: list[BM25Result],
        top_k: int = 10,
    ) -> list[HybridSearchResult]:
        """Combine vector and BM25 results using fusion strategy.
        
        Args:
            query: Original search query (for reference).
            vector_results: Results from vector search with fields:
                - chunk_id, text, source_file, chunk_index
                - distance (lower is better for cosine distance)
            bm25_results: Results from BM25 search.
            top_k: Number of final results to return.
            
        Returns:
            List of HybridSearchResult sorted by combined score.
        """
        if not vector_results and not bm25_results:
            logger.warning("No results from either vector or BM25 search")
            return []

        # Create result maps for quick lookup
        vector_map = {r["chunk_id"]: r for r in vector_results}
        bm25_map = {r.chunk_id: r for r in bm25_results}

        # Get union of all chunk IDs
        all_chunk_ids = list(set(vector_map.keys()) | set(bm25_map.keys()))

        # Calculate scores based on fusion method
        if self.fusion_method == "rrf":
            combined_scores = self._rrf_fusion(vector_results, bm25_results)
        elif self.fusion_method == "relative_score":
            combined_scores = self._relative_score_fusion(vector_map, bm25_map, all_chunk_ids)
        else:  # weighted_sum
            combined_scores = self._weighted_sum_fusion(vector_map, bm25_map, all_chunk_ids)

        # Sort by combined score and take top-k
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build final results
        results = []
        for rank, (chunk_id, combined_score) in enumerate(sorted_ids, 1):
            # Get data from whichever source has it
            vector_data = vector_map.get(chunk_id, {})
            bm25_data = bm25_map.get(chunk_id)

            text = vector_data.get("text") or (bm25_data.text if bm25_data else "")
            source_file = vector_data.get("source_file") or (bm25_data.source_file if bm25_data else "")
            chunk_index = vector_data.get("chunk_index", 0) or (bm25_data.chunk_index if bm25_data else 0)

            # Calculate individual scores (normalize distance to similarity)
            vector_score = 0.0
            if "distance" in vector_data:
                # ChromaDB uses cosine distance (0=identical, 2=opposite)
                # Convert to similarity score (0-1, higher is better)
                vector_score = 1.0 - (vector_data["distance"] / 2.0)
            elif "score" in vector_data:
                vector_score = float(vector_data["score"])

            bm25_score = bm25_data.score if bm25_data else 0.0

            # Extract labels and topics
            labels = self._extract_list_field(vector_data, "labels")
            topics = self._extract_list_field(vector_data, "topics")

            results.append(HybridSearchResult(
                chunk_id=chunk_id,
                text=text,
                source_file=source_file,
                chunk_index=chunk_index,
                labels=labels,
                topics=topics,
                vector_score=vector_score,
                bm25_score=bm25_score,
                combined_score=combined_score,
                rank=rank,
            ))

        logger.info(
            "Hybrid search returned %d results (vector: %d, bm25: %d, method: %s)",
            len(results), len(vector_results), len(bm25_results), self.fusion_method
        )
        return results

    def _extract_list_field(self, data: dict, field: str) -> list[str]:
        """Extract list field from metadata, handling JSON strings."""
        value = data.get(field, [])
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return []
        return value if isinstance(value, list) else []

    def _rrf_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[BM25Result],
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion - combine ranked lists.
        
        RRF formula: score(d) = sum(1 / (k + rank(d))) for each result list
        """
        scores: dict[str, float] = {}

        # Process vector results (rank by index in results list)
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + (1.0 / (self.rrf_k + rank))

        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + (1.0 / (self.rrf_k + rank))

        return scores

    def _relative_score_fusion(
        self,
        vector_map: dict[str, dict],
        bm25_map: dict[str, BM25Result],
        chunk_ids: list[str],
    ) -> dict[str, float]:
        """Relative score fusion - normalize scores to [0,1] before combining."""
        scores: dict[str, float] = {}

        # Get all vector distances
        vector_distances = [v.get("distance", float('inf')) for v in vector_map.values()]
        max_dist = max(vector_distances) if vector_distances and max(vector_distances) > 0 else 1.0

        # Get all BM25 scores
        bm25_scores = [b.score for b in bm25_map.values()]
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0

        for chunk_id in chunk_ids:
            vec_data = vector_map.get(chunk_id, {})
            bm25_data = bm25_map.get(chunk_id)

            # Vector score (convert distance to similarity)
            vector_score = 0.0
            if vec_data:
                dist = vec_data.get("distance", max_dist)
                vector_score = (1.0 - dist / max_dist) * self.vector_weight

            # BM25 score (normalized)
            bm25_score = 0.0
            if bm25_data:
                bm25_score = (bm25_data.score / max_bm25) * self.bm25_weight

            scores[chunk_id] = vector_score + bm25_score

        return scores

    def _weighted_sum_fusion(
        self,
        vector_map: dict[str, dict],
        bm25_map: dict[str, BM25Result],
        chunk_ids: list[str],
    ) -> dict[str, float]:
        """Weighted sum fusion - simple combination with weights."""
        scores: dict[str, float] = {}

        for chunk_id in chunk_ids:
            vec_data = vector_map.get(chunk_id, {})
            bm25_data = bm25_map.get(chunk_id)

            # Vector score (convert distance to similarity)
            vector_score = 0.0
            if vec_data:
                dist = vec_data.get("distance", 2.0)  # Default max distance
                vector_score = (1.0 - dist / 2.0) * self.vector_weight

            # BM25 score
            bm25_score = 0.0
            if bm25_data:
                bm25_score = bm25_data.score * self.bm25_weight

            scores[chunk_id] = vector_score + bm25_score

        return scores