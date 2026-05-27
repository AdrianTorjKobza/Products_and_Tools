"""
HotspotDetector
===============
Scores each node in an OpGraph and identifies top-N hotspots —
the operations most worth accelerating with a custom instruction.

Scoring methodology (weighted sum):
  - Execution time share  (40%)  — only available if profiled
  - FLOPs share           (35%)
  - Memory traffic share  (15%)
  - Compute-bound flag    (10%)

If no profiling data is available, weighting shifts to FLOPs + memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from riscv_cisg.analyzer.op_graph import OpGraph, OpNode, OpType

logger = logging.getLogger(__name__)

# Operations that are structurally acceleratable via custom ISA extensions
_ACCELERATABLE_OPS = {
    OpType.MATMUL,
    OpType.BATCH_MATMUL,
    OpType.MATVEC,
    OpType.DOT_PRODUCT,
    OpType.FUSED_MULTIPLY_ADD,
    OpType.CONV2D,
    OpType.SOFTMAX,
    OpType.LAYER_NORM,
    OpType.RMS_NORM,
    OpType.GELU,
    OpType.SILU,
    OpType.SCALED_DOT_PRODUCT_ATTENTION,
    OpType.ATTENTION_SCORE,
    OpType.SUM_REDUCE,
    OpType.MAX_REDUCE,
}


@dataclass
class HotspotResult:
    """
    Analysis result for a single hotspot.

    Attributes
    ----------
    node : OpNode
        The op graph node identified as a hotspot.
    hotspot_score : float
        Composite score in [0, 1]. Higher = more impactful to accelerate.
    time_pct : float
        Percentage of total profiled time this node consumes.
    flops_pct : float
        Percentage of total workload FLOPs.
    memory_pct : float
        Percentage of total memory traffic.
    is_acceleratable : bool
        Whether the op type is a good candidate for a custom instruction.
    acceleration_rationale : str
        Human-readable explanation of why this is a hotspot.
    """
    node: OpNode
    hotspot_score: float
    time_pct: float
    flops_pct: float
    memory_pct: float
    is_acceleratable: bool
    acceleration_rationale: str = ""
    rank: int = 0

    def __repr__(self) -> str:
        return (
            f"HotspotResult(rank={self.rank}, "
            f"op={self.node.op_type.name}, "
            f"score={self.hotspot_score:.3f}, "
            f"time={self.time_pct:.1f}%, "
            f"flops={self.flops_pct:.1f}%)"
        )


class HotspotDetector:
    """
    Identifies the top-N most impactful nodes in an OpGraph.

    Parameters
    ----------
    graph : OpGraph
        The analyzed workload graph.
    top_n : int
        Number of top hotspots to return.
    min_flop_threshold : int
        Minimum FLOPs for a node to be considered (filters trivial ops).
    only_acceleratable : bool
        If True, only return nodes whose op type is in _ACCELERATABLE_OPS.
    """

    def __init__(
        self,
        graph: OpGraph,
        top_n: int = 5,
        min_flop_threshold: int = 1_000,
        only_acceleratable: bool = True,
    ):
        self.graph = graph
        self.top_n = top_n
        self.min_flop_threshold = min_flop_threshold
        self.only_acceleratable = only_acceleratable

    def detect(self) -> List[HotspotResult]:
        """
        Run hotspot detection and return ranked list of HotspotResults.

        Returns
        -------
        list of HotspotResult
            Top-N hotspots, sorted by hotspot_score descending.
        """
        nodes = self.graph.nodes
        if not nodes:
            logger.warning("Empty graph — no hotspots to detect.")
            return []

        total_flops = max(self.graph.total_flops, 1)
        total_memory = max(self.graph.total_memory_bytes, 1)
        total_time = max(self.graph.total_profiled_time_us, 1e-9)
        has_timing = self.graph.total_profiled_time_us > 0

        results: List[HotspotResult] = []

        for node in nodes:
            if node.flops < self.min_flop_threshold:
                continue
            if self.only_acceleratable and node.op_type not in _ACCELERATABLE_OPS:
                continue

            flops_pct = (node.flops / total_flops) * 100
            memory_pct = (node.memory_bytes / total_memory) * 100
            time_pct = (
                (node.profiled_time_us / total_time) * 100 if has_timing else 0.0
            )

            score = self._compute_score(
                flops_pct, memory_pct, time_pct, node, has_timing
            )

            is_acc = node.op_type in _ACCELERATABLE_OPS
            rationale = self._build_rationale(node, flops_pct, time_pct, memory_pct)

            results.append(
                HotspotResult(
                    node=node,
                    hotspot_score=score,
                    time_pct=time_pct,
                    flops_pct=flops_pct,
                    memory_pct=memory_pct,
                    is_acceleratable=is_acc,
                    acceleration_rationale=rationale,
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.hotspot_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(results[: self.top_n]):
            r.rank = i + 1

        logger.info(
            "Detected %d hotspots (top %d returned)",
            len(results),
            min(self.top_n, len(results)),
        )
        return results[: self.top_n]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        flops_pct: float,
        memory_pct: float,
        time_pct: float,
        node: OpNode,
        has_timing: bool,
    ) -> float:
        """Weighted composite score in [0, 100]."""
        if has_timing:
            # With profiling data: time is the primary signal
            score = (
                0.40 * time_pct
                + 0.35 * flops_pct
                + 0.15 * memory_pct
                + 0.10 * (100 if node.is_compute_bound else 0)
            )
        else:
            # Without profiling: FLOPs + memory only
            score = (
                0.55 * flops_pct
                + 0.35 * memory_pct
                + 0.10 * (100 if node.is_compute_bound else 0)
            )
        return min(score, 100.0)

    def _build_rationale(
        self,
        node: OpNode,
        flops_pct: float,
        time_pct: float,
        memory_pct: float,
    ) -> str:
        """Construct a human-readable rationale for why this is a hotspot."""
        parts = []

        if flops_pct >= 30:
            parts.append(f"dominates compute at {flops_pct:.1f}% of total FLOPs")
        elif flops_pct >= 10:
            parts.append(f"significant compute contributor ({flops_pct:.1f}% of FLOPs)")

        if time_pct >= 20:
            parts.append(f"accounts for {time_pct:.1f}% of measured execution time")

        if memory_pct >= 25:
            parts.append(f"high memory traffic ({memory_pct:.1f}% of total)")

        if node.is_compute_bound:
            parts.append(
                f"compute-bound (arithmetic intensity={node.arithmetic_intensity:.1f} FLOPs/byte)"
            )
        else:
            parts.append(
                f"memory-bound (arithmetic intensity={node.arithmetic_intensity:.1f} FLOPs/byte)"
            )

        if node.op_type in _ACCELERATABLE_OPS:
            parts.append(f"op type {node.op_type.name} maps well to custom ISA extension")

        return "; ".join(parts) if parts else "identified as top compute consumer"


def detect_hotspots_from_graph(
    graph: OpGraph,
    top_n: int = 5,
    only_acceleratable: bool = True,
) -> List[HotspotResult]:
    """
    Convenience function: run HotspotDetector on a graph and return results.

    Parameters
    ----------
    graph : OpGraph
    top_n : int
    only_acceleratable : bool

    Returns
    -------
    list of HotspotResult
    """
    detector = HotspotDetector(graph, top_n=top_n, only_acceleratable=only_acceleratable)
    return detector.detect()
