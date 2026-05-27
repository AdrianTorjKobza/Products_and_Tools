"""
Operation Graph representation for ML workload analysis.
Represents a computation graph as a DAG of typed operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple
import json


class OpType(Enum):
    """Canonical operation types extracted from ML frameworks."""
    # Linear algebra
    MATMUL = auto()
    BATCH_MATMUL = auto()
    MATVEC = auto()
    DOT_PRODUCT = auto()
    OUTER_PRODUCT = auto()
    CONV2D = auto()

    # Element-wise
    ADD = auto()
    MUL = auto()
    FUSED_MULTIPLY_ADD = auto()
    DIV = auto()
    EXP = auto()
    LOG = auto()
    SQRT = auto()
    RSQRT = auto()
    NEG = auto()
    ABS = auto()
    MAX = auto()
    MIN = auto()

    # Reduction
    SUM_REDUCE = auto()
    MAX_REDUCE = auto()
    MEAN_REDUCE = auto()
    SOFTMAX = auto()

    # Normalization
    LAYER_NORM = auto()
    RMS_NORM = auto()
    BATCH_NORM = auto()

    # Activation
    RELU = auto()
    GELU = auto()
    SILU = auto()
    SIGMOID = auto()
    TANH = auto()

    # Attention-specific
    SCALED_DOT_PRODUCT_ATTENTION = auto()
    ATTENTION_SCORE = auto()

    # Memory / shape
    TRANSPOSE = auto()
    RESHAPE = auto()
    CONCAT = auto()
    SPLIT = auto()

    # Unknown
    UNKNOWN = auto()


class DataType(Enum):
    """Supported data types."""
    FP32 = "f32"
    FP16 = "f16"
    BF16 = "bf16"
    INT8 = "i8"
    INT32 = "i32"
    UINT8 = "u8"


@dataclass
class TensorShape:
    """Represents a tensor's shape and type."""
    dims: Tuple[int, ...]
    dtype: DataType = DataType.FP32

    @property
    def num_elements(self) -> int:
        result = 1
        for d in self.dims:
            result *= d
        return result

    @property
    def bytes(self) -> int:
        dtype_bytes = {
            DataType.FP32: 4, DataType.FP16: 2, DataType.BF16: 2,
            DataType.INT8: 1, DataType.INT32: 4, DataType.UINT8: 1,
        }
        return self.num_elements * dtype_bytes[self.dtype]

    def __str__(self) -> str:
        return f"[{', '.join(str(d) for d in self.dims)}]:{self.dtype.value}"


@dataclass
class OpNode:
    """
    A single operation node in the computation graph.

    Attributes
    ----------
    node_id : str
        Unique identifier for this node.
    op_type : OpType
        The canonical operation type.
    input_shapes : list of TensorShape
        Shapes of all input tensors.
    output_shapes : list of TensorShape
        Shapes of all output tensors.
    flops : int
        Estimated floating point operations for this node.
    memory_bytes : int
        Estimated memory traffic (reads + writes) in bytes.
    attributes : dict
        Additional op-specific attributes (e.g., stride, dilation).
    profiled_time_us : float
        Measured execution time in microseconds (0 if not profiled).
    source_framework : str
        Original framework op name (e.g., "aten::mm").
    """
    node_id: str
    op_type: OpType
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shapes: List[TensorShape] = field(default_factory=list)
    flops: int = 0
    memory_bytes: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    profiled_time_us: float = 0.0
    source_framework: str = ""
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte of memory traffic (roofline metric)."""
        if self.memory_bytes == 0:
            return float("inf")
        return self.flops / self.memory_bytes

    @property
    def is_compute_bound(self) -> bool:
        """Heuristic: arithmetic intensity > 32 FLOPs/byte → compute bound."""
        return self.arithmetic_intensity > 32.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op_type": self.op_type.name,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "arithmetic_intensity": round(self.arithmetic_intensity, 3),
            "profiled_time_us": self.profiled_time_us,
            "source_framework": self.source_framework,
            "input_shapes": [str(s) for s in self.input_shapes],
            "output_shapes": [str(s) for s in self.output_shapes],
            "attributes": self.attributes,
        }


class OpGraph:
    """
    Directed Acyclic Graph of OpNodes representing a complete ML workload.

    The graph is built by the WorkloadAnalyzer and consumed by the
    HotspotDetector and instruction proposers.
    """

    def __init__(self, name: str = "unnamed_workload"):
        self.name = name
        self._nodes: Dict[str, OpNode] = {}
        self._topo_order: Optional[List[str]] = None

    def add_node(self, node: OpNode) -> None:
        self._nodes[node.node_id] = node
        self._topo_order = None  # invalidate cache

    def get_node(self, node_id: str) -> OpNode:
        return self._nodes[node_id]

    @property
    def nodes(self) -> List[OpNode]:
        return list(self._nodes.values())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def total_flops(self) -> int:
        return sum(n.flops for n in self._nodes.values())

    @property
    def total_memory_bytes(self) -> int:
        return sum(n.memory_bytes for n in self._nodes.values())

    @property
    def total_profiled_time_us(self) -> float:
        return sum(n.profiled_time_us for n in self._nodes.values())

    def get_nodes_by_type(self, op_type: OpType) -> List[OpNode]:
        return [n for n in self._nodes.values() if n.op_type == op_type]

    def topological_order(self) -> List[str]:
        """Kahn's algorithm for topological sort."""
        if self._topo_order is not None:
            return self._topo_order

        in_degree = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for succ in node.successors:
                if succ in in_degree:
                    in_degree[succ] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for succ in self._nodes[nid].successors:
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        self._topo_order = order
        return order

    def subgraph(self, node_ids: List[str]) -> "OpGraph":
        """Extract a subgraph containing only the specified node IDs."""
        sg = OpGraph(name=f"{self.name}_subgraph")
        id_set = set(node_ids)
        for nid in node_ids:
            node = self._nodes[nid]
            import copy
            new_node = copy.deepcopy(node)
            new_node.successors = [s for s in node.successors if s in id_set]
            new_node.predecessors = [p for p in node.predecessors if p in id_set]
            sg.add_node(new_node)
        return sg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "total_flops": self.total_flops,
            "total_memory_bytes": self.total_memory_bytes,
            "total_profiled_time_us": self.total_profiled_time_us,
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return (
            f"OpGraph(name='{self.name}', nodes={self.num_nodes}, "
            f"total_flops={self.total_flops:,}, "
            f"total_time_us={self.total_profiled_time_us:.1f})"
        )
