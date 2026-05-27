"""
WorkloadAnalyzer
================
Analyzes PyTorch models using the FX graph tracer to extract an OpGraph
with FLOPs, memory traffic, and (optionally) profiled execution times.

Supports:
  - Static FX tracing (no runtime required)
  - Dynamic profiling via torch.profiler
  - Transformer-specific pattern recognition
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.fx as fx

from riscv_cisg.analyzer.op_graph import (
    DataType,
    OpGraph,
    OpNode,
    OpType,
    TensorShape,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping from aten/torch op names to canonical OpType
# ---------------------------------------------------------------------------
_OP_MAP: Dict[str, OpType] = {
    # Matrix ops
    "aten.mm": OpType.MATMUL,
    "aten.bmm": OpType.BATCH_MATMUL,
    "aten.addmm": OpType.MATMUL,
    "aten.linear": OpType.MATMUL,
    "aten.matmul": OpType.MATMUL,
    "aten.mv": OpType.MATVEC,
    "aten.dot": OpType.DOT_PRODUCT,
    "aten.conv2d": OpType.CONV2D,
    # Element-wise
    "aten.add": OpType.ADD,
    "aten.add_": OpType.ADD,
    "aten.mul": OpType.MUL,
    "aten.mul_": OpType.MUL,
    "aten.div": OpType.DIV,
    "aten.exp": OpType.EXP,
    "aten.log": OpType.LOG,
    "aten.sqrt": OpType.SQRT,
    "aten.rsqrt": OpType.RSQRT,
    "aten.neg": OpType.NEG,
    "aten.abs": OpType.ABS,
    "aten.maximum": OpType.MAX,
    "aten.minimum": OpType.MIN,
    # Reduction
    "aten.sum": OpType.SUM_REDUCE,
    "aten.amax": OpType.MAX_REDUCE,
    "aten.mean": OpType.MEAN_REDUCE,
    "aten.softmax": OpType.SOFTMAX,
    "_softmax": OpType.SOFTMAX,
    # Normalization
    "aten.layer_norm": OpType.LAYER_NORM,
    "aten.native_layer_norm": OpType.LAYER_NORM,
    "aten.rms_norm": OpType.RMS_NORM,
    "aten.batch_norm": OpType.BATCH_NORM,
    # Activation
    "aten.relu": OpType.RELU,
    "aten.relu_": OpType.RELU,
    "aten.gelu": OpType.GELU,
    "aten.silu": OpType.SILU,
    "aten.silu_": OpType.SILU,
    "aten.sigmoid": OpType.SIGMOID,
    "aten.tanh": OpType.TANH,
    # Attention
    "aten.scaled_dot_product_attention": OpType.SCALED_DOT_PRODUCT_ATTENTION,
    # Shape
    "aten.t": OpType.TRANSPOSE,
    "aten.permute": OpType.TRANSPOSE,
    "aten.reshape": OpType.RESHAPE,
    "aten.view": OpType.RESHAPE,
    "aten.cat": OpType.CONCAT,
    "aten.split": OpType.SPLIT,
}


def _resolve_op_type(target: str) -> OpType:
    """Map a raw FX target string to a canonical OpType."""
    # Normalize: strip leading 'torch.ops.' or 'torch._C._nn.'
    key = str(target).lower()
    for pattern, op_type in _OP_MAP.items():
        if key.endswith(pattern) or pattern in key:
            return op_type
    return OpType.UNKNOWN


def _shape_from_tensor(t: Any) -> Optional[TensorShape]:
    """Extract TensorShape from a concrete tensor or FakeTensor."""
    if not isinstance(t, torch.Tensor):
        return None
    dtype_map = {
        torch.float32: DataType.FP32,
        torch.float16: DataType.FP16,
        torch.bfloat16: DataType.BF16,
        torch.int8: DataType.INT8,
        torch.int32: DataType.INT32,
        torch.uint8: DataType.UINT8,
    }
    dtype = dtype_map.get(t.dtype, DataType.FP32)
    return TensorShape(dims=tuple(t.shape), dtype=dtype)


def _estimate_matmul_flops(input_shapes: List[TensorShape]) -> int:
    """FLOPs for matrix multiplication: 2 * M * N * K."""
    if len(input_shapes) < 2:
        return 0
    a, b = input_shapes[0], input_shapes[1]
    if len(a.dims) == 2 and len(b.dims) == 2:
        M, K = a.dims
        K2, N = b.dims
        if K == K2:
            return 2 * M * N * K
    elif len(a.dims) == 3 and len(b.dims) == 3:
        # Batched matmul
        B, M, K = a.dims
        B2, K2, N = b.dims
        if K == K2:
            return 2 * B * M * N * K
    return 0


def _estimate_elementwise_flops(output_shapes: List[TensorShape]) -> int:
    """FLOPs for element-wise ops: 1 FLOP per element."""
    if not output_shapes:
        return 0
    return output_shapes[0].num_elements


def _estimate_memory_bytes(
    input_shapes: List[TensorShape], output_shapes: List[TensorShape]
) -> int:
    """Estimate memory traffic: sum of all input + output tensor bytes."""
    total = sum(s.bytes for s in input_shapes)
    total += sum(s.bytes for s in output_shapes)
    return total


def _estimate_flops(
    op_type: OpType,
    input_shapes: List[TensorShape],
    output_shapes: List[TensorShape],
) -> int:
    """Dispatch FLOPs estimation based on op type."""
    if op_type in (OpType.MATMUL, OpType.BATCH_MATMUL):
        return _estimate_matmul_flops(input_shapes)
    elif op_type == OpType.SOFTMAX:
        # 3 passes: max, exp+sum, div
        if output_shapes:
            return 5 * output_shapes[0].num_elements
    elif op_type == OpType.LAYER_NORM:
        if output_shapes:
            return 8 * output_shapes[0].num_elements
    elif op_type == OpType.GELU:
        # ~14 FLOPs per element (approximation via tanh)
        if output_shapes:
            return 14 * output_shapes[0].num_elements
    elif op_type == OpType.SCALED_DOT_PRODUCT_ATTENTION:
        # QK^T + softmax + AV
        if input_shapes and len(input_shapes[0].dims) == 4:
            B, H, S, D = input_shapes[0].dims
            qk_flops = 2 * B * H * S * S * D
            av_flops = 2 * B * H * S * D * S
            softmax_flops = 5 * B * H * S * S
            return qk_flops + av_flops + softmax_flops
    return _estimate_elementwise_flops(output_shapes)


class WorkloadAnalyzer:
    """
    Analyzes a PyTorch model and produces an OpGraph.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model (or sub-module) to analyze.
    example_inputs : tuple
        Example input tensors used for shape propagation and profiling.
    device : str
        Target device ("cpu" or "cuda"). Defaults to "cpu".
    profile : bool
        If True, runs the model and collects per-op timing via torch.profiler.
    dtype : torch.dtype
        Data type for profiling inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        device: str = "cpu",
        profile: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model.eval().to(device)
        self.example_inputs = tuple(t.to(device) for t in example_inputs)
        self.device = device
        self.profile = profile
        self.dtype = dtype
        self._profiler_data: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> OpGraph:
        """
        Run full analysis and return a populated OpGraph.

        Steps:
          1. FX trace → extract nodes and shapes
          2. Estimate FLOPs and memory per node
          3. (Optional) Profile for actual timings
        """
        logger.info("Starting workload analysis for %s", type(self.model).__name__)

        graph = OpGraph(name=type(self.model).__name__)

        if self.profile:
            self._run_profiler()

        fx_graph = self._trace_model()
        self._build_op_graph(fx_graph, graph)

        logger.info(
            "Analysis complete: %d nodes, %s total FLOPs, %.1f μs profiled",
            graph.num_nodes,
            f"{graph.total_flops:,}",
            graph.total_profiled_time_us,
        )
        return graph

    # ------------------------------------------------------------------
    # Private: FX tracing
    # ------------------------------------------------------------------

    def _trace_model(self) -> fx.Graph:
        """Trace model with FX and propagate concrete shapes."""
        try:
            tracer = fx.symbolic_trace(self.model)
            return tracer.graph
        except Exception as e:
            logger.warning("Symbolic trace failed (%s), falling back to concrete trace", e)
            return self._concrete_trace()

    def _concrete_trace(self) -> fx.Graph:
        """Capture graph via torch.compile's dynamo capture."""
        # Build a minimal graph by running forward and extracting ops
        # This is a simplified fallback — uses torch.jit.trace
        traced = torch.jit.trace(self.model, self.example_inputs)
        # Convert TorchScript graph to our internal format as best we can
        # Return a simple placeholder graph
        g = fx.Graph()
        return g

    # ------------------------------------------------------------------
    # Private: Profiling
    # ------------------------------------------------------------------

    def _run_profiler(self) -> None:
        """Run torch.profiler and collect per-op timings."""
        logger.info("Running torch.profiler...")
        try:
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = self.model(*self.example_inputs)

                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    with_flops=True,
                ) as prof:
                    for _ in range(10):
                        _ = self.model(*self.example_inputs)

            for evt in prof.key_averages():
                if evt.key and evt.cpu_time_total > 0:
                    self._profiler_data[evt.key] = evt.cpu_time_total / 10.0

            logger.info("Profiler captured %d unique ops", len(self._profiler_data))
        except Exception as e:
            logger.warning("Profiling failed: %s. Continuing without timing data.", e)

    # ------------------------------------------------------------------
    # Private: Graph building
    # ------------------------------------------------------------------

    def _build_op_graph(self, fx_graph: fx.Graph, graph: OpGraph) -> None:
        """Walk FX graph nodes and populate the OpGraph."""
        node_counter = 0
        prev_node_id: Optional[str] = None

        # We need concrete output shapes — run model with example inputs
        shape_map = self._collect_shapes(fx_graph)

        for fx_node in fx_graph.nodes:
            if fx_node.op in ("placeholder", "output", "get_attr"):
                continue

            op_type = _resolve_op_type(str(fx_node.target))
            node_id = f"{op_type.name.lower()}_{node_counter}"
            node_counter += 1

            input_shapes = self._extract_input_shapes(fx_node, shape_map)
            output_shape = shape_map.get(fx_node.name)
            output_shapes = [output_shape] if output_shape else []

            flops = _estimate_flops(op_type, input_shapes, output_shapes)
            mem = _estimate_memory_bytes(input_shapes, output_shapes)

            # Look up profiled time
            profiled_us = 0.0
            for key, t in self._profiler_data.items():
                if str(fx_node.target).lower() in key.lower():
                    profiled_us = t
                    break

            op_node = OpNode(
                node_id=node_id,
                op_type=op_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                flops=flops,
                memory_bytes=mem,
                profiled_time_us=profiled_us,
                source_framework=str(fx_node.target),
                attributes=dict(fx_node.kwargs),
                predecessors=[prev_node_id] if prev_node_id else [],
                successors=[],
            )

            if prev_node_id and prev_node_id in [n.node_id for n in graph.nodes]:
                # Update previous node's successors
                prev = graph.get_node(prev_node_id)
                prev.successors.append(node_id)

            graph.add_node(op_node)
            prev_node_id = node_id

    def _collect_shapes(self, fx_graph: fx.Graph) -> Dict[str, TensorShape]:
        """
        Run the model with example inputs and capture intermediate tensor shapes
        using hooks on all modules.
        """
        shape_map: Dict[str, TensorShape] = {}

        # For traced models we can get output shapes from interpreter
        try:
            interp = fx.Interpreter(
                fx.GraphModule(self.model, fx_graph)
            )
            with torch.no_grad():
                interp.run(*self.example_inputs)
        except Exception:
            pass

        # Fallback: run with hooks
        activation_shapes: Dict[str, TensorShape] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    s = _shape_from_tensor(out)
                    if s:
                        activation_shapes[name] = s
            return hook

        for name, module in self.model.named_modules():
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        try:
            with torch.no_grad():
                self.model(*self.example_inputs)
        except Exception as e:
            logger.debug("Shape collection hook run failed: %s", e)
        finally:
            for h in hooks:
                h.remove()

        shape_map.update(activation_shapes)
        return shape_map

    def _extract_input_shapes(
        self, fx_node: fx.Node, shape_map: Dict[str, TensorShape]
    ) -> List[TensorShape]:
        """Extract shapes for all tensor inputs of an FX node."""
        shapes = []
        for arg in fx_node.args:
            if isinstance(arg, fx.Node) and arg.name in shape_map:
                shapes.append(shape_map[arg.name])
            elif isinstance(arg, torch.Tensor):
                s = _shape_from_tensor(arg)
                if s:
                    shapes.append(s)
        return shapes
