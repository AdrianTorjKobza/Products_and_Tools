#!/usr/bin/env python3
"""
riscv-cisg CLI
==============
Command-line interface for the RISC-V Custom Instruction Suggestion Generator.

Usage:
    # Analyze a built-in workload
    riscv-cisg analyze --workload transformer --d-model 768 --seq-len 128

    # Analyze a custom PyTorch model file
    riscv-cisg analyze --model-file my_model.py --class MyModel

    # List available workloads
    riscv-cisg list-workloads

    # Show version
    riscv-cisg --version
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python -m riscv_cisg.cli`
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run the CISG analysis pipeline."""
    from riscv_cisg.pipeline import CISGPipeline

    pipeline = CISGPipeline(
        output_dir=args.output_dir,
        top_n_hotspots=args.top_n,
        profile=not args.no_profile,
        speedup_target=args.speedup_target,
        hw_peak_flops=args.hw_peak_flops,
        hw_peak_bandwidth_gbs=args.hw_peak_bandwidth,
        spike_binary=args.spike_binary,
        verbose=not args.quiet,
    )

    if args.workload:
        results = _run_builtin_workload(args, pipeline)
    elif args.model_file:
        results = _run_model_file(args, pipeline)
    else:
        print("Error: provide --workload or --model-file", file=sys.stderr)
        return 1

    if results is None:
        return 1

    print(results.summary())
    print(f"\n✓ Analysis complete.")
    print(f"  Report:   {results.output_dir}/reports/analysis_report.md")
    print(f"  TableGen: {results.output_dir}/tablegen/")
    print(f"  Spike:    {results.output_dir}/spike_extension/")
    return 0


def _run_builtin_workload(args, pipeline):
    """Run one of the built-in named workloads."""
    workload = args.workload.lower()

    if workload == "transformer":
        from examples.transformer_example import build_transformer_graph
        d_model = getattr(args, "d_model", 768)
        n_heads = getattr(args, "n_heads", 12)
        seq_len = getattr(args, "seq_len", 128)
        print(f"Building Transformer graph: d_model={d_model}, n_heads={n_heads}, seq_len={seq_len}")
        graph = build_transformer_graph(d_model=d_model, seq_len=seq_len, n_heads=n_heads)
        return pipeline.run_from_graph(
            graph,
            workload_description=f"TransformerEncoderLayer d_model={d_model} seq_len={seq_len}",
        )

    elif workload == "attention":
        from riscv_cisg.analyzer.op_graph import OpGraph, OpNode, OpType, TensorShape, DataType
        B, H, S, D = 1, 8, 128, 64
        graph = OpGraph(name="ScaledDotProductAttention")
        fp32 = DataType.FP32
        def Sh(*dims): return TensorShape(dims=dims, dtype=fp32)
        graph.add_node(OpNode(
            node_id="sdpa_0",
            op_type=OpType.SCALED_DOT_PRODUCT_ATTENTION,
            input_shapes=[Sh(B, H, S, D), Sh(B, H, S, D), Sh(B, H, S, D)],
            output_shapes=[Sh(B, H, S, D)],
            flops=2 * B * H * S * S * D + 5 * B * H * S * S + 2 * B * H * S * D * S,
            memory_bytes=(3 * B * H * S * D + B * H * S * D) * 4,
            profiled_time_us=8000.0,
        ))
        return pipeline.run_from_graph(graph, workload_description="Scaled dot-product attention")

    elif workload == "ffn":
        from riscv_cisg.analyzer.op_graph import OpGraph, OpNode, OpType, TensorShape, DataType
        B, S, D = 1, 128, 768
        FFN = D * 4
        graph = OpGraph(name="FeedForwardNetwork")
        fp32 = DataType.FP32
        def Sh(*dims): return TensorShape(dims=dims, dtype=fp32)
        graph.add_node(OpNode(
            node_id="ffn_linear1",
            op_type=OpType.MATMUL,
            input_shapes=[Sh(B, S, D), Sh(D, FFN)],
            output_shapes=[Sh(B, S, FFN)],
            flops=2 * B * S * D * FFN,
            memory_bytes=(B*S*D + D*FFN + B*S*FFN) * 4,
            profiled_time_us=5000.0,
        ))
        graph.add_node(OpNode(
            node_id="ffn_gelu",
            op_type=OpType.GELU,
            input_shapes=[Sh(B, S, FFN)],
            output_shapes=[Sh(B, S, FFN)],
            flops=14 * B * S * FFN,
            memory_bytes=2 * B * S * FFN * 4,
            profiled_time_us=600.0,
        ))
        graph.add_node(OpNode(
            node_id="ffn_linear2",
            op_type=OpType.MATMUL,
            input_shapes=[Sh(B, S, FFN), Sh(FFN, D)],
            output_shapes=[Sh(B, S, D)],
            flops=2 * B * S * FFN * D,
            memory_bytes=(B*S*FFN + FFN*D + B*S*D) * 4,
            profiled_time_us=4800.0,
        ))
        return pipeline.run_from_graph(graph, workload_description="FFN: Linear+GELU+Linear")

    else:
        print(f"Unknown workload: '{workload}'. Use --list-workloads to see options.", file=sys.stderr)
        return None


def _run_model_file(args, pipeline):
    """Dynamically load a model from a Python file and run analysis."""
    import importlib.util
    import torch

    model_path = Path(args.model_file)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        return None

    spec = importlib.util.spec_from_file_location("user_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_name = args.model_class or "Model"
    if not hasattr(module, class_name):
        print(f"Error: class '{class_name}' not found in {model_path}", file=sys.stderr)
        print(f"Available names: {[n for n in dir(module) if not n.startswith('_')]}")
        return None

    ModelClass = getattr(module, class_name)
    model = ModelClass()

    # Build example inputs from --input-shape arguments
    # e.g. --input-shape 1,128,768
    input_shapes = getattr(args, "input_shape", None) or ["1,128,768"]
    example_inputs = []
    for shape_str in input_shapes:
        dims = tuple(int(d) for d in shape_str.split(","))
        example_inputs.append(torch.randn(*dims))

    return pipeline.run(
        model=model,
        example_inputs=tuple(example_inputs),
        workload_description=f"User model: {class_name} from {model_path.name}",
    )


def cmd_list_workloads(_args: argparse.Namespace) -> int:
    """List available built-in workloads."""
    workloads = {
        "transformer": "Full Transformer encoder layer (MHA + FFN + LayerNorm)",
        "attention":   "Scaled dot-product attention (QKV only)",
        "ffn":         "Feed-forward network (Linear + GELU + Linear)",
    }
    print("\nBuilt-in workloads:\n")
    for name, desc in workloads.items():
        print(f"  {name:<16} {desc}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riscv-cisg",
        description="RISC-V Custom Instruction Suggestion Generator for ML Workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  riscv-cisg analyze --workload transformer --d-model 768 --seq-len 128
  riscv-cisg analyze --workload attention --output-dir ./my_output
  riscv-cisg analyze --workload ffn --top-n 3 --no-profile
  riscv-cisg analyze --model-file my_model.py --model-class MyTransformer
  riscv-cisg list-workloads
        """,
    )
    parser.add_argument("--version", action="version", version="riscv-cisg 1.0.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze ──────────────────────────────────────────────────────────────
    analyze_p = subparsers.add_parser("analyze", help="Run the CISG analysis pipeline")

    # Workload source
    src = analyze_p.add_mutually_exclusive_group()
    src.add_argument("--workload", "-w", metavar="NAME",
        help="Built-in workload name (transformer | attention | ffn)")
    src.add_argument("--model-file", metavar="PATH",
        help="Path to a Python file containing a PyTorch nn.Module class")

    # Model-file options
    analyze_p.add_argument("--model-class", metavar="CLASS", default="Model",
        help="Class name to instantiate from --model-file (default: Model)")
    analyze_p.add_argument("--input-shape", metavar="DIM,...", action="append",
        help="Input tensor shape, e.g. 1,128,768 (can be repeated)")

    # Transformer-specific
    analyze_p.add_argument("--d-model",  type=int, default=768,  help="Model dimension (default: 768)")
    analyze_p.add_argument("--n-heads",  type=int, default=12,   help="Attention heads (default: 12)")
    analyze_p.add_argument("--seq-len",  type=int, default=128,  help="Sequence length (default: 128)")

    # Pipeline options
    analyze_p.add_argument("--top-n",    type=int,   default=5,    help="Top-N hotspots (default: 5)")
    analyze_p.add_argument("--output-dir", "-o", default="./cisg_output", help="Output directory")
    analyze_p.add_argument("--no-profile", action="store_true", help="Skip torch.profiler")
    analyze_p.add_argument("--speedup-target", type=float, default=10.0, help="Target speedup (default: 10)")
    analyze_p.add_argument("--hw-peak-flops",  type=float, default=10.0, help="HW peak GFLOPs/s (default: 10)")
    analyze_p.add_argument("--hw-peak-bandwidth", type=float, default=8.0, help="HW peak GB/s (default: 8)")
    analyze_p.add_argument("--spike-binary", metavar="PATH", default=None, help="Path to Spike binary")
    analyze_p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress logs")

    # ── list-workloads ────────────────────────────────────────────────────────
    subparsers.add_parser("list-workloads", help="List available built-in workloads")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "list-workloads":
        return cmd_list_workloads(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
