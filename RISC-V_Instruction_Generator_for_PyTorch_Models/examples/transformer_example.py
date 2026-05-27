"""
Example: Analyze a Transformer Encoder Layer
=============================================
Demonstrates the full CISG pipeline on a standard PyTorch
TransformerEncoderLayer — the canonical use case.

Run:
    python examples/transformer_example.py

Or with options:
    python examples/transformer_example.py --d-model 512 --seq-len 64 --no-profile
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from riscv_cisg import CISGPipeline
from riscv_cisg.analyzer.op_graph import OpGraph, OpNode, OpType, TensorShape, DataType
from riscv_cisg.analyzer.hotspot_detector import detect_hotspots_from_graph


def build_transformer_graph(d_model: int, seq_len: int, n_heads: int) -> OpGraph:
    """
    Manually construct a representative OpGraph for a Transformer encoder layer.

    This bypasses the FX tracer and gives deterministic, shape-correct results
    for demonstration purposes. In production, WorkloadAnalyzer handles this.

    Architecture:
        MultiHeadAttention:
          Q, K, V projections  →  3x MATMUL
          QK^T attention       →  BATCH_MATMUL
          Softmax              →  SOFTMAX
          Attention × V        →  BATCH_MATMUL
          Output projection    →  MATMUL
        Add + LayerNorm        →  LAYER_NORM
        FFN:
          Linear 1 (4x expand)  →  MATMUL
          GELU                  →  GELU
          Linear 2 (project)    →  MATMUL
        Add + LayerNorm         →  LAYER_NORM
    """
    graph = OpGraph(name=f"TransformerEncoderLayer_d{d_model}_s{seq_len}")

    fp32 = DataType.FP32
    head_dim = d_model // n_heads

    def S(*dims): return TensorShape(dims=dims, dtype=fp32)

    def add(node_id, op_type, in_shapes, out_shapes, flops, mem_bytes, time_us=0.0, src=""):
        node = OpNode(
            node_id=node_id, op_type=op_type,
            input_shapes=in_shapes, output_shapes=out_shapes,
            flops=flops, memory_bytes=mem_bytes,
            profiled_time_us=time_us, source_framework=src,
        )
        graph.add_node(node)

    B = 1  # batch size

    # ── MultiHead Attention ──────────────────────────────────────────────
    # Q, K, V projections: (B, S, D) × (D, D) → (B, S, D)
    proj_flops = 2 * B * seq_len * d_model * d_model
    proj_mem   = (B * seq_len * d_model + d_model * d_model + B * seq_len * d_model) * 4

    add("mha_q_proj", OpType.MATMUL,
        [S(B, seq_len, d_model), S(d_model, d_model)], [S(B, seq_len, d_model)],
        proj_flops, proj_mem, time_us=1200.0, src="aten::linear")
    add("mha_k_proj", OpType.MATMUL,
        [S(B, seq_len, d_model), S(d_model, d_model)], [S(B, seq_len, d_model)],
        proj_flops, proj_mem, time_us=1200.0, src="aten::linear")
    add("mha_v_proj", OpType.MATMUL,
        [S(B, seq_len, d_model), S(d_model, d_model)], [S(B, seq_len, d_model)],
        proj_flops, proj_mem, time_us=1200.0, src="aten::linear")

    # QK^T: (B, H, S, D_h) × (B, H, D_h, S) → (B, H, S, S)
    qkt_flops = 2 * B * n_heads * seq_len * seq_len * head_dim
    qkt_mem   = (2 * B * n_heads * seq_len * head_dim + B * n_heads * seq_len * seq_len) * 4

    add("mha_qkt", OpType.BATCH_MATMUL,
        [S(B, n_heads, seq_len, head_dim), S(B, n_heads, head_dim, seq_len)],
        [S(B, n_heads, seq_len, seq_len)],
        qkt_flops, qkt_mem, time_us=800.0, src="aten::bmm")

    # Softmax: (B, H, S, S)
    softmax_flops = 5 * B * n_heads * seq_len * seq_len
    softmax_mem   = 2 * B * n_heads * seq_len * seq_len * 4

    add("mha_softmax", OpType.SOFTMAX,
        [S(B, n_heads, seq_len, seq_len)], [S(B, n_heads, seq_len, seq_len)],
        softmax_flops, softmax_mem, time_us=400.0, src="aten::softmax")

    # Attention × V: (B, H, S, S) × (B, H, S, D_h) → (B, H, S, D_h)
    av_flops = 2 * B * n_heads * seq_len * seq_len * head_dim
    av_mem   = (B * n_heads * seq_len * seq_len + B * n_heads * seq_len * head_dim) * 4 * 2

    add("mha_av", OpType.BATCH_MATMUL,
        [S(B, n_heads, seq_len, seq_len), S(B, n_heads, seq_len, head_dim)],
        [S(B, n_heads, seq_len, head_dim)],
        av_flops, av_mem, time_us=750.0, src="aten::bmm")

    # Output projection
    add("mha_out_proj", OpType.MATMUL,
        [S(B, seq_len, d_model), S(d_model, d_model)], [S(B, seq_len, d_model)],
        proj_flops, proj_mem, time_us=1100.0, src="aten::linear")

    # LayerNorm 1
    ln_flops = 10 * B * seq_len * d_model
    ln_mem   = 3 * B * seq_len * d_model * 4
    add("ln1", OpType.LAYER_NORM,
        [S(B, seq_len, d_model)], [S(B, seq_len, d_model)],
        ln_flops, ln_mem, time_us=150.0, src="aten::layer_norm")

    # ── Feed-Forward Network ─────────────────────────────────────────────
    ffn_dim = d_model * 4

    # FFN Linear 1: (B, S, D) × (D, 4D) → (B, S, 4D)
    ffn1_flops = 2 * B * seq_len * d_model * ffn_dim
    ffn1_mem   = (B * seq_len * d_model + d_model * ffn_dim + B * seq_len * ffn_dim) * 4
    add("ffn_linear1", OpType.MATMUL,
        [S(B, seq_len, d_model), S(d_model, ffn_dim)], [S(B, seq_len, ffn_dim)],
        ffn1_flops, ffn1_mem, time_us=4800.0, src="aten::linear")

    # GELU: (B, S, 4D)
    gelu_flops = 14 * B * seq_len * ffn_dim
    gelu_mem   = 2 * B * seq_len * ffn_dim * 4
    add("ffn_gelu", OpType.GELU,
        [S(B, seq_len, ffn_dim)], [S(B, seq_len, ffn_dim)],
        gelu_flops, gelu_mem, time_us=600.0, src="aten::gelu")

    # FFN Linear 2: (B, S, 4D) × (4D, D) → (B, S, D)
    ffn2_flops = 2 * B * seq_len * ffn_dim * d_model
    ffn2_mem   = (B * seq_len * ffn_dim + ffn_dim * d_model + B * seq_len * d_model) * 4
    add("ffn_linear2", OpType.MATMUL,
        [S(B, seq_len, ffn_dim), S(ffn_dim, d_model)], [S(B, seq_len, d_model)],
        ffn2_flops, ffn2_mem, time_us=4600.0, src="aten::linear")

    # LayerNorm 2
    add("ln2", OpType.LAYER_NORM,
        [S(B, seq_len, d_model)], [S(B, seq_len, d_model)],
        ln_flops, ln_mem, time_us=150.0, src="aten::layer_norm")

    return graph


def main():
    parser = argparse.ArgumentParser(
        description="CISG: Analyze a Transformer layer and propose custom RISC-V instructions"
    )
    parser.add_argument("--d-model",    type=int, default=768,  help="Model dimension (default: 768)")
    parser.add_argument("--n-heads",    type=int, default=12,   help="Number of attention heads (default: 12)")
    parser.add_argument("--seq-len",    type=int, default=128,  help="Sequence length (default: 128)")
    parser.add_argument("--top-n",      type=int, default=5,    help="Number of hotspots (default: 5)")
    parser.add_argument("--output-dir", type=str, default="./cisg_output", help="Output directory")
    parser.add_argument("--no-profile", action="store_true",    help="Skip torch profiler")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  RISC-V CISG — Transformer Analysis")
    print(f"{'='*60}")
    print(f"  d_model = {args.d_model}")
    print(f"  n_heads = {args.n_heads}")
    print(f"  seq_len = {args.seq_len}")
    print(f"  top_n   = {args.top_n}")
    print(f"  output  = {args.output_dir}")
    print(f"{'='*60}\n")

    # Build the graph directly (no FX tracing needed for the demo)
    graph = build_transformer_graph(
        d_model=args.d_model,
        seq_len=args.seq_len,
        n_heads=args.n_heads,
    )

    print(f"Graph: {graph}")
    print(f"Total FLOPs: {graph.total_flops:,}")
    print(f"Total memory: {graph.total_memory_bytes / 1e6:.1f} MB")
    print(f"Total profiled time: {graph.total_profiled_time_us:.0f} μs\n")

    # Run pipeline from pre-built graph
    pipeline = CISGPipeline(
        output_dir=args.output_dir,
        top_n_hotspots=args.top_n,
        profile=False,  # graph already built
        verbose=True,
    )
    results = pipeline.run_from_graph(
        graph,
        workload_description=(
            f"Single Transformer encoder layer. "
            f"d_model={args.d_model}, n_heads={args.n_heads}, seq_len={args.seq_len}. "
            f"Includes MHA (Q/K/V projections, attention, output proj) and FFN (2×linear + GELU)."
        ),
    )

    print(results.summary())
    print(f"\nMarkdown report: {results.output_dir}/reports/analysis_report.md")
    print(f"JSON report:     {results.output_dir}/reports/analysis_report.json")
    print(f"TableGen files:  {results.output_dir}/tablegen/")
    print(f"Spike plugin:    {results.output_dir}/spike_extension/\n")


if __name__ == "__main__":
    main()
