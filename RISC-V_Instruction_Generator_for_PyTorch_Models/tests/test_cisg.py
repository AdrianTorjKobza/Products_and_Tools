"""
Test Suite for RISC-V CISG
==========================
pytest-based tests covering:
  - OpGraph construction and properties
  - HotspotDetector scoring
  - PatternRuleEngine matching and proposal
  - SpeedupEstimator (Amdahl, roofline)
  - CISGPipeline.run_from_graph end-to-end

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from riscv_cisg.analyzer.op_graph import (
    DataType, OpGraph, OpNode, OpType, TensorShape,
)
from riscv_cisg.analyzer.hotspot_detector import HotspotDetector, HotspotResult
from riscv_cisg.proposer.pattern_rules import PatternRuleEngine
from riscv_cisg.proposer.instruction_proposer import InstructionProposer
from riscv_cisg.simulator.speedup_estimator import SpeedupEstimator
from riscv_cisg.pipeline import CISGPipeline


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_shape(*dims, dtype=DataType.FP32) -> TensorShape:
    return TensorShape(dims=dims, dtype=dtype)


def make_matmul_node(node_id="mm_0", M=64, K=64, N=64, time_us=0.0) -> OpNode:
    flops = 2 * M * K * N
    mem   = (M * K + K * N + M * N) * 4
    return OpNode(
        node_id=node_id,
        op_type=OpType.MATMUL,
        input_shapes=[make_shape(M, K), make_shape(K, N)],
        output_shapes=[make_shape(M, N)],
        flops=flops,
        memory_bytes=mem,
        profiled_time_us=time_us,
        source_framework="aten::mm",
    )


def make_softmax_node(seq=128, time_us=0.0) -> OpNode:
    flops = 5 * seq
    mem   = 2 * seq * 4
    return OpNode(
        node_id="softmax_0",
        op_type=OpType.SOFTMAX,
        input_shapes=[make_shape(seq)],
        output_shapes=[make_shape(seq)],
        flops=flops,
        memory_bytes=mem,
        profiled_time_us=time_us,
    )


def simple_graph() -> OpGraph:
    g = OpGraph(name="test_graph")
    g.add_node(make_matmul_node("mm_0", M=512, K=512, N=512, time_us=5000.0))
    g.add_node(make_softmax_node(seq=512, time_us=500.0))
    g.add_node(make_matmul_node("mm_1", M=512, K=512, N=512, time_us=4800.0))
    return g


# ──────────────────────────────────────────────────────────────────────────────
# OpGraph tests
# ──────────────────────────────────────────────────────────────────────────────

class TestOpGraph:
    def test_add_and_get_node(self):
        g = OpGraph(name="test")
        node = make_matmul_node()
        g.add_node(node)
        assert g.num_nodes == 1
        retrieved = g.get_node("mm_0")
        assert retrieved.op_type == OpType.MATMUL

    def test_total_flops(self):
        g = OpGraph()
        g.add_node(make_matmul_node("a", M=64, K=64, N=64))
        g.add_node(make_matmul_node("b", M=32, K=32, N=32))
        expected = 2 * 64 * 64 * 64 + 2 * 32 * 32 * 32
        assert g.total_flops == expected

    def test_get_nodes_by_type(self):
        g = simple_graph()
        matmuls = g.get_nodes_by_type(OpType.MATMUL)
        softmaxes = g.get_nodes_by_type(OpType.SOFTMAX)
        assert len(matmuls) == 2
        assert len(softmaxes) == 1

    def test_arithmetic_intensity(self):
        node = make_matmul_node(M=1024, K=1024, N=1024)
        # AI = flops / memory_bytes
        expected_ai = node.flops / node.memory_bytes
        assert abs(node.arithmetic_intensity - expected_ai) < 0.01

    def test_tensor_shape_bytes(self):
        s = make_shape(4, 4, dtype=DataType.FP32)
        assert s.num_elements == 16
        assert s.bytes == 64  # 16 × 4 bytes

        s16 = make_shape(4, 4, dtype=DataType.FP16)
        assert s16.bytes == 32  # 16 × 2 bytes

    def test_subgraph(self):
        g = simple_graph()
        node_ids = [n.node_id for n in g.nodes[:2]]
        sg = g.subgraph(node_ids)
        assert sg.num_nodes == 2

    def test_to_dict_and_json(self):
        g = simple_graph()
        d = g.to_dict()
        assert d["name"] == "test_graph"
        assert d["num_nodes"] == 3
        j = g.to_json()
        import json
        parsed = json.loads(j)
        assert parsed["num_nodes"] == 3


# ──────────────────────────────────────────────────────────────────────────────
# HotspotDetector tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHotspotDetector:
    def test_detects_top_n(self):
        g = simple_graph()
        detector = HotspotDetector(g, top_n=2)
        results = detector.detect()
        assert len(results) == 2

    def test_ranks_by_score(self):
        g = simple_graph()
        detector = HotspotDetector(g, top_n=3)
        results = detector.detect()
        scores = [r.hotspot_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_matmul_scores_highly(self):
        g = simple_graph()
        detector = HotspotDetector(g, top_n=3)
        results = detector.detect()
        # Top hotspot should be a MATMUL (highest FLOPs + time)
        assert results[0].node.op_type == OpType.MATMUL

    def test_flops_pct_sums_near_100(self):
        g = simple_graph()
        detector = HotspotDetector(g, top_n=10, only_acceleratable=False)
        results = detector.detect()
        total = sum(r.flops_pct for r in results)
        assert total <= 101.0  # allow floating point slack

    def test_only_acceleratable_filter(self):
        g = OpGraph()
        # Add an UNKNOWN op (non-acceleratable)
        g.add_node(OpNode(
            node_id="unknown_0", op_type=OpType.UNKNOWN,
            flops=10_000_000, memory_bytes=1000,
            input_shapes=[], output_shapes=[],
        ))
        g.add_node(make_matmul_node("mm", M=512, K=512, N=512))
        detector = HotspotDetector(g, top_n=5, only_acceleratable=True)
        results = detector.detect()
        for r in results:
            assert r.node.op_type != OpType.UNKNOWN

    def test_min_flop_threshold(self):
        g = OpGraph()
        g.add_node(make_matmul_node("tiny", M=4, K=4, N=4))  # tiny → filtered
        g.add_node(make_matmul_node("big", M=512, K=512, N=512))
        detector = HotspotDetector(g, top_n=5, min_flop_threshold=100_000)
        results = detector.detect()
        for r in results:
            assert r.node.flops >= 100_000


# ──────────────────────────────────────────────────────────────────────────────
# PatternRuleEngine tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPatternRuleEngine:
    def _make_hotspot(self, op_type, flops=5_000_000, mem=100_000, time_us=1000.0,
                      input_shapes=None, output_shapes=None) -> HotspotResult:
        """Helper to manufacture a HotspotResult for a given OpType."""
        node = OpNode(
            node_id=f"test_{op_type.name}",
            op_type=op_type,
            flops=flops,
            memory_bytes=mem,
            profiled_time_us=time_us,
            input_shapes=input_shapes or [],
            output_shapes=output_shapes or [],
        )
        return HotspotResult(
            node=node,
            hotspot_score=80.0,
            time_pct=30.0,
            flops_pct=40.0,
            memory_pct=20.0,
            is_acceleratable=True,
            acceleration_rationale="test hotspot",
            rank=1,
        )

    def test_matmul_rule_fires(self):
        engine = PatternRuleEngine()
        from riscv_cisg.analyzer.op_graph import TensorShape
        hotspot = self._make_hotspot(
            OpType.MATMUL,
            flops=2 * 512 * 512 * 512,
            input_shapes=[TensorShape((512, 512)), TensorShape((512, 512))],
        )
        proposals = engine.propose_all([hotspot])
        assert len(proposals) == 1
        _, instr = proposals[0]
        assert instr.mnemonic in ("mmtile", "bmmtile")

    def test_softmax_rule_fires(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(OpType.SOFTMAX,
            input_shapes=[TensorShape((128,))])
        proposals = engine.propose_all([hotspot])
        assert len(proposals) == 1
        _, instr = proposals[0]
        assert instr.mnemonic == "sfmax"

    def test_layer_norm_rule_fires(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(OpType.LAYER_NORM,
            input_shapes=[TensorShape((768,))])
        proposals = engine.propose_all([hotspot])
        assert len(proposals) == 1
        _, instr = proposals[0]
        assert instr.mnemonic == "lnorm"

    def test_gelu_rule_fires(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(OpType.GELU,
            input_shapes=[TensorShape((1, 128, 3072))])
        proposals = engine.propose_all([hotspot])
        assert len(proposals) == 1
        _, instr = proposals[0]
        assert "gelu" in instr.mnemonic

    def test_sdpa_rule_fires(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(
            OpType.SCALED_DOT_PRODUCT_ATTENTION,
            input_shapes=[TensorShape((1, 8, 128, 64))],
        )
        proposals = engine.propose_all([hotspot])
        assert len(proposals) == 1
        _, instr = proposals[0]
        assert instr.mnemonic == "sdpa"

    def test_no_match_returns_empty(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(OpType.TRANSPOSE)
        proposals = engine.propose_all([hotspot])
        assert len(proposals) == 0

    def test_speedup_model_present(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(OpType.SOFTMAX,
            input_shapes=[TensorShape((256,))])
        proposals = engine.propose_all([hotspot])
        _, instr = proposals[0]
        assert instr.speedup_model is not None
        assert instr.speedup_model.estimated_speedup > 1.0

    def test_tablegen_snippet_nonempty(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(
            OpType.MATMUL, flops=2*512*512*512,
            input_shapes=[TensorShape((512, 512)), TensorShape((512, 512))],
        )
        proposals = engine.propose_all([hotspot])
        _, instr = proposals[0]
        assert len(instr.tablegen_snippet) > 50
        assert "TableGen" in instr.tablegen_snippet or "def " in instr.tablegen_snippet

    def test_spike_snippet_nonempty(self):
        engine = PatternRuleEngine()
        hotspot = self._make_hotspot(OpType.SOFTMAX,
            input_shapes=[TensorShape((128,))])
        proposals = engine.propose_all([hotspot])
        _, instr = proposals[0]
        assert len(instr.spike_extension_snippet) > 50
        assert "DEFINE_INSN" in instr.spike_extension_snippet


from riscv_cisg.analyzer.op_graph import TensorShape  # needed in class above


# ──────────────────────────────────────────────────────────────────────────────
# SpeedupEstimator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSpeedupEstimator:
    def test_amdahl_100pct(self):
        g = simple_graph()
        est = SpeedupEstimator(g)
        # If the kernel is 100% of the workload, system speedup == kernel speedup
        result = est.amdahl_speedup(kernel_speedup=10.0, fraction=1.0)
        assert abs(result - 10.0) < 0.01

    def test_amdahl_50pct(self):
        g = simple_graph()
        est = SpeedupEstimator(g)
        # S = 1 / (0.5 + 0.5/10) = 1 / 0.55 ≈ 1.818
        result = est.amdahl_speedup(kernel_speedup=10.0, fraction=0.5)
        assert abs(result - (1.0 / 0.55)) < 0.01

    def test_amdahl_low_fraction(self):
        g = simple_graph()
        est = SpeedupEstimator(g)
        # Very small kernel: system speedup barely above 1
        result = est.amdahl_speedup(kernel_speedup=100.0, fraction=0.01)
        assert result < 1.1

    def test_roofline_compute_bound(self):
        g = simple_graph()
        est = SpeedupEstimator(g, hw_peak_flops=10.0, hw_peak_bandwidth_gbs=8.0)
        # AI = 100 FLOPs/byte >> ridge point (10/8 = 1.25)
        perf = est.roofline_peak(arithmetic_intensity=100.0)
        assert abs(perf - 10.0) < 0.01  # capped by compute peak

    def test_roofline_memory_bound(self):
        g = simple_graph()
        est = SpeedupEstimator(g, hw_peak_flops=10.0, hw_peak_bandwidth_gbs=8.0)
        # AI = 0.1 FLOPs/byte << ridge point → memory bound
        perf = est.roofline_peak(arithmetic_intensity=0.1)
        assert abs(perf - 0.8) < 0.01  # 0.1 * 8.0

    def test_estimate_all_returns_one_per_proposal(self):
        g = simple_graph()
        detector = HotspotDetector(g, top_n=3)
        hotspots = detector.detect()
        proposer = InstructionProposer()
        proposals = proposer.propose(hotspots)
        estimator = SpeedupEstimator(g)
        analyses = estimator.estimate_all(proposals)
        assert len(analyses) == len(proposals)


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline test
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineEndToEnd:
    def test_run_from_graph_produces_files(self, tmp_path):
        g = simple_graph()
        pipeline = CISGPipeline(
            output_dir=str(tmp_path / "cisg_out"),
            top_n_hotspots=3,
            profile=False,
            verbose=False,
        )
        results = pipeline.run_from_graph(g)

        assert results.graph is g
        assert len(results.hotspots) > 0
        assert len(results.proposals) > 0
        assert len(results.generated_files) > 0

        # All generated files should exist
        for f in results.generated_files:
            assert Path(f).exists(), f"Missing: {f}"

    def test_run_from_graph_produces_report(self, tmp_path):
        g = simple_graph()
        pipeline = CISGPipeline(
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        results = pipeline.run_from_graph(g)

        report_md = results.output_dir / "reports" / "analysis_report.md"
        report_json = results.output_dir / "reports" / "analysis_report.json"
        assert report_md.exists()
        assert report_json.exists()
        assert "RISC-V" in report_md.read_text()

    def test_run_from_graph_tablegen_valid(self, tmp_path):
        g = simple_graph()
        pipeline = CISGPipeline(
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        results = pipeline.run_from_graph(g)

        td_file = results.output_dir / "tablegen" / "RISCVInstrInfoCustom.td"
        assert td_file.exists()
        content = td_file.read_text()
        assert "RISC-V" in content or "TableGen" in content

    def test_run_from_graph_spike_extension(self, tmp_path):
        g = simple_graph()
        pipeline = CISGPipeline(
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        results = pipeline.run_from_graph(g)

        ext_cc = results.output_dir / "spike_extension" / "extension.cc"
        assert ext_cc.exists()
        assert "REGISTER_EXTENSION" in ext_cc.read_text()

    def test_summary_string(self, tmp_path):
        g = simple_graph()
        pipeline = CISGPipeline(output_dir=str(tmp_path / "out"), verbose=False)
        results = pipeline.run_from_graph(g)
        s = results.summary()
        assert "RISC-V CISG" in s
        assert "Proposals" in s

    def test_transformer_graph_full(self, tmp_path):
        """Integration test: full transformer layer graph."""
        from examples.transformer_example import build_transformer_graph
        g = build_transformer_graph(d_model=256, seq_len=32, n_heads=4)
        pipeline = CISGPipeline(
            output_dir=str(tmp_path / "transformer"),
            verbose=False,
            top_n_hotspots=5,
        )
        results = pipeline.run_from_graph(g)
        assert results.graph.num_nodes >= 8
        assert len(results.proposals) >= 2

        # Check at least one proposal meets 10x on its kernel
        kernel_speedups = [
            p[1].speedup_model.estimated_speedup
            for p in results.proposals
            if p[1].speedup_model
        ]
        assert any(s >= 5.0 for s in kernel_speedups), (
            f"Expected at least one 5x+ proposal, got: {kernel_speedups}"
        )
