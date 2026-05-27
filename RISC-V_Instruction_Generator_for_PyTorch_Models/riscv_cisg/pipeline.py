"""
CISGPipeline
============
The main orchestration class that wires together all CISG stages:

  1. WorkloadAnalyzer  → OpGraph
  2. HotspotDetector   → List[HotspotResult]
  3. InstructionProposer → List[(HotspotResult, CustomInstruction)]
  4. SpeedupEstimator  → List[SpeedupAnalysis]
  5. TableGenEmitter   → LLVM .td files
  6. SpikeExtensionEmitter → Spike plugin files
  7. ReportGenerator   → Markdown + JSON reports

Usage:
    from riscv_cisg import CISGPipeline
    import torch
    import torch.nn as nn

    model = nn.TransformerEncoderLayer(d_model=768, nhead=12)
    inputs = (torch.randn(1, 128, 768),)

    pipeline = CISGPipeline(output_dir="./cisg_output")
    results  = pipeline.run(model, inputs)
    print(results.summary())
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from riscv_cisg.analyzer.hotspot_detector import HotspotResult, detect_hotspots_from_graph
from riscv_cisg.analyzer.op_graph import OpGraph
from riscv_cisg.analyzer.workload_analyzer import WorkloadAnalyzer
from riscv_cisg.backend.spike_emitter import SpikeExtensionEmitter
from riscv_cisg.backend.tablegen_emitter import TableGenEmitter
from riscv_cisg.proposer.custom_instruction import CustomInstruction
from riscv_cisg.proposer.instruction_proposer import InstructionProposer
from riscv_cisg.proposer.pattern_rules import PatternRule
from riscv_cisg.reporter.report_generator import ReportGenerator
from riscv_cisg.simulator.speedup_estimator import SpeedupAnalysis, SpeedupEstimator

logger = logging.getLogger(__name__)


@dataclass
class CISGResults:
    """
    Container for all outputs of a CISGPipeline.run() call.

    Attributes
    ----------
    graph : OpGraph
    hotspots : list of HotspotResult
    proposals : list of (HotspotResult, CustomInstruction)
    speedup_analyses : list of SpeedupAnalysis
    generated_files : list of Path
    output_dir : Path
    """
    graph: OpGraph
    hotspots: List[HotspotResult]
    proposals: List[Tuple[HotspotResult, CustomInstruction]]
    speedup_analyses: List[SpeedupAnalysis]
    generated_files: List[Path]
    output_dir: Path

    def summary(self) -> str:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            "║         RISC-V CISG — Analysis Complete                         ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            f"║  Workload   : {self.graph.name:<51}║",
            f"║  Nodes      : {self.graph.num_nodes:<51}║",
            f"║  Total FLOPs: {str(self.graph.total_flops):<51}║",
            f"║  Hotspots   : {len(self.hotspots):<51}║",
            f"║  Proposals  : {len(self.proposals):<51}║",
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  INSTRUCTION PROPOSALS                                           ║",
        ]

        for hotspot, instr in self.proposals:
            speedup = (
                instr.speedup_model.estimated_speedup if instr.speedup_model else 0.0
            )
            flag = "✓" if speedup >= 10.0 else "~"
            line = f"  {flag} {instr.mnemonic:<16} → {instr.target_op_type:<24} {speedup:.1f}x"
            lines.append(f"║  {line:<66}║")

        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  SYSTEM-LEVEL SPEEDUP (Amdahl's Law)                            ║",
        ]
        for sa in self.speedup_analyses:
            line = (
                f"  {sa.mnemonic:<16} kernel={sa.kernel_speedup:.1f}x  "
                f"system={sa.system_speedup:.2f}x"
            )
            lines.append(f"║  {line:<66}║")

        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  GENERATED FILES                                                 ║",
        ]
        seen_files = []
        seen_names = set()
        for f in self.generated_files:
            short = str(f).replace(str(self.output_dir) + "/", "")
            if short not in seen_names:
                seen_names.add(short)
                seen_files.append(short)
        for short in seen_files[:8]:
            lines.append(f"║    {short:<64}║")
        if len(seen_files) > 8:
            lines.append(f"║    ... and {len(seen_files)-8} more files{' '*48}║")

        lines += [
            f"║  Output dir : {str(self.output_dir):<51}║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
        ]
        return "\n".join(lines)


class CISGPipeline:
    """
    End-to-end pipeline: ML model → custom RISC-V instruction proposals.

    Parameters
    ----------
    output_dir : str or Path
        Root directory for all generated artifacts.
    top_n_hotspots : int
        Number of hotspots to analyze and propose instructions for.
    profile : bool
        Whether to run torch.profiler for actual execution timings.
        Set False for faster static analysis.
    speedup_target : float
        Minimum kernel speedup target. Used for flagging in reports.
    hw_peak_flops : float
        Assumed peak GFLOPs/s of the target RISC-V core.
    hw_peak_bandwidth_gbs : float
        Assumed peak memory bandwidth in GB/s.
    extra_rules : list of PatternRule, optional
        Additional custom pattern-matching rules.
    spike_binary : str, optional
        Path to Spike binary for empirical validation.
    verbose : bool
        Enable INFO-level logging.
    """

    def __init__(
        self,
        output_dir: str = "./cisg_output",
        top_n_hotspots: int = 5,
        profile: bool = True,
        speedup_target: float = 10.0,
        hw_peak_flops: float = 10.0,
        hw_peak_bandwidth_gbs: float = 8.0,
        extra_rules: Optional[List[PatternRule]] = None,
        spike_binary: Optional[str] = None,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.top_n_hotspots = top_n_hotspots
        self.profile = profile
        self.speedup_target = speedup_target
        self.hw_peak_flops = hw_peak_flops
        self.hw_peak_bandwidth_gbs = hw_peak_bandwidth_gbs
        self.extra_rules = extra_rules or []
        self.spike_binary = spike_binary

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )

    def run(
        self,
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        workload_description: str = "",
        device: str = "cpu",
    ) -> CISGResults:
        """
        Run the full CISG pipeline.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to analyze.
        example_inputs : tuple of Tensor
            Representative inputs (used for shape propagation and profiling).
        workload_description : str
            Optional human-readable description included in the report.
        device : str
            "cpu" or "cuda".

        Returns
        -------
        CISGResults
        """
        generated_files: List[Path] = []
        workload_name = type(model).__name__

        logger.info("═" * 60)
        logger.info("  RISC-V CISG Pipeline starting")
        logger.info("  Model: %s  Device: %s", workload_name, device)
        logger.info("═" * 60)

        # ── Stage 1: Workload Analysis ─────────────────────────────────
        logger.info("[1/6] Workload analysis...")
        analyzer = WorkloadAnalyzer(
            model=model,
            example_inputs=example_inputs,
            device=device,
            profile=self.profile,
        )
        graph = analyzer.analyze()

        # ── Stage 2: Hotspot Detection ─────────────────────────────────
        logger.info("[2/6] Hotspot detection...")
        hotspots = detect_hotspots_from_graph(
            graph, top_n=self.top_n_hotspots, only_acceleratable=True
        )

        if not hotspots:
            logger.warning("No hotspots detected — trying without acceleratable filter.")
            hotspots = detect_hotspots_from_graph(
                graph, top_n=self.top_n_hotspots, only_acceleratable=False
            )

        # ── Stage 3: Instruction Proposal ─────────────────────────────
        logger.info("[3/6] Generating custom instruction proposals...")
        proposer = InstructionProposer(
            extra_rules=self.extra_rules,
            speedup_target=self.speedup_target,
        )
        proposals = proposer.propose(hotspots)

        # ── Stage 4: Speedup Estimation ───────────────────────────────
        logger.info("[4/6] Speedup estimation (Roofline + Amdahl)...")
        estimator = SpeedupEstimator(
            graph=graph,
            hw_peak_flops=self.hw_peak_flops,
            hw_peak_bandwidth_gbs=self.hw_peak_bandwidth_gbs,
            spike_binary=self.spike_binary,
        )
        speedup_analyses = estimator.estimate_all(proposals)

        # ── Stage 5: Backend Emission ──────────────────────────────────
        logger.info("[5/6] Emitting LLVM TableGen + Spike extension files...")

        tablegen_emitter = TableGenEmitter(
            output_dir=str(self.output_dir / "tablegen"),
            workload_name=workload_name,
        )
        generated_files.extend(tablegen_emitter.emit(proposals))

        spike_emitter = SpikeExtensionEmitter(
            output_dir=str(self.output_dir / "spike_extension"),
            workload_name=workload_name,
        )
        generated_files.extend(spike_emitter.emit(proposals))

        # ── Stage 6: Report Generation ─────────────────────────────────
        logger.info("[6/6] Generating analysis report...")
        reporter = ReportGenerator(
            output_dir=str(self.output_dir / "reports")
        )
        md_path, json_path = reporter.generate(
            graph=graph,
            hotspots=hotspots,
            proposals=proposals,
            workload_description=workload_description,
        )
        generated_files.extend([md_path, json_path])

        logger.info("Pipeline complete. Output: %s", self.output_dir)

        return CISGResults(
            graph=graph,
            hotspots=hotspots,
            proposals=proposals,
            speedup_analyses=speedup_analyses,
            generated_files=generated_files,
            output_dir=self.output_dir,
        )

    def run_from_graph(
        self,
        graph: OpGraph,
        workload_description: str = "",
    ) -> CISGResults:
        """
        Run the pipeline from a pre-built OpGraph (skips model tracing).
        Useful for testing or when you construct the graph manually.
        """
        generated_files: List[Path] = []

        hotspots = detect_hotspots_from_graph(
            graph, top_n=self.top_n_hotspots, only_acceleratable=True
        )
        proposer = InstructionProposer(
            extra_rules=self.extra_rules, speedup_target=self.speedup_target
        )
        proposals = proposer.propose(hotspots)
        estimator = SpeedupEstimator(
            graph=graph,
            hw_peak_flops=self.hw_peak_flops,
            hw_peak_bandwidth_gbs=self.hw_peak_bandwidth_gbs,
        )
        speedup_analyses = estimator.estimate_all(proposals)

        tablegen_emitter = TableGenEmitter(
            output_dir=str(self.output_dir / "tablegen"),
            workload_name=graph.name,
        )
        generated_files.extend(tablegen_emitter.emit(proposals))

        spike_emitter = SpikeExtensionEmitter(
            output_dir=str(self.output_dir / "spike_extension"),
            workload_name=graph.name,
        )
        generated_files.extend(spike_emitter.emit(proposals))

        reporter = ReportGenerator(output_dir=str(self.output_dir / "reports"))
        md_path, json_path = reporter.generate(
            graph=graph,
            hotspots=hotspots,
            proposals=proposals,
            workload_description=workload_description,
        )
        generated_files.extend([md_path, json_path])

        return CISGResults(
            graph=graph,
            hotspots=hotspots,
            proposals=proposals,
            speedup_analyses=speedup_analyses,
            generated_files=generated_files,
            output_dir=self.output_dir,
        )
