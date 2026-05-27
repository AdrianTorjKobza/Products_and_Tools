"""
SpeedupEstimator
================
Validates and refines speedup estimates using:
  1. Analytical roofline model
  2. Amdahl's Law (system-level speedup from kernel-level gains)
  3. (Optional) Spike subprocess invocation for empirical validation

This module gives the realistic speedup picture — both at the kernel
level (what the custom instruction achieves) and at the system level
(what the workload as a whole gains).
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from riscv_cisg.analyzer.hotspot_detector import HotspotResult
from riscv_cisg.analyzer.op_graph import OpGraph
from riscv_cisg.proposer.custom_instruction import CustomInstruction, SpeedupModel

logger = logging.getLogger(__name__)


@dataclass
class SpeedupAnalysis:
    """
    Complete speedup analysis for a single instruction proposal.

    Attributes
    ----------
    mnemonic : str
    kernel_speedup : float
        Speedup of the target kernel alone.
    system_speedup : float
        System-level speedup via Amdahl's Law.
    hotspot_fraction : float
        Fraction of total runtime consumed by the hotspot.
    roofline_peak_speedup : float
        Theoretical maximum speedup from roofline model.
    bottleneck : str
        "compute" or "memory" — the binding constraint.
    meets_10x_kernel : bool
    meets_10x_system : bool
    notes : str
    """
    mnemonic: str
    kernel_speedup: float
    system_speedup: float
    hotspot_fraction: float
    roofline_peak_speedup: float
    bottleneck: str
    meets_10x_kernel: bool
    meets_10x_system: bool
    notes: str = ""

    def summary_line(self) -> str:
        return (
            f"{self.mnemonic:<20} "
            f"kernel={self.kernel_speedup:.1f}x  "
            f"system={self.system_speedup:.2f}x  "
            f"(hotspot={self.hotspot_fraction*100:.1f}%  "
            f"bottleneck={self.bottleneck})"
        )


class SpeedupEstimator:
    """
    Validates speedup estimates for all proposals.

    Parameters
    ----------
    graph : OpGraph
        The analyzed workload graph (used for Amdahl fractions).
    hw_peak_flops : float
        Theoretical peak throughput in GFLOPs/s. Default: 10 GFLOPs/s
        (representative in-order RISC-V core at 1 GHz with 4-wide SIMD).
    hw_peak_bandwidth_gbs : float
        Peak memory bandwidth in GB/s. Default: 8 GB/s.
    spike_binary : str or None
        Path to Spike binary. If set, enables empirical validation mode.
    """

    def __init__(
        self,
        graph: OpGraph,
        hw_peak_flops: float = 10.0,     # GFLOPs/s
        hw_peak_bandwidth_gbs: float = 8.0,  # GB/s
        spike_binary: Optional[str] = None,
    ):
        self.graph = graph
        self.hw_peak_flops = hw_peak_flops
        self.hw_peak_bandwidth_gbs = hw_peak_bandwidth_gbs
        self.spike_binary = spike_binary
        self._ridge_point = hw_peak_flops / hw_peak_bandwidth_gbs  # FLOPs/byte

    def estimate_all(
        self, proposals: List[Tuple[HotspotResult, CustomInstruction]]
    ) -> List[SpeedupAnalysis]:
        """
        Run speedup analysis for all proposals.

        Returns list of SpeedupAnalysis, one per proposal.
        """
        results = []
        for hotspot, instr in proposals:
            analysis = self._analyze_one(hotspot, instr)
            results.append(analysis)
            logger.info(analysis.summary_line())
        return results

    def validate_against_spike(
        self,
        elf_path: str,
        extension_so: str,
    ) -> Optional[Dict[str, float]]:
        """
        Run Spike with the custom extension and return measured cycle counts.

        Parameters
        ----------
        elf_path : str
            Path to RISC-V ELF binary.
        extension_so : str
            Path to compiled Spike extension shared library.

        Returns
        -------
        dict or None
            {"cycles": int, "instructions": int} if Spike is available,
            None otherwise.
        """
        if not self.spike_binary:
            logger.info("Spike binary not configured — skipping empirical validation.")
            return None

        spike_path = Path(self.spike_binary)
        if not spike_path.exists():
            logger.warning("Spike binary not found at %s", self.spike_binary)
            return None

        cmd = [
            str(spike_path),
            f"--extension={extension_so}",
            "--log-commits",
            elf_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            return self._parse_spike_output(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            logger.warning("Spike timed out after 60s")
        except FileNotFoundError:
            logger.warning("Spike binary not executable: %s", self.spike_binary)
        return None

    def amdahl_speedup(self, kernel_speedup: float, fraction: float) -> float:
        """
        Amdahl's Law: system speedup from a kernel-level improvement.

        S_system = 1 / ((1 - f) + f / S_kernel)

        Parameters
        ----------
        kernel_speedup : float
            Speedup of the accelerated kernel.
        fraction : float
            Fraction of total runtime in the kernel [0, 1].
        """
        if kernel_speedup <= 0:
            return 1.0
        return 1.0 / ((1.0 - fraction) + fraction / kernel_speedup)

    def roofline_peak(self, arithmetic_intensity: float) -> float:
        """
        Roofline model: peak achievable GFLOPs/s for a given AI.

        performance = min(hw_peak_flops, AI * hw_bandwidth)
        """
        memory_ceiling = arithmetic_intensity * self.hw_peak_bandwidth_gbs
        return min(self.hw_peak_flops, memory_ceiling)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _analyze_one(
        self, hotspot: HotspotResult, instr: CustomInstruction
    ) -> SpeedupAnalysis:
        node = hotspot.node

        # 1. Kernel speedup from SpeedupModel
        model: Optional[SpeedupModel] = instr.speedup_model
        kernel_speedup = model.estimated_speedup if model else 1.0

        # 2. Hotspot fraction (by time if available, else by FLOPs)
        total_time = self.graph.total_profiled_time_us
        if total_time > 0:
            fraction = node.profiled_time_us / total_time
        else:
            fraction = node.flops / max(self.graph.total_flops, 1)

        fraction = max(min(fraction, 1.0), 0.0)

        # 3. System-level speedup via Amdahl
        system_speedup = self.amdahl_speedup(kernel_speedup, fraction)

        # 4. Roofline ceiling
        roofline_peak = self.roofline_peak(node.arithmetic_intensity)
        # Express roofline peak as speedup over assumed scalar baseline (1 GFLOPs)
        roofline_speedup = roofline_peak / 1.0  # vs. scalar 1 GFLOPs/s

        bottleneck = "compute" if node.is_compute_bound else "memory"

        # 5. Adjust kernel speedup: memory-bound ops can't exceed memory ceiling
        if not node.is_compute_bound:
            # Memory-bound: speedup capped by memory reduction factor
            mem_factor = model.memory_reduction_factor if model else 1.0
            memory_bound_cap = 1.0 / mem_factor if mem_factor > 0 else kernel_speedup
            kernel_speedup = min(kernel_speedup, memory_bound_cap * 2)
            system_speedup = self.amdahl_speedup(kernel_speedup, fraction)

        notes_parts = [
            f"hotspot fraction={fraction*100:.1f}%",
            f"ridge_point={self._ridge_point:.1f} FLOPs/B",
            f"AI={node.arithmetic_intensity:.1f} FLOPs/B",
        ]
        if fraction < 0.1:
            notes_parts.append(
                "⚠ Low hotspot fraction — system-level gain limited by Amdahl"
            )
        if not node.is_compute_bound:
            notes_parts.append(
                "⚠ Memory-bound kernel — custom instruction most effective if "
                "it reduces memory traffic"
            )

        return SpeedupAnalysis(
            mnemonic=instr.mnemonic,
            kernel_speedup=kernel_speedup,
            system_speedup=system_speedup,
            hotspot_fraction=fraction,
            roofline_peak_speedup=roofline_speedup,
            bottleneck=bottleneck,
            meets_10x_kernel=kernel_speedup >= 10.0,
            meets_10x_system=system_speedup >= 10.0,
            notes="; ".join(notes_parts),
        )

    def _parse_spike_output(self, output: str) -> Dict[str, float]:
        """Parse Spike's log output for cycle/instruction counts."""
        metrics: Dict[str, float] = {}
        for line in output.splitlines():
            line = line.strip()
            if "mcycle" in line.lower() or "cycles" in line.lower():
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.isdigit():
                        metrics["cycles"] = float(p)
                        break
            if "minstret" in line.lower() or "instructions" in line.lower():
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        metrics["instructions"] = float(p)
                        break
        return metrics
