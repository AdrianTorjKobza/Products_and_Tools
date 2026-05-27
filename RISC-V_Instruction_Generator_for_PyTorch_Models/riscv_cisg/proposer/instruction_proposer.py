"""
InstructionProposer
===================
High-level interface for generating custom instruction proposals from
a list of hotspot analysis results.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from riscv_cisg.analyzer.hotspot_detector import HotspotResult
from riscv_cisg.proposer.custom_instruction import CustomInstruction
from riscv_cisg.proposer.pattern_rules import PatternRule, PatternRuleEngine

logger = logging.getLogger(__name__)


class InstructionProposer:
    """
    Generates custom RISC-V instruction proposals for a set of hotspots.

    Parameters
    ----------
    extra_rules : list of PatternRule, optional
        Additional custom rules to register beyond the built-in defaults.
    speedup_target : float
        Minimum speedup threshold. Proposals below this are flagged (not filtered).
    """

    def __init__(
        self,
        extra_rules: Optional[List[PatternRule]] = None,
        speedup_target: float = 10.0,
    ):
        self.speedup_target = speedup_target
        self._engine = PatternRuleEngine()

        if extra_rules:
            for rule in extra_rules:
                self._engine.add_rule(rule)

    def propose(
        self, hotspots: List[HotspotResult]
    ) -> List[Tuple[HotspotResult, CustomInstruction]]:
        """
        Generate instruction proposals for all hotspots.

        Parameters
        ----------
        hotspots : list of HotspotResult
            Ranked hotspots from HotspotDetector.

        Returns
        -------
        list of (HotspotResult, CustomInstruction)
            Pairs of (hotspot, proposal), sorted by estimated speedup descending.
        """
        if not hotspots:
            logger.warning("No hotspots provided — nothing to propose.")
            return []

        proposals = self._engine.propose_all(hotspots)

        # Sort by estimated speedup
        proposals.sort(
            key=lambda p: (
                p[1].speedup_model.estimated_speedup
                if p[1].speedup_model
                else 0.0
            ),
            reverse=True,
        )

        # Log summary
        for hotspot, instr in proposals:
            speedup = (
                instr.speedup_model.estimated_speedup if instr.speedup_model else 0.0
            )
            meets = "✓" if speedup >= self.speedup_target else "✗"
            logger.info(
                "%s Proposed %-15s for %-35s → %.1fx speedup",
                meets,
                instr.mnemonic,
                hotspot.node.op_type.name,
                speedup,
            )

        return proposals

    def summary(
        self, proposals: List[Tuple[HotspotResult, CustomInstruction]]
    ) -> str:
        """Return a plain-text summary of all proposals."""
        if not proposals:
            return "No proposals generated."

        lines = [
            "=" * 70,
            "  RISC-V CUSTOM INSTRUCTION PROPOSALS",
            "=" * 70,
        ]

        for i, (hotspot, instr) in enumerate(proposals, 1):
            speedup = (
                instr.speedup_model.estimated_speedup if instr.speedup_model else 0.0
            )
            meets = "MEETS 10x TARGET" if speedup >= self.speedup_target else f"below target"
            lines += [
                f"\n[{i}] {instr.mnemonic.upper()}",
                f"    Target op  : {instr.target_op_type}",
                f"    Description: {instr.description[:100]}...",
                f"    Encoding   : {instr.encoding_summary}",
                f"    Assembly   : {instr.asm_syntax}",
                f"    Speedup    : {speedup:.1f}x  [{meets}]",
                f"    Rationale  : {hotspot.acceleration_rationale[:100]}",
            ]
            if instr.fusion_opportunity:
                lines.append(f"    Fusion with: {', '.join(instr.fusion_partners)}")

        lines += ["", "=" * 70]
        return "\n".join(lines)
