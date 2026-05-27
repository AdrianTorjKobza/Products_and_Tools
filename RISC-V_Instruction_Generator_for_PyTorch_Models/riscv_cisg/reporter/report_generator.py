"""
ReportGenerator
===============
Produces a comprehensive Markdown report and a structured JSON report
from a complete CISG analysis run.

The Markdown report is designed to be rendered on GitHub and includes:
  - Executive summary
  - Workload analysis table
  - Hotspot breakdown with Roofline metrics
  - Per-instruction proposal cards
  - Speedup analysis
  - Integration guide
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from riscv_cisg.analyzer.hotspot_detector import HotspotResult
from riscv_cisg.analyzer.op_graph import OpGraph
from riscv_cisg.proposer.custom_instruction import CustomInstruction

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates Markdown and JSON reports from a CISG analysis.

    Parameters
    ----------
    output_dir : str or Path
        Directory where report files are written.
    """

    def __init__(self, output_dir: str = "./output/reports"):
        self.output_dir = Path(output_dir)

    def generate(
        self,
        graph: OpGraph,
        hotspots: List[HotspotResult],
        proposals: List[Tuple[HotspotResult, CustomInstruction]],
        workload_description: str = "",
    ) -> Tuple[Path, Path]:
        """
        Generate both Markdown and JSON reports.

        Returns
        -------
        (markdown_path, json_path)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        md_path = self.output_dir / "analysis_report.md"
        json_path = self.output_dir / "analysis_report.json"

        md_path.write_text(
            self._build_markdown(graph, hotspots, proposals, workload_description),
            encoding="utf-8",
        )
        json_path.write_text(
            self._build_json(graph, hotspots, proposals),
            encoding="utf-8",
        )

        logger.info("Reports written to %s", self.output_dir)
        return md_path, json_path

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _build_markdown(
        self,
        graph: OpGraph,
        hotspots: List[HotspotResult],
        proposals: List[Tuple[HotspotResult, CustomInstruction]],
        workload_description: str,
    ) -> str:
        sections = [
            self._md_header(graph, workload_description),
            self._md_workload_summary(graph),
            self._md_hotspot_table(hotspots),
            self._md_roofline(hotspots),
            self._md_proposals(proposals),
            self._md_speedup_summary(proposals),
            self._md_integration_guide(proposals),
            self._md_footer(),
        ]
        return "\n\n".join(sections)

    def _md_header(self, graph: OpGraph, description: str) -> str:
        return f"""\
# RISC-V CISG — Analysis Report

**Workload:** `{graph.name}`
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total FLOPs:** {graph.total_flops:,}
**Total Memory Traffic:** {graph.total_memory_bytes / 1e6:.2f} MB

{description if description else '_No description provided._'}

---"""

    def _md_workload_summary(self, graph: OpGraph) -> str:
        from riscv_cisg.analyzer.op_graph import OpType
        from collections import Counter

        op_counts: Counter = Counter()
        op_flops: dict = {}

        for node in graph.nodes:
            op_counts[node.op_type.name] += 1
            op_flops[node.op_type.name] = (
                op_flops.get(node.op_type.name, 0) + node.flops
            )

        rows = []
        total_flops = max(graph.total_flops, 1)
        for op_name, count in sorted(op_counts.items()):
            flops = op_flops.get(op_name, 0)
            pct = flops / total_flops * 100
            rows.append(f"| `{op_name}` | {count} | {flops:,} | {pct:.1f}% |")

        table = "\n".join(rows)
        return f"""\
## 1. Workload Analysis

| Op Type | Count | FLOPs | FLOPs % |
|---------|------:|------:|--------:|
{table}

> **Total nodes:** {graph.num_nodes}
> **Total FLOPs:** {graph.total_flops:,}
> **Profiled time:** {graph.total_profiled_time_us:.1f} μs"""

    def _md_hotspot_table(self, hotspots: List[HotspotResult]) -> str:
        rows = []
        for h in hotspots:
            meets = "✅" if h.hotspot_score >= 70 else "⚠️"
            rows.append(
                f"| {h.rank} | `{h.node.op_type.name}` | "
                f"{h.hotspot_score:.1f} | "
                f"{h.flops_pct:.1f}% | "
                f"{h.time_pct:.1f}% | "
                f"{h.memory_pct:.1f}% | "
                f"{h.node.arithmetic_intensity:.1f} | "
                f"{'Compute' if h.node.is_compute_bound else 'Memory'} | "
                f"{'Yes ' + meets if h.is_acceleratable else 'No'} |"
            )
        table = "\n".join(rows)
        return f"""\
## 2. Hotspot Detection

| Rank | Op Type | Score | FLOPs% | Time% | Mem% | AI (FLOPs/B) | Bound | Acceleratable |
|-----:|---------|------:|-------:|------:|-----:|-------------:|-------|---------------|
{table}

> **AI** = Arithmetic Intensity (higher = more compute-bound)"""

    def _md_roofline(self, hotspots: List[HotspotResult]) -> str:
        lines = ["## 3. Roofline Analysis\n"]
        lines.append(
            "The Roofline model identifies whether each hotspot is limited by "
            "compute throughput or memory bandwidth. Custom instructions are most "
            "effective for **compute-bound** kernels.\n"
        )
        lines.append("```")
        lines.append(f"{'Op Type':<35} {'AI (FLOPs/B)':>14} {'Bound':>10} {'Acceleratable':>14}")
        lines.append("─" * 76)
        for h in hotspots:
            bound = "COMPUTE" if h.node.is_compute_bound else "MEMORY"
            acc = "YES ✓" if h.is_acceleratable else "no"
            lines.append(
                f"{h.node.op_type.name:<35} "
                f"{h.node.arithmetic_intensity:>14.2f} "
                f"{bound:>10} "
                f"{acc:>14}"
            )
        lines.append("```")
        return "\n".join(lines)

    def _md_proposals(
        self, proposals: List[Tuple[HotspotResult, CustomInstruction]]
    ) -> str:
        sections = ["## 4. Custom Instruction Proposals\n"]
        for i, (hotspot, instr) in enumerate(proposals, 1):
            speedup = (
                instr.speedup_model.estimated_speedup if instr.speedup_model else 0.0
            )
            meets_badge = (
                "![10x](https://img.shields.io/badge/speedup-10x%20target%20met-brightgreen)"
                if speedup >= 10.0
                else f"![speedup](https://img.shields.io/badge/speedup-{speedup:.1f}x-yellow)"
            )

            fusion_note = ""
            if instr.fusion_opportunity:
                fusion_note = (
                    f"\n> 💡 **Fusion opportunity:** This instruction can be fused with "
                    f"`{', '.join(instr.fusion_partners)}` for additional gains."
                )

            sections.append(f"""\
### 4.{i} `{instr.mnemonic}` {meets_badge}

**Accelerates:** `{instr.target_op_type}`
**Estimated speedup:** **{speedup:.1f}x**{' ✅ meets 10x target' if speedup >= 10.0 else ' ⚠️ below 10x target'}
**Encoding:** `{instr.encoding_summary}`
{fusion_note}

**Description:**
{instr.description}

**Assembly syntax:**
```asm
{instr.asm_syntax}
```

**Operands:**

| Name | Bits | Type | Description |
|------|-----:|------|-------------|
{"".join(f'| `{op.name}` | {op.bits} | {("Reg ("+op.reg_file+")") if op.is_register else "Imm"} | {op.description} |' + chr(10) for op in instr.operands)}

**Semantics (pseudocode):**
```c
{instr.semantics_pseudocode}
```

**Speedup Model:**

| Metric | Baseline | Proposed | Improvement |
|--------|--------:|--------:|------------:|
{"| Cycles | " + (f"{instr.speedup_model.baseline_cycles:,} | {instr.speedup_model.proposed_cycles:,} | {speedup:.1f}× |" if instr.speedup_model else "N/A | N/A | N/A |")}
{"| Memory factor | 1.0× | " + (f"{instr.speedup_model.memory_reduction_factor:.2f}× | {1/instr.speedup_model.memory_reduction_factor:.1f}× less |" if instr.speedup_model else "N/A | N/A |")}

> {instr.speedup_model.notes if instr.speedup_model else ""}

**Rationale:** {hotspot.acceleration_rationale}

---""")
        return "\n".join(sections)

    def _md_speedup_summary(
        self, proposals: List[Tuple[HotspotResult, CustomInstruction]]
    ) -> str:
        rows = []
        for _, instr in proposals:
            speedup = (
                instr.speedup_model.estimated_speedup if instr.speedup_model else 0.0
            )
            rows.append(
                f"| `{instr.mnemonic}` | `{instr.target_op_type}` | "
                f"{speedup:.1f}× | "
                f"{'✅ Yes' if speedup >= 10.0 else '⚠️ No'} |"
            )
        table = "\n".join(rows)
        return f"""\
## 5. Speedup Summary

| Instruction | Target Op | Est. Speedup | Meets 10× Target |
|-------------|-----------|-------------:|:----------------:|
{table}

> **Note:** Speedup estimates are for the *target kernel only*, not end-to-end.
> Actual system speedup depends on the fraction of time spent in these kernels
> (Amdahl's Law). Always validate with cycle-accurate simulation (Spike/gem5)."""

    def _md_integration_guide(
        self, proposals: List[Tuple[HotspotResult, CustomInstruction]]
    ) -> str:
        return """\
## 6. Integration Guide

### Step 1: LLVM Backend

```bash
# Copy TableGen definitions
cp output/tablegen/RISCVInstrInfoCustom.td    /path/to/llvm/lib/Target/RISCV/
cp output/tablegen/RISCVCustomInstrPatterns.td /path/to/llvm/lib/Target/RISCV/

# Add include to RISCVInstrInfo.td
echo 'include "RISCVInstrInfoCustom.td"' >> /path/to/llvm/lib/Target/RISCV/RISCVInstrInfo.td

# Rebuild
cd llvm-build && cmake --build . --target llc -j$(nproc)
```

### Step 2: Spike Simulator

```bash
# Build the extension
cd output/spike_extension
mkdir build && cd build
cmake .. -DSPIKE_ROOT=$HOME/.local
make

# Run with extension loaded
spike --extension=./libcisg_extension.so your_program.elf
```

### Step 3: Validation

```bash
# Disassemble to verify instruction encoding
riscv64-unknown-elf-objdump -d your_program.elf | grep -A2 "custom"

# Run tests
cd output/spike_extension
riscv64-unknown-elf-gcc -march=rv64imfd -o tests/test_instr.elf tests/test_vdotacc.S
spike --extension=./libcisg_extension.so tests/test_instr.elf
```

### Step 4: Intrinsics (optional, for compiler-driven codegen)

Define LLVM intrinsics in `llvm/include/llvm/IR/IntrinsicsRISCV.td` to allow
the compiler to automatically emit these instructions from C code."""

    def _md_footer(self) -> str:
        return """\
---

*Generated by [RISC-V CISG](https://github.com/your-username/riscv-cisg) — \
Automated Custom Instruction Suggestion Generator*
*MIT License*"""

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _build_json(
        self,
        graph: OpGraph,
        hotspots: List[HotspotResult],
        proposals: List[Tuple[HotspotResult, CustomInstruction]],
    ) -> str:
        data = {
            "meta": {
                "generated": datetime.now().isoformat(),
                "tool": "RISC-V CISG v1.0.0",
            },
            "workload": graph.to_dict(),
            "hotspots": [
                {
                    "rank": h.rank,
                    "op_type": h.node.op_type.name,
                    "hotspot_score": round(h.hotspot_score, 3),
                    "flops_pct": round(h.flops_pct, 2),
                    "time_pct": round(h.time_pct, 2),
                    "memory_pct": round(h.memory_pct, 2),
                    "arithmetic_intensity": round(h.node.arithmetic_intensity, 3),
                    "is_compute_bound": h.node.is_compute_bound,
                    "is_acceleratable": h.is_acceleratable,
                    "rationale": h.acceleration_rationale,
                }
                for h in hotspots
            ],
            "proposals": [
                {
                    "hotspot_rank": hotspot.rank,
                    "instruction": instr.to_dict(),
                }
                for hotspot, instr in proposals
            ],
        }
        return json.dumps(data, indent=2)
