# RISC-V CISG — Custom Instruction Suggestion Generator

[![CI](https://github.com/your-username/riscv-cisg/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/riscv-cisg/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)

> **Hardware-Software Co-design tool** that analyzes ML workloads, identifies compute hotspots, and proposes custom RISC-V ISA extensions targeting **10× kernel speedup**.

---

## Overview

Modern ML accelerators (TPUs, NPUs) achieve their performance through tight hardware-software co-design — tailoring the instruction set to the workload. **RISC-V CISG** automates the first step of this process for RISC-V targets.

Given a PyTorch model (or a pre-built computation graph), CISG:

1. **Analyzes** the workload using PyTorch FX tracing + `torch.profiler`
2. **Identifies** the top-N compute hotspots via a multi-metric scoring model (FLOPs, memory traffic, execution time)
3. **Proposes** custom RISC-V instructions using a deterministic pattern-rule engine
4. **Estimates** speedup analytically (Roofline model + Amdahl's Law)
5. **Emits** production-ready integration artifacts:
   - LLVM TableGen definitions for the RISC-V backend
   - Spike ISA simulator plugin (C++ extension)
   - Comprehensive Markdown + JSON reports

```
ML Model (PyTorch)
      │
      ▼
┌─────────────────┐
│ WorkloadAnalyzer│  FX trace + torch.profiler → OpGraph
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│HotspotDetector  │  FLOPs% + Time% + AI → ranked hotspots
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│InstructionPropo-│  Pattern rule engine → CustomInstruction specs
│      ser        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│SpeedupEstimator │  Roofline + Amdahl → kernel & system speedup
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
TableGen    Spike       + Markdown/JSON reports
 (.td)    extension.cc
```

---

## Tech Stack
* **ML Framework**: PyTorch 2.0+
* **Compiler Infrastructure**: LLVM TableGen
* **ISA Simulation**: Spike RISC-V ISA Simulator

---

## Quickstart

### Install
* Clone the repo
* Rename parent folder to `riscv-cisg`
* Setup the virtual env: `python -m venv venv`
- On Linux/MacOs use `source venv/bin/activate`
- On Windows use `venv\Scripts\activate`
* Install dependencies: `pip install -e .`

### Analyze a Transformer layer

```bash
# Built-in Transformer encoder layer
riscv-cisg analyze --workload transformer --d-model 768 --seq-len 128

# Output in ./cisg_output/
#   reports/analysis_report.md
#   tablegen/RISCVInstrInfoCustom.td
#   spike_extension/extension.cc
```

### Python API

```python
from riscv_cisg import CISGPipeline
import torch
import torch.nn as nn

model = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
inputs = (torch.randn(1, 128, 768),)

pipeline = CISGPipeline(output_dir="./cisg_output")
results  = pipeline.run(model, inputs)

print(results.summary())
# ╔══════════════════════════════════════════════════════════════════╗
# ║         RISC-V CISG — Analysis Complete                         ║
# ║  Workload   : TransformerEncoderLayer                           ║
# ║  Proposals  : 4                                                 ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  ✓ mmtile          → MATMUL                    12.4x            ║
# ║  ✓ sfmax           → SOFTMAX                   7.5x             ║
# ║  ✓ lnorm           → LAYER_NORM                5.0x             ║
# ║  ✓ fusact.gelu     → GELU                      7.0x             ║
# ╚══════════════════════════════════════════════════════════════════╝
```

---

## Architecture

### Modules

| Module | Responsibility |
|--------|---------------|
| `analyzer/op_graph.py` | DAG data model: `OpGraph`, `OpNode`, `TensorShape`, `OpType` |
| `analyzer/workload_analyzer.py` | PyTorch FX tracing + `torch.profiler` integration |
| `analyzer/hotspot_detector.py` | Multi-metric hotspot scoring and ranking |
| `proposer/custom_instruction.py` | `CustomInstruction` data model with encoding, speedup, snippets |
| `proposer/pattern_rules.py` | Deterministic rule engine: 6 built-in rules + extensible base |
| `proposer/instruction_proposer.py` | Orchestrates rule engine → ranked proposals |
| `simulator/speedup_estimator.py` | Roofline model, Amdahl's Law, optional Spike validation |
| `backend/tablegen_emitter.py` | LLVM RISC-V TableGen file generation |
| `backend/spike_emitter.py` | Spike ISA simulator plugin generation |
| `reporter/report_generator.py` | Markdown + JSON report generation |
| `pipeline.py` | `CISGPipeline` — end-to-end orchestration |
| `cli.py` | `riscv-cisg` command-line interface |

### Built-in Pattern Rules

| Rule | Target Op | Instruction | Key Idea |
|------|-----------|-------------|----------|
| `ScaledDotProductAttentionRule` | `SCALED_DOT_PRODUCT_ATTENTION` | `sdpa` | FlashAttention-style fused QK^T+softmax+AV, eliminates O(S²) memory |
| `MatMulTileRule` | `MATMUL`, `BATCH_MATMUL` | `mmtile` | 8×8 tiled matmul, systolic-array dataflow |
| `DotProductAccumulateRule` | `DOT_PRODUCT`, `MATVEC` | `vdotacc` | Fused dot-product accumulate over short vectors (≤64 elements) |
| `SoftmaxRule` | `SOFTMAX` | `sfmax` | 3-pass softmax fused into single-pass hardware pipeline |
| `LayerNormRule` | `LAYER_NORM`, `RMS_NORM` | `lnorm` / `rmsnorm` | Fused mean-variance-normalize |
| `ActivationFusionRule` | `GELU`, `SiLU` | `fusact.gelu` / `fusact.silu` | Degree-3 Chebyshev polynomial approximation |

---

## RISC-V Encoding

All proposed instructions use RISC-V's reserved custom opcode spaces:

| Opcode Space | Value | Used for |
|---|---|---|
| `custom-0` | `0x0B` | `vdotacc`, `sfmax`, `fusact.*` |
| `custom-1` | `0x2B` | `mmtile`, `lnorm`, `rmsnorm` |
| `custom-2` | `0x5B` | `sdpa` |
| `custom-3` | `0x7B` | Reserved for future extensions |

Example encoding for `mmtile` (R-type):

```
 31      25 24   20 19   15 14  12 11    7 6      0
┌──────────┬───────┬───────┬──────┬───────┬────────┐
│  funct7  │  rs2  │  rs1  │funct3│  rd   │ opcode │
│  0000001 │ xxxxx │ xxxxx │ 001  │ xxxxx │0101011 │
└──────────┴───────┴───────┴──────┴───────┴────────┘
```

---

## LLVM Integration

### 1. Copy TableGen definitions

```bash
cp cisg_output/tablegen/RISCVInstrInfoCustom.td \
   /path/to/llvm/lib/Target/RISCV/

echo 'include "RISCVInstrInfoCustom.td"' \
   >> /path/to/llvm/lib/Target/RISCV/RISCVInstrInfo.td
```

### 2. Rebuild LLVM

```bash
cd llvm-build/
cmake --build . --target llc -j$(nproc)
```

### 3. Verify

```bash
llc -march=riscv32 -mattr=+m,+f -o /dev/null /dev/null
# Should not error with the new instruction definitions
```

---

## Spike Integration

### Build the extension

```bash
# Requires: Spike ≥ 1.1.0, GCC/Clang with C++17

cd cisg_output/spike_extension/
mkdir build && cd build
cmake .. -DSPIKE_ROOT=$HOME/.local   # adjust to your Spike install
make -j$(nproc)
```

### Run with extension

```bash
spike --extension=./libcisg_extension.so your_program.elf
```

### Run assembly tests

```bash
# Build test for mmtile
riscv64-unknown-elf-gcc -march=rv64imfd \
  -o tests/test_mmtile.elf tests/test_mmtile.S

spike --extension=./libcisg_extension.so tests/test_mmtile.elf
```

---

## CLI Reference

```
riscv-cisg analyze [OPTIONS]

  --workload NAME        Built-in workload: transformer | attention | ffn
  --model-file PATH      Path to Python file with nn.Module class
  --model-class CLASS    Class name in --model-file (default: Model)
  --input-shape DIM,...  Input tensor shape e.g. 1,128,768
  --d-model INT          Model dimension for transformer (default: 768)
  --n-heads INT          Attention heads for transformer (default: 12)
  --seq-len INT          Sequence length for transformer (default: 128)
  --top-n INT            Number of hotspots to analyze (default: 5)
  --output-dir PATH      Output directory (default: ./cisg_output)
  --no-profile           Skip torch.profiler (faster, less accurate)
  --speedup-target FLOAT Target speedup threshold (default: 10.0)
  --hw-peak-flops FLOAT  HW peak GFLOPs/s (default: 10.0)
  --hw-peak-bandwidth F  HW peak GB/s (default: 8.0)
  --spike-binary PATH    Path to Spike for empirical validation
  --quiet                Suppress progress output

riscv-cisg list-workloads
```

---

## Adding Custom Rules

Extend the rule engine by subclassing `PatternRule`:

```python
from riscv_cisg.proposer.pattern_rules import PatternRule
from riscv_cisg.analyzer.hotspot_detector import HotspotResult
from riscv_cisg.proposer.custom_instruction import CustomInstruction

class MyCustomRule(PatternRule):
    priority = 25  # lower = evaluated first

    def matches(self, hotspot: HotspotResult) -> bool:
        return hotspot.node.op_type == OpType.MY_OP

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        return CustomInstruction(
            mnemonic="mycustom",
            description="My custom op...",
            # ... full specification
        )

# Inject into pipeline
pipeline = CISGPipeline(extra_rules=[MyCustomRule()])
```

See [`examples/custom_rule_example.py`](examples/custom_rule_example.py) for a complete RoPE fusion example.

---

## Speedup Methodology

### Kernel-level speedup

Estimated analytically from the `SpeedupModel` in each pattern rule:

```
speedup_kernel = baseline_cycles / proposed_cycles
```

Baseline assumes a scalar RISC-V loop. Proposed counts assume:
- `mmtile`: 8×8 systolic tile engine at 4 FMAs/cycle
- `sfmax`: pipelined 3-pass softmax at 2 cycles/element
- `vdotacc`: 8-wide FP32 pipeline
- `lnorm`/`rmsnorm`: 2-pass fused pipeline
- `fusact.*`: degree-3 polynomial, 2 cycles/element

### System-level speedup (Amdahl's Law)

```
S_system = 1 / ((1 - f) + f / S_kernel)
```

where `f` = fraction of total runtime in the hotspot kernel.

### Important caveats

- Speedup estimates are **kernel-level**, not end-to-end
- Memory-bound kernels are capped by the memory reduction factor
- Validation against a cycle-accurate simulator (Spike with custom extension) is the gold standard

---

## Development

```bash
# Install dev deps
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=riscv_cisg --cov-report=term-missing

# Format
black riscv_cisg/ tests/ examples/

# Lint
ruff check riscv_cisg/ tests/
```

---

## Project Structure

```
riscv-cisg/
├── riscv_cisg/
│   ├── __init__.py
│   ├── pipeline.py              # CISGPipeline — main orchestrator
│   ├── cli.py                   # riscv-cisg CLI
│   ├── analyzer/
│   │   ├── op_graph.py          # OpGraph, OpNode, OpType, TensorShape
│   │   ├── workload_analyzer.py # PyTorch FX tracer + profiler
│   │   └── hotspot_detector.py  # Hotspot scoring & ranking
│   ├── proposer/
│   │   ├── custom_instruction.py # CustomInstruction data model
│   │   ├── pattern_rules.py      # Rule engine + 6 built-in rules
│   │   └── instruction_proposer.py
│   ├── simulator/
│   │   └── speedup_estimator.py  # Roofline + Amdahl + Spike validation
│   ├── backend/
│   │   ├── tablegen_emitter.py   # LLVM .td file generation
│   │   └── spike_emitter.py      # Spike plugin generation
│   └── reporter/
│       └── report_generator.py   # Markdown + JSON reports
├── examples/
│   ├── transformer_example.py    # Full transformer layer demo
│   └── custom_rule_example.py    # Adding a custom rule (RoPE)
├── tests/
│   └── test_cisg.py              # pytest test suite (30+ tests)
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Roadmap (ideas)

- [ ] MLIR LINALG op extraction (complement FX tracing)
- [ ] gem5 integration for cycle-accurate validation
- [ ] Multi-instruction fusion proposals (e.g., `mmtile` → `sfmax` → `vdotacc`)
- [ ] INT8 / FP16 quantized instruction variants
- [ ] Web UI for interactive report browsing
- [ ] Support for more workloads: ResNet, BERT, LLaMA, Whisper

---

## Related Work

- [Gemmini](https://github.com/ucb-bar/gemmini) — RISC-V systolic array generator
- [IREE](https://github.com/openxla/iree) — ML compiler targeting custom backends
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — inspiration for `sdpa` instruction
- [RISC-V "P" extension](https://github.com/riscv/riscv-p-spec) — packed-SIMD standard

---

## License

MIT © 2024 — see [LICENSE](LICENSE)