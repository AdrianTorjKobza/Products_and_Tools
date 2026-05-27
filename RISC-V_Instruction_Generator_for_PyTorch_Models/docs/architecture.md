# Architecture Deep Dive

## Pipeline Stages

### Stage 1: Workload Analysis (`WorkloadAnalyzer`)

Uses PyTorch FX symbolic tracing to extract a computation graph without executing the model.
For each graph node the analyzer:

- Resolves the op type via `_OP_MAP` (aten string → `OpType` enum)
- Propagates tensor shapes using a `fx.Interpreter` run or forward-hook fallback
- Estimates FLOPs with op-specific formulas (2·M·N·K for matmul, 5·N for softmax, etc.)
- Estimates memory traffic as the sum of all input and output tensor bytes
- Optionally collects per-op execution times via `torch.profiler`

**Limitations of static FX tracing:**
- Dynamic control flow (data-dependent shapes) requires the concrete trace fallback
- Fused kernels (e.g. `scaled_dot_product_attention` calling into FlashAttention) appear as
  a single node — the analyzer recognizes these and proposes the corresponding fused instruction

### Stage 2: Hotspot Detection (`HotspotDetector`)

Scores each node using a weighted sum:

| Signal | Weight (with timing) | Weight (no timing) |
|--------|--------------------|-------------------|
| Time % | 40% | 0% |
| FLOPs % | 35% | 55% |
| Memory % | 15% | 35% |
| Compute-bound flag | 10% | 10% |

Only nodes in `_ACCELERATABLE_OPS` and above `min_flop_threshold` are considered.
The rationale string explains why each node was selected.

### Stage 3: Instruction Proposal (`PatternRuleEngine`)

Rules are pure Python dataclasses with two methods:

```python
def matches(self, hotspot: HotspotResult) -> bool: ...
def propose(self, hotspot: HotspotResult) -> CustomInstruction: ...
```

Rules are sorted by `priority` (lower = higher priority). The first matching rule wins per hotspot.
Each rule produces a `CustomInstruction` containing:

- **Encoding**: opcode space, funct3, funct7 (no encoding conflicts between rules)
- **Semantics**: pseudocode describing the hardware operation
- **TableGen snippet**: ready to paste into LLVM's `RISCVInstrInfo.td`
- **Spike snippet**: C++ `DEFINE_INSN` block for the simulator
- **SpeedupModel**: analytical cycle count estimates

### Stage 4: Speedup Estimation (`SpeedupEstimator`)

Two models run in sequence:

**Roofline model:**
```
perf_achievable = min(hw_peak_flops, AI × hw_bandwidth)
```
Determines whether the kernel is compute-bound or memory-bound.
Memory-bound kernels have their speedup capped by their memory reduction factor.

**Amdahl's Law:**
```
S_system = 1 / ((1 − f) + f / S_kernel)
```
Converts kernel speedup to system-level speedup using the hotspot's runtime fraction.
This is the critical "reality check" — a 10× kernel speedup with f=0.2 gives only ~1.8× system speedup.

### Stage 5: Backend Emission

#### TableGen Emitter
Generates two files:
- `RISCVInstrInfoCustom.td`: instruction class definitions with bit-field encodings
- `RISCVCustomInstrPatterns.td`: ISel DAG pattern stubs (require user-defined intrinsics)

R-type and R4-type formats are handled. R4-type (4-register operands) is used for
`vdotacc` and `sdpa` which need 3 source registers + 1 destination.

#### Spike Extension Emitter
Generates a complete plugin directory:
- `extension.cc`: plugin entry point with `REGISTER_EXTENSION` macro
- `insns/<mnemonic>.h`: per-instruction `DEFINE_INSN` implementation
- `tests/test_<mnemonic>.S`: RISC-V assembly test using `.insn` directive
- `CMakeLists.txt` + `Makefile`: build system for the shared library

### Stage 6: Report Generation

The Markdown report is structured for GitHub rendering:
- Executive summary table
- Per-op FLOPs breakdown table
- Roofline analysis table
- Per-instruction "cards" with encoding, operands, pseudocode, speedup tables
- Integration guide with shell commands
- Amdahl warning when system-level gains are limited

---

## Data Flow

```
nn.Module
    │  FX symbolic_trace()
    ▼
fx.Graph ──────────────────────────────────────┐
    │  _resolve_op_type()                       │
    │  _estimate_flops()                        │
    │  _estimate_memory_bytes()                 │
    ▼                                           │
OpGraph                                        │
    │  torch.profiler (optional)               │
    │  → profiled_time_us per node             │
    ▼                                           │
HotspotDetector                                │
    │  score = f(time%, flops%, mem%, AI)      │
    ▼                                           │
[HotspotResult, ...]  ─────────────────────────┤
    │  PatternRuleEngine.propose_all()          │
    ▼                                           │
[CustomInstruction, ...]                        │
    │                                           │
    ├─→ TableGenEmitter.emit()                  │
    │       → RISCVInstrInfoCustom.td           │
    │       → RISCVCustomInstrPatterns.td       │
    │                                           │
    ├─→ SpikeExtensionEmitter.emit()            │
    │       → extension.cc                      │
    │       → insns/*.h                         │
    │       → tests/*.S                         │
    │                                           │
    ├─→ SpeedupEstimator.estimate_all()         │
    │       → [SpeedupAnalysis, ...]            │
    │                                           │
    └─→ ReportGenerator.generate()             │
            → analysis_report.md               │
            → analysis_report.json ────────────┘
```

---

## Extending CISG

### Adding a new OpType

1. Add to `OpType` enum in `analyzer/op_graph.py`
2. Add aten string mapping in `workload_analyzer.py::_OP_MAP`
3. Add FLOPs estimator case in `_estimate_flops()`
4. Add to `_ACCELERATABLE_OPS` in `hotspot_detector.py` (if it's a good target)

### Adding a new pattern rule

```python
class MyRule(PatternRule):
    priority = 40  # 1 = highest priority, 100 = lowest

    def matches(self, hotspot: HotspotResult) -> bool:
        return hotspot.node.op_type == OpType.MY_OP

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        return CustomInstruction(
            mnemonic="myop",
            description="...",
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R,
            opcode_space=CustomOpcodeSpace.CUSTOM_3,  # pick an unused space
            funct3=0x0,
            funct7=0x20,
            operands=[...],
            asm_syntax="myop rd, rs1, rs2",
            semantics_pseudocode="...",
            speedup_model=SpeedupModel(...),
            tablegen_snippet="...",
            spike_extension_snippet="...",
        )
```

Register it:
```python
pipeline = CISGPipeline(extra_rules=[MyRule()])
```

### Encoding conflict avoidance

Current instruction encodings:

| Instruction | Opcode | funct3 | funct7 |
|-------------|--------|--------|--------|
| `vdotacc`   | 0x0B   | 0x0    | 0x00   |
| `sfmax`     | 0x0B   | 0x1    | 0x02   |
| `fusact.gelu`| 0x0B  | 0x2    | 0x05   |
| `fusact.silu`| 0x0B  | 0x2    | 0x06   |
| `mmtile`    | 0x2B   | 0x1    | 0x01   |
| `bmmtile`   | 0x2B   | 0x2    | 0x01   |
| `lnorm`     | 0x2B   | 0x3    | 0x03   |
| `rmsnorm`   | 0x2B   | 0x4    | 0x03   |
| `sdpa`      | 0x5B   | 0x0    | 0x04   |

When adding new instructions, choose `funct3`/`funct7` values not already in use
within the same opcode space.

---

## Design Decisions

### Why rule-based rather than learned?
Determinism and interpretability. A rule-based system produces the same proposal
every time, is easy to audit, and makes the rationale explicit. Learned approaches
(e.g. RL-based ISA search) are an active research area but are not yet reliable
enough for a portfolio-quality tool.

### Why FX tracing rather than MLIR LINALG?
PyTorch FX tracing is self-contained (no LLVM build required) and covers the
aten op vocabulary used by all standard PyTorch models. MLIR integration would
give finer-grained loop analysis but adds a significant build dependency.
The architecture is layered so that a MLIR-based analyzer can replace
`WorkloadAnalyzer` without touching any other module.

### Why Spike rather than gem5?
Spike's extension mechanism (a simple `DEFINE_INSN` macro + shared library) is
the lightest possible path from a proposed encoding to a running simulation.
gem5 would give cycle-accurate results but requires modifying C++ pipeline models —
a multi-week effort. The generated Spike plugin is a stepping stone to gem5 or
an FPGA prototype.

### Why Amdahl rather than direct speedup multiplication?
A 10× kernel speedup is only a 10× system speedup if the kernel is 100% of the
workload. Amdahl's Law is the correct framing and forces honest reporting.
Many hardware accelerator proposals in the literature are criticised for
quoting kernel speedups without the system-level context.
