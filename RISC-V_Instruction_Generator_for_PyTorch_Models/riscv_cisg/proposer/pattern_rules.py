"""
PatternRuleEngine
=================
Deterministic, rule-based engine that maps OpType + shape characteristics
to custom RISC-V instruction proposals.

Each rule is a Python dataclass that:
  1. Matches a specific OpType (and optionally shape constraints)
  2. Produces a fully-specified CustomInstruction

Rules are evaluated in priority order. The first matching rule wins.
Multiple rules can fire for the same hotspot if they cover different
aspects (e.g., a compute rule + a fusion rule).
"""

from __future__ import annotations

import logging
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from riscv_cisg.analyzer.op_graph import OpNode, OpType, TensorShape
from riscv_cisg.analyzer.hotspot_detector import HotspotResult
from riscv_cisg.proposer.custom_instruction import (
    CustomInstruction,
    CustomOpcodeSpace,
    InstructionFormat,
    InstructionOperand,
    SpeedupModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base rule
# ---------------------------------------------------------------------------

class PatternRule(ABC):
    """Abstract base for all pattern rules."""

    priority: int = 50  # Lower number = evaluated first

    @abstractmethod
    def matches(self, hotspot: HotspotResult) -> bool:
        """Return True if this rule applies to the given hotspot."""
        ...

    @abstractmethod
    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        """Produce a CustomInstruction for the matched hotspot."""
        ...

    def _first_input(self, hotspot: HotspotResult) -> Optional[TensorShape]:
        shapes = hotspot.node.input_shapes
        return shapes[0] if shapes else None

    def _second_input(self, hotspot: HotspotResult) -> Optional[TensorShape]:
        shapes = hotspot.node.input_shapes
        return shapes[1] if len(shapes) > 1 else None


# ---------------------------------------------------------------------------
# Rule: Dot-Product Accumulate  (MATMUL / DOT_PRODUCT on small vectors)
# ---------------------------------------------------------------------------

class DotProductAccumulateRule(PatternRule):
    """
    Matches small dot-product loops (inner dimension ≤ 64).
    Proposes VDOTACC: fused multiply-accumulate across a vector tile.

    Instruction semantics:
        rd = rd + sum_i(rs1[i] * rs2[i])   for i in 0..N-1

    Encoding: custom-0, R4-type
    Speedup rationale: replaces N FMAs + N-1 ADDs with 1 instruction.
    """
    priority = 10

    def matches(self, hotspot: HotspotResult) -> bool:
        op = hotspot.node.op_type
        if op not in (OpType.DOT_PRODUCT, OpType.MATVEC):
            return False
        inp = self._first_input(hotspot)
        if inp and len(inp.dims) >= 1:
            inner = inp.dims[-1]
            return inner <= 64
        return op == OpType.DOT_PRODUCT

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        inp = self._first_input(hotspot)
        vec_len = inp.dims[-1] if inp and len(inp.dims) >= 1 else 32

        baseline_cycles = vec_len * 2  # 1 FMA + 1 ADD per element
        proposed_cycles = max(4, vec_len // 8)  # 8-wide SIMD pipeline

        return CustomInstruction(
            mnemonic="vdotacc",
            description=(
                f"Vector dot-product accumulate: rd += sum(rs1[0..{vec_len-1}] * rs2[0..{vec_len-1}]). "
                "Fuses multiply-accumulate across a fixed-length vector tile into a single instruction, "
                "eliminating loop overhead and intermediate register pressure."
            ),
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R4,
            opcode_space=CustomOpcodeSpace.CUSTOM_0,
            funct3=0x0,
            funct7=0x00,
            operands=[
                InstructionOperand("rd",  5, True,  "f", "Accumulator (fp register, read-write)"),
                InstructionOperand("rs1", 5, True,  "f", "Base address or vector reg of operand A"),
                InstructionOperand("rs2", 5, True,  "f", "Base address or vector reg of operand B"),
                InstructionOperand("rs3", 5, True,  "x", "Length register (number of elements)"),
            ],
            asm_syntax=f"vdotacc  rd, rs1, rs2, rs3",
            semantics_pseudocode=textwrap.dedent(f"""\
                // VDOTACC rd, rs1, rs2, rs3
                // rs3 = number of elements (max {vec_len})
                float acc = FReg[rd];
                int  n   = XReg[rs3];
                for (int i = 0; i < n; i++) {{
                    acc += FMem[rs1 + i] * FMem[rs2 + i];
                }}
                FReg[rd] = acc;
            """),
            speedup_model=SpeedupModel(
                baseline_ops=vec_len * 2,
                proposed_ops=1,
                baseline_cycles=baseline_cycles,
                proposed_cycles=proposed_cycles,
                memory_reduction_factor=0.5,
                notes=f"Assumes 8-wide FP32 pipeline; vector length={vec_len}",
            ),
            tablegen_snippet=_tablegen_r4("VDOTACC", "vdotacc", 0x0B, 0x0, 0x00),
            spike_extension_snippet=_spike_vdotacc(vec_len),
            rationale=hotspot.acceleration_rationale,
        )


# ---------------------------------------------------------------------------
# Rule: Matrix Multiply (large MATMUL)
# ---------------------------------------------------------------------------

class MatMulTileRule(PatternRule):
    """
    Matches large matrix multiplications (M*N*K ≥ 1M).
    Proposes MMTILE: a tiled matrix-multiply instruction operating on
    a fixed tile (e.g., 8×8×8) using register-file tiling.

    Speedup rationale:
      - Eliminates loop control overhead (3 nested loops → 1 instruction)
      - Enables systolic-array-style dataflow in the backend
      - Reuses A/B tiles across multiple output rows
    """
    priority = 20

    def matches(self, hotspot: HotspotResult) -> bool:
        if hotspot.node.op_type not in (OpType.MATMUL, OpType.BATCH_MATMUL):
            return False
        return hotspot.node.flops >= 1_000_000

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        inp_a = self._first_input(hotspot)
        inp_b = self._second_input(hotspot)
        is_batched = hotspot.node.op_type == OpType.BATCH_MATMUL

        if inp_a and len(inp_a.dims) >= 2:
            M = inp_a.dims[-2]
            K = inp_a.dims[-1]
        else:
            M, K = 64, 64
        if inp_b and len(inp_b.dims) >= 2:
            N = inp_b.dims[-1]
        else:
            N = 64

        TILE = 8
        tiles = ((M + TILE - 1) // TILE) * ((N + TILE - 1) // TILE) * ((K + TILE - 1) // TILE)
        baseline_cycles = 2 * M * N * K  # scalar FMA loop
        proposed_cycles = tiles * (TILE ** 3 // 4)  # ~4 FMAs/cycle in tile engine

        return CustomInstruction(
            mnemonic="mmtile" if not is_batched else "bmmtile",
            description=(
                f"{'Batched ' if is_batched else ''}Tiled matrix multiply: "
                f"C[TILE×TILE] += A[TILE×{K}] × B[{K}×TILE]. "
                f"Operates on 8×8 output tiles using register-level tiling. "
                f"Designed to feed a systolic-array execution unit."
            ),
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R,
            opcode_space=CustomOpcodeSpace.CUSTOM_1,
            funct3=0x1 if not is_batched else 0x2,
            funct7=0x01,
            operands=[
                InstructionOperand("rd",  5, True, "x", "Base ptr to output tile C (8×8 fp32)"),
                InstructionOperand("rs1", 5, True, "x", "Base ptr to input tile A (8×K fp32)"),
                InstructionOperand("rs2", 5, True, "x", "Base ptr to input tile B (K×8 fp32)"),
            ],
            asm_syntax="mmtile  rd, rs1, rs2" if not is_batched else "bmmtile rd, rs1, rs2",
            semantics_pseudocode=textwrap.dedent(f"""\
                // MMTILE rd, rs1, rs2
                // C (8x8) at Mem[rd], A (8x{K}) at Mem[rs1], B ({K}x8) at Mem[rs2]
                float C[8][8];  // loaded from Mem[rd]
                float A[8][{K}]; float B[{K}][8];
                load_tile(C, Mem[rd], 8, 8);
                load_tile(A, Mem[rs1], 8, {K});
                load_tile(B, Mem[rs2], {K}, 8);
                for (int i=0; i<8; i++)
                  for (int j=0; j<8; j++)
                    for (int k=0; k<{K}; k++)
                      C[i][j] += A[i][k] * B[k][j];
                store_tile(C, Mem[rd], 8, 8);
            """),
            speedup_model=SpeedupModel(
                baseline_ops=2 * M * N * K,
                proposed_ops=tiles,
                baseline_cycles=baseline_cycles,
                proposed_cycles=proposed_cycles,
                memory_reduction_factor=0.3,
                notes=f"8×8 tile, M={M}, N={N}, K={K}; ~4 FMAs/cycle throughput",
            ),
            tablegen_snippet=_tablegen_r("MMTILE", "mmtile", 0x2B, 0x1, 0x01),
            spike_extension_snippet=_spike_mmtile(M, N, K),
            rationale=hotspot.acceleration_rationale,
        )


# ---------------------------------------------------------------------------
# Rule: Softmax
# ---------------------------------------------------------------------------

class SoftmaxRule(PatternRule):
    """
    Matches SOFTMAX ops.
    Proposes SFMAX: fused softmax in a single instruction.

    3-pass softmax (max, exp+sum, normalize) is replaced with a
    hardware pipeline that overlaps all 3 passes.
    """
    priority = 15

    def matches(self, hotspot: HotspotResult) -> bool:
        return hotspot.node.op_type == OpType.SOFTMAX

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        inp = self._first_input(hotspot)
        seq_len = inp.dims[-1] if inp and len(inp.dims) >= 1 else 128

        baseline_cycles = seq_len * 15  # 3 passes × 5 ops/element
        proposed_cycles = seq_len * 2   # pipelined: ~2 cycles/element

        return CustomInstruction(
            mnemonic="sfmax",
            description=(
                f"Fused softmax: computes max→exp→sum→normalize in a single-pass "
                f"hardware pipeline over a vector of length rs2. "
                f"Eliminates the 3-pass sequential dependency of software softmax."
            ),
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R,
            opcode_space=CustomOpcodeSpace.CUSTOM_0,
            funct3=0x1,
            funct7=0x02,
            operands=[
                InstructionOperand("rd",  5, True, "x", "Base ptr to output vector"),
                InstructionOperand("rs1", 5, True, "x", "Base ptr to input vector"),
                InstructionOperand("rs2", 5, True, "x", "Length of vector (number of elements)"),
            ],
            asm_syntax="sfmax  rd, rs1, rs2",
            semantics_pseudocode=textwrap.dedent(f"""\
                // SFMAX rd, rs1, rs2
                int n = XReg[rs2];
                float* x = (float*)Mem[rs1];
                float* y = (float*)Mem[rd];
                // Hardware-pipelined single-pass:
                float m = -INF;
                for (int i=0; i<n; i++) m = max(m, x[i]);      // pass 1
                float s = 0;
                for (int i=0; i<n; i++) s += exp(x[i] - m);    // pass 2
                for (int i=0; i<n; i++) y[i] = exp(x[i]-m)/s;  // pass 3
                // (In hardware: all 3 passes overlap via systolic pipeline)
            """),
            speedup_model=SpeedupModel(
                baseline_ops=seq_len * 15,
                proposed_ops=seq_len * 2,
                baseline_cycles=baseline_cycles,
                proposed_cycles=proposed_cycles,
                memory_reduction_factor=0.67,
                notes=f"3-pass fusion, seq_len={seq_len}",
            ),
            tablegen_snippet=_tablegen_r("SFMAX", "sfmax", 0x0B, 0x1, 0x02),
            spike_extension_snippet=_spike_sfmax(),
            fusion_opportunity=True,
            fusion_partners=["mmtile", "vdotacc"],
            rationale=hotspot.acceleration_rationale,
        )


# ---------------------------------------------------------------------------
# Rule: LayerNorm / RMSNorm
# ---------------------------------------------------------------------------

class LayerNormRule(PatternRule):
    """
    Matches LAYER_NORM / RMS_NORM ops.
    Proposes LNORM: fused mean-variance-normalize instruction.
    """
    priority = 15

    def matches(self, hotspot: HotspotResult) -> bool:
        return hotspot.node.op_type in (OpType.LAYER_NORM, OpType.RMS_NORM)

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        is_rms = hotspot.node.op_type == OpType.RMS_NORM
        mnemonic = "rmsnorm" if is_rms else "lnorm"
        inp = self._first_input(hotspot)
        hidden = inp.dims[-1] if inp and len(inp.dims) >= 1 else 768

        baseline_cycles = hidden * 10  # mean + var + normalize: ~10 ops/element
        proposed_cycles = hidden * 2

        desc = (
            "RMS normalization: y[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]"
            if is_rms else
            "Layer normalization: y[i] = (x[i]-mean) / sqrt(var+eps) * gamma[i] + beta[i]"
        )

        return CustomInstruction(
            mnemonic=mnemonic,
            description=desc,
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R,
            opcode_space=CustomOpcodeSpace.CUSTOM_1,
            funct3=0x3 if not is_rms else 0x4,
            funct7=0x03,
            operands=[
                InstructionOperand("rd",  5, True, "x", "Output vector ptr"),
                InstructionOperand("rs1", 5, True, "x", "Input vector ptr"),
                InstructionOperand("rs2", 5, True, "x", "Weight/bias vector ptr (gamma, beta)"),
            ],
            asm_syntax=f"{mnemonic}  rd, rs1, rs2",
            semantics_pseudocode=textwrap.dedent(f"""\
                // {mnemonic.upper()} rd, rs1, rs2
                // Operates on hidden_dim={hidden} elements
                float* x = Mem[rs1]; float* y = Mem[rd];
                float* gamma = Mem[rs2];
                {"float ms = mean_square(x, " + str(hidden) + ");" if is_rms else
                 "float m = mean(x, " + str(hidden) + "); float v = variance(x, m, " + str(hidden) + ");"}
                float eps = 1e-5;
                for (int i=0; i<{hidden}; i++) {{
                    {"y[i] = x[i] / sqrt(ms + eps) * gamma[i];" if is_rms else
                     "y[i] = (x[i]-m) / sqrt(v+eps) * gamma[i] + beta[i];"}
                }}
            """),
            speedup_model=SpeedupModel(
                baseline_ops=hidden * 10,
                proposed_ops=hidden * 2,
                baseline_cycles=baseline_cycles,
                proposed_cycles=proposed_cycles,
                memory_reduction_factor=0.5,
                notes=f"hidden_dim={hidden}, fused 2-pass pipeline",
            ),
            tablegen_snippet=_tablegen_r(mnemonic.upper(), mnemonic, 0x2B, 0x3, 0x03),
            spike_extension_snippet=_spike_lnorm(is_rms, hidden),
            rationale=hotspot.acceleration_rationale,
        )


# ---------------------------------------------------------------------------
# Rule: Scaled Dot-Product Attention (full flash-attention style)
# ---------------------------------------------------------------------------

class ScaledDotProductAttentionRule(PatternRule):
    """
    Matches SCALED_DOT_PRODUCT_ATTENTION.
    Proposes SDPA: full fused attention instruction (FlashAttention-style).

    Covers: QK^T scaling → softmax → AV in one instruction.
    Eliminates the N² intermediate score matrix from memory.
    """
    priority = 5  # Highest priority — catches full attention before sub-ops

    def matches(self, hotspot: HotspotResult) -> bool:
        return hotspot.node.op_type == OpType.SCALED_DOT_PRODUCT_ATTENTION

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        inp = self._first_input(hotspot)
        if inp and len(inp.dims) == 4:
            B, H, S, D = inp.dims
        else:
            B, H, S, D = 1, 8, 128, 64

        baseline_cycles = 2 * B * H * S * S * D + 5 * B * H * S * S + 2 * B * H * S * D * S
        proposed_cycles = B * H * S * D * 4  # FlashAttention: O(S*D) not O(S^2)

        return CustomInstruction(
            mnemonic="sdpa",
            description=(
                f"Scaled dot-product attention: Out = softmax(Q @ K^T / sqrt(d)) @ V. "
                f"Fuses QK^T matmul, scaling, softmax, and AV matmul into a single "
                f"tiled instruction. Eliminates the O(S²) attention score matrix "
                f"from memory (FlashAttention-style tiling). "
                f"Targets: B={B}, H={H}, S={S}, D={D}."
            ),
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R4,
            opcode_space=CustomOpcodeSpace.CUSTOM_2,
            funct3=0x0,
            funct7=0x04,
            operands=[
                InstructionOperand("rd",  5, True, "x", "Output matrix O ptr (B×H×S×D)"),
                InstructionOperand("rs1", 5, True, "x", "Query matrix Q ptr (B×H×S×D)"),
                InstructionOperand("rs2", 5, True, "x", "Key matrix K ptr (B×H×S×D)"),
                InstructionOperand("rs3", 5, True, "x", "Value matrix V ptr (B×H×S×D)"),
            ],
            asm_syntax="sdpa  rd, rs1, rs2, rs3",
            semantics_pseudocode=textwrap.dedent(f"""\
                // SDPA rd, rs1, rs2, rs3
                // Q,K,V at Mem[rs1/rs2/rs3]; shape (B={B},H={H},S={S},D={D})
                float scale = 1.0f / sqrt({D});
                for b in 0..{B}: for h in 0..{H}:
                  // Tiled FlashAttention: process in tiles of BLOCK_S
                  for tile in tiles(S):
                    float S_tile[BLOCK_S][BLOCK_S];
                    matmul(S_tile, Q[b,h,tile], K[b,h,:].T);  // QK^T
                    scale_inplace(S_tile, scale);
                    online_softmax(S_tile);                    // numerically stable
                    matmul_acc(O[b,h,tile], S_tile, V[b,h,:]); // AV
                    // S_tile never written to main memory
            """),
            speedup_model=SpeedupModel(
                baseline_ops=int(baseline_cycles),
                proposed_ops=int(proposed_cycles),
                baseline_cycles=int(baseline_cycles),
                proposed_cycles=int(proposed_cycles),
                memory_reduction_factor=1.0 / S,  # eliminates S×S matrix
                notes=(
                    f"FlashAttention tiling eliminates O(S²)={S*S} intermediate "
                    f"scores from memory. B={B},H={H},S={S},D={D}."
                ),
            ),
            tablegen_snippet=_tablegen_r4("SDPA", "sdpa", 0x5B, 0x0, 0x04),
            spike_extension_snippet=_spike_sdpa(B, H, S, D),
            fusion_opportunity=True,
            fusion_partners=["mmtile", "sfmax", "vdotacc"],
            rationale=hotspot.acceleration_rationale,
        )


# ---------------------------------------------------------------------------
# Rule: GELU / SiLU activation
# ---------------------------------------------------------------------------

class ActivationFusionRule(PatternRule):
    """
    Matches GELU / SiLU ops.
    Proposes FUSACT: fast polynomial-approximated activation instruction.
    """
    priority = 30

    def matches(self, hotspot: HotspotResult) -> bool:
        return hotspot.node.op_type in (OpType.GELU, OpType.SILU)

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        is_gelu = hotspot.node.op_type == OpType.GELU
        mnemonic = "fusact.gelu" if is_gelu else "fusact.silu"

        inp = self._first_input(hotspot)
        n = inp.num_elements if inp else 4096

        baseline_cycles = n * 14  # tanh-based GELU
        proposed_cycles = n * 2   # degree-3 polynomial approx

        return CustomInstruction(
            mnemonic=mnemonic,
            description=(
                f"Fused {'GELU' if is_gelu else 'SiLU'} activation using a degree-3 "
                f"Chebyshev polynomial approximation. Replaces transcendental function "
                f"calls with a 3-multiply + 2-add sequence accurate to 1e-4."
            ),
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R,
            opcode_space=CustomOpcodeSpace.CUSTOM_0,
            funct3=0x2,
            funct7=0x05 if is_gelu else 0x06,
            operands=[
                InstructionOperand("rd",  5, True, "x", "Output vector ptr"),
                InstructionOperand("rs1", 5, True, "x", "Input vector ptr"),
                InstructionOperand("rs2", 5, True, "x", "Length register"),
            ],
            asm_syntax=f"{mnemonic}  rd, rs1, rs2",
            semantics_pseudocode=textwrap.dedent(f"""\
                // {mnemonic.upper()} rd, rs1, rs2
                int n = XReg[rs2];
                float* x = Mem[rs1]; float* y = Mem[rd];
                for (int i=0; i<n; i++) {{
                    {'// GELU approx: x * 0.5 * (1 + tanh(0.7978*(x + 0.0447*x^3)))' if is_gelu
                     else '// SiLU: x * sigmoid(x) = x / (1 + exp(-x))'}
                    {'float c = 0.7978f*(x[i]+0.04471f*x[i]*x[i]*x[i]);' if is_gelu else ''}
                    {'y[i] = x[i]*0.5f*(1.0f+poly_tanh(c));' if is_gelu
                     else 'y[i] = x[i] / (1.0f + poly_exp(-x[i]));'}
                }}
            """),
            speedup_model=SpeedupModel(
                baseline_ops=n * 14,
                proposed_ops=n * 2,
                baseline_cycles=baseline_cycles,
                proposed_cycles=proposed_cycles,
                notes=f"Degree-3 Chebyshev approx, n={n}",
            ),
            tablegen_snippet=_tablegen_r(
                mnemonic.upper().replace(".", "_"), mnemonic, 0x0B, 0x2, 0x05
            ),
            spike_extension_snippet=_spike_activation(is_gelu),
            rationale=hotspot.acceleration_rationale,
        )


# ---------------------------------------------------------------------------
# TableGen + Spike snippet generators
# ---------------------------------------------------------------------------

def _tablegen_r(
    class_name: str, mnemonic: str, opcode: int, funct3: int, funct7: int
) -> str:
    return textwrap.dedent(f"""\
        // ===-- {class_name} TableGen Definition --===
        // File: lib/Target/RISCV/RISCVInstrInfoCustom.td

        def {class_name} : RVInst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
            "{mnemonic}", "$rd, $rs1, $rs2", [], InstFormatR> {{
          bits<5> rs2;
          bits<5> rs1;
          bits<5> rd;
          let Inst{{31-25}} = 0b{funct7:07b};  // funct7 = 0x{funct7:02X}
          let Inst{{24-20}} = rs2;
          let Inst{{19-15}} = rs1;
          let Inst{{14-12}} = 0b{funct3:03b};   // funct3 = 0x{funct3:01X}
          let Inst{{11-7}}  = rd;
          let Inst{{6-0}}   = 0b{opcode:07b};  // opcode = 0x{opcode:02X}
          let hasSideEffects = 0;
          let mayLoad = 1;
          let mayStore = 1;
        }}
    """)


def _tablegen_r4(
    class_name: str, mnemonic: str, opcode: int, funct3: int, funct7: int
) -> str:
    return textwrap.dedent(f"""\
        // ===-- {class_name} TableGen Definition (R4-type) --===
        // File: lib/Target/RISCV/RISCVInstrInfoCustom.td

        def {class_name} : RVInst<(outs GPR:$rd),
            (ins GPR:$rs1, GPR:$rs2, GPR:$rs3),
            "{mnemonic}", "$rd, $rs1, $rs2, $rs3", [], InstFormatR4> {{
          bits<5> rs3;
          bits<5> rs2;
          bits<5> rs1;
          bits<5> rd;
          let Inst{{31-27}} = rs3;
          let Inst{{26-25}} = 0b{(funct7 >> 5) & 0x3:02b};
          let Inst{{24-20}} = rs2;
          let Inst{{19-15}} = rs1;
          let Inst{{14-12}} = 0b{funct3:03b};
          let Inst{{11-7}}  = rd;
          let Inst{{6-0}}   = 0b{opcode:07b};
          let hasSideEffects = 0;
          let mayLoad = 1;
          let mayStore = 1;
        }}
    """)


def _spike_vdotacc(vec_len: int) -> str:
    return textwrap.dedent(f"""\
        // ===-- Spike ISA Extension: VDOTACC --===
        // File: riscv/insns/vdotacc.h
        //
        // Build: spike --extension=libvdotacc.so <elf>

        #include "spike/decode.h"
        #include <cmath>

        DEFINE_INSN(vdotacc) {{
            // Decode R4-type fields
            reg_t rs1_addr = insn.rs1();
            reg_t rs2_addr = insn.rs2();
            reg_t rs3_addr = insn.rs3();
            reg_t rd_addr  = insn.rd();

            int   n   = (int)RS3;           // number of elements
            float acc = (float)FRS(rd_addr); // existing accumulator

            if (n > {vec_len}) n = {vec_len}; // clamp to max

            for (int i = 0; i < n; i++) {{
                float a = p->get_mem<float>(RS1 + i * sizeof(float));
                float b = p->get_mem<float>(RS2 + i * sizeof(float));
                acc += a * b;
            }}
            WRITE_FRD(f32(acc));
        }}
    """)


def _spike_mmtile(M: int, N: int, K: int) -> str:
    return textwrap.dedent(f"""\
        // ===-- Spike ISA Extension: MMTILE --===
        // File: riscv/insns/mmtile.h

        #include "spike/decode.h"

        #define TILE 8

        DEFINE_INSN(mmtile) {{
            reg_t rd_addr  = insn.rd();
            reg_t rs1_addr = insn.rs1();
            reg_t rs2_addr = insn.rs2();

            reg_t C_base = RD;
            reg_t A_base = RS1;
            reg_t B_base = RS2;

            float C[TILE][TILE];
            // Load C tile
            for (int i=0; i<TILE; i++)
              for (int j=0; j<TILE; j++)
                C[i][j] = p->get_mem<float>(C_base + (i*TILE+j)*4);

            // Accumulate A × B into C
            for (int i=0; i<TILE; i++)
              for (int k=0; k<{K}; k++) {{
                float a = p->get_mem<float>(A_base + (i*{K}+k)*4);
                for (int j=0; j<TILE; j++) {{
                  float b = p->get_mem<float>(B_base + (k*TILE+j)*4);
                  C[i][j] += a * b;
                }}
              }}

            // Store C tile
            for (int i=0; i<TILE; i++)
              for (int j=0; j<TILE; j++)
                p->set_mem<float>(C_base + (i*TILE+j)*4, C[i][j]);
        }}
    """)


def _spike_sfmax() -> str:
    return textwrap.dedent("""\
        // ===-- Spike ISA Extension: SFMAX (Fused Softmax) --===
        // File: riscv/insns/sfmax.h

        #include "spike/decode.h"
        #include <cmath>
        #include <limits>

        DEFINE_INSN(sfmax) {
            reg_t out_ptr = RD;
            reg_t in_ptr  = RS1;
            int   n       = (int)RS2;

            std::vector<float> x(n), y(n);
            for (int i=0; i<n; i++)
              x[i] = p->get_mem<float>(in_ptr + i*4);

            // Pass 1: max for numerical stability
            float m = -std::numeric_limits<float>::infinity();
            for (int i=0; i<n; i++) m = std::max(m, x[i]);

            // Pass 2: exp + sum
            float s = 0.0f;
            for (int i=0; i<n; i++) { y[i] = std::exp(x[i] - m); s += y[i]; }

            // Pass 3: normalize
            for (int i=0; i<n; i++) {
              y[i] /= s;
              p->set_mem<float>(out_ptr + i*4, y[i]);
            }
        }
    """)


def _spike_sdpa(B: int, H: int, S: int, D: int) -> str:
    return textwrap.dedent(f"""\
        // ===-- Spike ISA Extension: SDPA (Scaled Dot-Product Attention) --===
        // File: riscv/insns/sdpa.h
        // Shape: B={B}, H={H}, S={S}, D={D}

        #include "spike/decode.h"
        #include <cmath>
        #include <vector>

        DEFINE_INSN(sdpa) {{
            reg_t O_ptr = RD;
            reg_t Q_ptr = RS1;
            reg_t K_ptr = RS2;
            reg_t V_ptr = RS3;

            const int B={B}, H={H}, S={S}, D={D};
            const float scale = 1.0f / std::sqrt((float)D);
            const size_t head_stride = S * D * sizeof(float);

            for (int b=0; b<B; b++) for (int h=0; h<H; h++) {{
                size_t off = (b*H+h)*S*D;
                // Compute S_tile = Q * K^T * scale
                std::vector<float> scores(S*S, 0.0f);
                for (int i=0; i<S; i++) for (int k=0; k<D; k++) {{
                    float q = p->get_mem<float>(Q_ptr+(off+i*D+k)*4);
                    for (int j=0; j<S; j++) {{
                        float kv = p->get_mem<float>(K_ptr+(off+j*D+k)*4);
                        scores[i*S+j] += q * kv;
                    }}
                }}
                // Softmax over rows
                for (int i=0; i<S; i++) {{
                    float m=-1e38f, s=0;
                    for (int j=0; j<S; j++) m=std::max(m,scores[i*S+j]);
                    for (int j=0; j<S; j++) {{ scores[i*S+j]=std::exp(scores[i*S+j]-m); s+=scores[i*S+j]; }}
                    for (int j=0; j<S; j++) scores[i*S+j] /= s;
                }}
                // O = scores * V
                for (int i=0; i<S; i++) for (int d=0; d<D; d++) {{
                    float acc = 0;
                    for (int j=0; j<S; j++) {{
                        float v = p->get_mem<float>(V_ptr+(off+j*D+d)*4);
                        acc += scores[i*S+j]*v;
                    }}
                    p->set_mem<float>(O_ptr+(off+i*D+d)*4, acc*scale);
                }}
            }}
        }}
    """)


def _spike_lnorm(is_rms: bool, hidden: int) -> str:
    name = "rmsnorm" if is_rms else "lnorm"
    return textwrap.dedent(f"""\
        // ===-- Spike ISA Extension: {name.upper()} --===
        // File: riscv/insns/{name}.h

        #include "spike/decode.h"
        #include <cmath>

        DEFINE_INSN({name}) {{
            reg_t out_ptr    = RD;
            reg_t in_ptr     = RS1;
            reg_t weight_ptr = RS2;
            const int H = {hidden};
            const float eps = 1e-5f;

            std::vector<float> x(H), y(H), gamma(H);
            for (int i=0; i<H; i++) {{
                x[i]     = p->get_mem<float>(in_ptr     + i*4);
                gamma[i] = p->get_mem<float>(weight_ptr + i*4);
            }}

            {"// RMS Norm" if is_rms else "// Layer Norm"}
            {"float ms=0; for(int i=0;i<H;i++) ms+=x[i]*x[i]; ms/=H;" if is_rms
             else "float m=0; for(int i=0;i<H;i++) m+=x[i]; m/=H; float v=0; for(int i=0;i<H;i++) v+=(x[i]-m)*(x[i]-m); v/=H;"}
            for (int i=0; i<H; i++) {{
                {"y[i] = x[i] / sqrt(ms + eps) * gamma[i];" if is_rms
                 else "y[i] = (x[i]-m) / sqrt(v+eps) * gamma[i];"}
                p->set_mem<float>(out_ptr + i*4, y[i]);
            }}
        }}
    """)


def _spike_activation(is_gelu: bool) -> str:
    name = "fusact_gelu" if is_gelu else "fusact_silu"
    return textwrap.dedent(f"""\
        // ===-- Spike ISA Extension: {name.upper()} --===
        // File: riscv/insns/{name}.h

        #include "spike/decode.h"
        #include <cmath>

        DEFINE_INSN({name}) {{
            reg_t out_ptr = RD;
            reg_t in_ptr  = RS1;
            int   n       = (int)RS2;

            for (int i=0; i<n; i++) {{
                float x = p->get_mem<float>(in_ptr + i*4);
                float y;
                {"// GELU: x * 0.5 * (1 + tanh(0.7978*(x + 0.04471*x^3)))" if is_gelu else "// SiLU: x * sigmoid(x)"}
                {"float c = 0.7978f*(x+0.04471f*x*x*x); y = x*0.5f*(1.0f+std::tanh(c));" if is_gelu
                 else "y = x / (1.0f + std::exp(-x));"}
                p->set_mem<float>(out_ptr + i*4, y);
            }}
        }}
    """)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PatternRuleEngine:
    """
    Evaluates all registered PatternRules against a list of HotspotResults
    and returns one CustomInstruction per hotspot.

    Rules are evaluated in priority order. Each hotspot is matched against
    all rules; the first match wins.
    """

    def __init__(self) -> None:
        self._rules: List[PatternRule] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._rules = sorted([
            ScaledDotProductAttentionRule(),
            DotProductAccumulateRule(),
            MatMulTileRule(),
            SoftmaxRule(),
            LayerNormRule(),
            ActivationFusionRule(),
        ], key=lambda r: r.priority)

    def add_rule(self, rule: PatternRule) -> None:
        """Register a custom rule and re-sort by priority."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def propose_all(
        self, hotspots: List[HotspotResult]
    ) -> List[tuple]:
        """
        Run all rules against all hotspots.

        Returns
        -------
        list of (HotspotResult, CustomInstruction)
        """
        results = []
        for hotspot in hotspots:
            instruction = self._match_one(hotspot)
            if instruction:
                results.append((hotspot, instruction))
                logger.info(
                    "Hotspot %s → proposed instruction '%s' (%.1fx speedup)",
                    hotspot.node.op_type.name,
                    instruction.mnemonic,
                    instruction.speedup_model.estimated_speedup
                    if instruction.speedup_model
                    else 0,
                )
        return results

    def _match_one(self, hotspot: HotspotResult) -> Optional[CustomInstruction]:
        for rule in self._rules:
            try:
                if rule.matches(hotspot):
                    return rule.propose(hotspot)
            except Exception as e:
                logger.warning(
                    "Rule %s failed on %s: %s",
                    type(rule).__name__,
                    hotspot.node.op_type.name,
                    e,
                )
        logger.debug("No rule matched for op type: %s", hotspot.node.op_type.name)
        return None
