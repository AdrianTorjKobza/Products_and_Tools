"""
Example: Custom Pattern Rule
============================
Shows how to add your own pattern-matching rule to the CISG engine.

This example adds a rule for fused RoPE (Rotary Position Embedding),
a common operation in LLaMA-style transformers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import textwrap
from riscv_cisg.analyzer.op_graph import OpType
from riscv_cisg.analyzer.hotspot_detector import HotspotResult
from riscv_cisg.proposer.pattern_rules import PatternRule
from riscv_cisg.proposer.custom_instruction import (
    CustomInstruction, CustomOpcodeSpace, InstructionFormat,
    InstructionOperand, SpeedupModel,
)
from riscv_cisg.analyzer.op_graph import OpGraph, OpNode, TensorShape, DataType
from riscv_cisg.pipeline import CISGPipeline


class RoPEFusionRule(PatternRule):
    """
    Custom rule: detect element-wise MUL patterns that look like RoPE
    (Rotary Position Embedding) and propose a fused ROPE instruction.

    RoPE pattern:
        x_rot = x * cos + rotate_half(x) * sin
    →  FUSROPE rd, rs1, rs2  (fuses cos/sin scale + rotation)
    """
    priority = 25

    def matches(self, hotspot: HotspotResult) -> bool:
        # Match MUL ops with 3D tensors (B, S, D) — typical RoPE shape
        if hotspot.node.op_type != OpType.MUL:
            return False
        shapes = hotspot.node.input_shapes
        if not shapes:
            return False
        if len(shapes[0].dims) == 3:
            _, seq, dim = shapes[0].dims
            return dim >= 64  # at least 64-dim head
        return False

    def propose(self, hotspot: HotspotResult) -> CustomInstruction:
        inp = self._first_input(hotspot)
        if inp and len(inp.dims) == 3:
            B, S, D = inp.dims
        else:
            B, S, D = 1, 128, 64

        baseline_cycles = D * 6   # 2 MUL + 2 ADD + rotate overhead
        proposed_cycles = D // 4  # 4-wide SIMD, single pass

        return CustomInstruction(
            mnemonic="fusrope",
            description=(
                f"Fused Rotary Position Embedding (RoPE): "
                f"out[i] = x[i]*cos[i] + rotate_half(x)[i]*sin[i]. "
                f"Combines the dual-stream multiply-add of RoPE into one instruction "
                f"operating on D={D}-dimensional head vectors."
            ),
            target_op_type=hotspot.node.op_type.name,
            instruction_format=InstructionFormat.R4,
            opcode_space=CustomOpcodeSpace.CUSTOM_1,
            funct3=0x5,
            funct7=0x10,
            operands=[
                InstructionOperand("rd",  5, True, "x", "Output vector ptr"),
                InstructionOperand("rs1", 5, True, "x", "Input x vector ptr (head)"),
                InstructionOperand("rs2", 5, True, "x", "cos embedding ptr"),
                InstructionOperand("rs3", 5, True, "x", "sin embedding ptr"),
            ],
            asm_syntax="fusrope  rd, rs1, rs2, rs3",
            semantics_pseudocode=textwrap.dedent(f"""\
                // FUSROPE rd, rs1, rs2, rs3
                // x at Mem[rs1], cos at Mem[rs2], sin at Mem[rs3]; dim={D}
                float* x   = Mem[rs1]; float* out = Mem[rd];
                float* cos = Mem[rs2]; float* sin = Mem[rs3];
                // rotate_half: swap halves and negate first half
                for (int i = 0; i < {D//2}; i++) {{
                    out[i]       = x[i]*cos[i]       - x[i+{D//2}]*sin[i];
                    out[i+{D//2}] = x[i+{D//2}]*cos[i+{D//2}] + x[i]*sin[i+{D//2}];
                }}
            """),
            speedup_model=SpeedupModel(
                baseline_ops=D * 6,
                proposed_ops=D,
                baseline_cycles=baseline_cycles,
                proposed_cycles=proposed_cycles,
                memory_reduction_factor=0.5,
                notes=f"RoPE fusion, head_dim={D}, 4-wide SIMD",
            ),
            tablegen_snippet=(
                f"// FUSROPE TableGen — see RISCVInstrInfoCustom.td\n"
                f"// def FUSROPE : RVInstR4Custom<0x1, 0x5, 0x2B, ...>"
            ),
            spike_extension_snippet=textwrap.dedent(f"""\
                DEFINE_INSN(fusrope) {{
                    reg_t out = RD; reg_t x_ptr = RS1;
                    reg_t cos_ptr = RS2; reg_t sin_ptr = RS3;
                    const int D = {D};
                    for (int i = 0; i < D/2; i++) {{
                        float xi  = p->get_mem<float>(x_ptr + i*4);
                        float xih = p->get_mem<float>(x_ptr + (i+D/2)*4);
                        float c   = p->get_mem<float>(cos_ptr + i*4);
                        float s   = p->get_mem<float>(sin_ptr + i*4);
                        p->set_mem<float>(out + i*4,       xi*c - xih*s);
                        p->set_mem<float>(out + (i+D/2)*4, xih*c + xi*s);
                    }}
                }}
            """),
            rationale=hotspot.acceleration_rationale,
        )


def main():
    print("CISG Example: Custom Rule (RoPE Fusion)\n")

    # Build a small graph that has MUL ops with RoPE-like shapes
    graph = OpGraph(name="LlamaAttentionLayer")

    fp32 = DataType.FP32
    def S(*dims): return TensorShape(dims=dims, dtype=fp32)

    # Add a MUL op with shape (1, 128, 64) — looks like RoPE
    rope_node = OpNode(
        node_id="rope_mul_0",
        op_type=OpType.MUL,
        input_shapes=[S(1, 128, 64), S(1, 128, 64)],
        output_shapes=[S(1, 128, 64)],
        flops=128 * 64 * 6,
        memory_bytes=128 * 64 * 4 * 3,
        profiled_time_us=200.0,
        source_framework="aten::mul",
    )
    graph.add_node(rope_node)

    # Also add a large matmul to give context
    mm_node = OpNode(
        node_id="qkv_proj",
        op_type=OpType.MATMUL,
        input_shapes=[S(1, 128, 4096), S(4096, 4096)],
        output_shapes=[S(1, 128, 4096)],
        flops=2 * 128 * 4096 * 4096,
        memory_bytes=(128*4096 + 4096*4096 + 128*4096) * 4,
        profiled_time_us=15000.0,
        source_framework="aten::linear",
    )
    graph.add_node(mm_node)

    # Run pipeline with the custom rule injected
    pipeline = CISGPipeline(
        output_dir="./cisg_output_custom_rule",
        top_n_hotspots=5,
        profile=False,
        verbose=True,
        extra_rules=[RoPEFusionRule()],
    )
    results = pipeline.run_from_graph(
        graph,
        workload_description="LLaMA-style attention with RoPE embeddings.",
    )
    print(results.summary())


if __name__ == "__main__":
    main()
