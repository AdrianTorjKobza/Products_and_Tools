"""
CustomInstruction
=================
Data model representing a proposed RISC-V custom instruction extension.

RISC-V reserves four custom opcode spaces:
  custom-0: opcode 0x0B  (funct3 / funct7 available)
  custom-1: opcode 0x2B
  custom-2: opcode 0x5B
  custom-3: opcode 0x7B

Each proposed instruction includes:
  - Mnemonic and semantic description
  - Encoding (opcode, funct3, funct7)
  - Operand specification
  - Assembly syntax
  - Estimated speedup model
  - TableGen snippet for LLVM backend
  - Spike C++ extension snippet
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class InstructionFormat(Enum):
    """RISC-V instruction format families."""
    R = "R"        # Register-Register
    R4 = "R4"      # 4-register (fused ops)
    I = "I"        # Immediate
    S = "S"        # Store
    B = "B"        # Branch
    U = "U"        # Upper immediate
    J = "J"        # Jump
    VTYPE = "V"    # Vector (RVV-style)
    CUSTOM = "C"   # Fully custom encoding


class CustomOpcodeSpace(Enum):
    """RISC-V reserved custom opcode spaces."""
    CUSTOM_0 = 0x0B  # 0b0001011
    CUSTOM_1 = 0x2B  # 0b0101011
    CUSTOM_2 = 0x5B  # 0b1011011
    CUSTOM_3 = 0x7B  # 0b1111011


@dataclass
class InstructionOperand:
    """Describes a single operand of a custom instruction."""
    name: str           # e.g., "rd", "rs1", "rs2"
    bits: int           # width in bits
    is_register: bool   # True = register file, False = immediate
    reg_file: str = "x"  # "x" = integer, "f" = float, "v" = vector
    description: str = ""


@dataclass
class SpeedupModel:
    """
    Analytical speedup estimate for a custom instruction.

    speedup = baseline_cycles / proposed_cycles
    """
    baseline_ops: int          # Operations in baseline (scalar/vector loop)
    proposed_ops: int          # Operations with custom instruction
    baseline_cycles: int       # Estimated baseline cycles
    proposed_cycles: int       # Estimated cycles with custom instruction
    memory_reduction_factor: float = 1.0  # <1.0 = fewer memory accesses
    notes: str = ""

    @property
    def estimated_speedup(self) -> float:
        if self.proposed_cycles == 0:
            return float("inf")
        return self.baseline_cycles / self.proposed_cycles

    @property
    def meets_10x_target(self) -> bool:
        return self.estimated_speedup >= 10.0

    def __str__(self) -> str:
        return (
            f"Speedup: {self.estimated_speedup:.1f}x  "
            f"(baseline={self.baseline_cycles} cycles → "
            f"proposed={self.proposed_cycles} cycles)"
        )


@dataclass
class CustomInstruction:
    """
    A complete specification of a proposed custom RISC-V instruction.

    Attributes
    ----------
    mnemonic : str
        Assembly mnemonic (e.g., "vdotacc").
    description : str
        Natural language description of what the instruction does.
    target_op_type : str
        The OpType name this instruction accelerates.
    instruction_format : InstructionFormat
    opcode_space : CustomOpcodeSpace
    funct3 : int
        3-bit function code.
    funct7 : int
        7-bit function code (R-type only).
    operands : list of InstructionOperand
    asm_syntax : str
        Assembly syntax string.
    semantics_pseudocode : str
        Pseudocode describing the operation.
    speedup_model : SpeedupModel
    tablegen_snippet : str
        LLVM TableGen definition for this instruction.
    spike_extension_snippet : str
        C++ code for Spike ISA simulator extension.
    fusion_opportunity : bool
        Whether this instruction is part of a fusion group.
    fusion_partners : list of str
        Mnemonics of instructions this fuses with.
    """
    mnemonic: str
    description: str
    target_op_type: str
    instruction_format: InstructionFormat = InstructionFormat.R
    opcode_space: CustomOpcodeSpace = CustomOpcodeSpace.CUSTOM_0
    funct3: int = 0
    funct7: int = 0
    operands: List[InstructionOperand] = field(default_factory=list)
    asm_syntax: str = ""
    semantics_pseudocode: str = ""
    speedup_model: Optional[SpeedupModel] = None
    tablegen_snippet: str = ""
    spike_extension_snippet: str = ""
    fusion_opportunity: bool = False
    fusion_partners: List[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def opcode_hex(self) -> str:
        return f"0x{self.opcode_space.value:02X}"

    @property
    def encoding_summary(self) -> str:
        return (
            f"opcode={self.opcode_hex} "
            f"funct3=0x{self.funct3:01X} "
            f"funct7=0x{self.funct7:02X}"
        )

    def to_dict(self) -> dict:
        d = {
            "mnemonic": self.mnemonic,
            "description": self.description,
            "target_op_type": self.target_op_type,
            "format": self.instruction_format.value,
            "encoding": {
                "opcode_space": self.opcode_space.name,
                "opcode_hex": self.opcode_hex,
                "funct3": self.funct3,
                "funct7": self.funct7,
            },
            "asm_syntax": self.asm_syntax,
            "semantics": self.semantics_pseudocode,
            "operands": [
                {
                    "name": op.name,
                    "bits": op.bits,
                    "is_register": op.is_register,
                    "reg_file": op.reg_file,
                    "description": op.description,
                }
                for op in self.operands
            ],
            "rationale": self.rationale,
        }
        if self.speedup_model:
            d["speedup"] = {
                "estimated_speedup": round(self.speedup_model.estimated_speedup, 2),
                "baseline_cycles": self.speedup_model.baseline_cycles,
                "proposed_cycles": self.speedup_model.proposed_cycles,
                "meets_10x_target": self.speedup_model.meets_10x_target,
                "notes": self.speedup_model.notes,
            }
        if self.fusion_opportunity:
            d["fusion"] = {
                "partners": self.fusion_partners,
            }
        return d

    def __repr__(self) -> str:
        speedup = (
            f"{self.speedup_model.estimated_speedup:.1f}x"
            if self.speedup_model
            else "N/A"
        )
        return (
            f"CustomInstruction(mnemonic='{self.mnemonic}', "
            f"target='{self.target_op_type}', "
            f"speedup={speedup})"
        )
