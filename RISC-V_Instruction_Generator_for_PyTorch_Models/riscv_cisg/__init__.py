"""
RISC-V Custom Instruction Suggestion Generator (CISG)
======================================================
A hardware-software co-design tool that analyzes ML workloads,
identifies computational hotspots, and proposes custom RISC-V
ISA extensions to accelerate them.

Author: Generated for portfolio demonstration
License: MIT
"""

__version__ = "1.0.0"
__author__ = "RISC-V CISG"
__description__ = "Automated Custom RISC-V Instruction Generator for ML Workloads"

from riscv_cisg.pipeline import CISGPipeline

__all__ = ["CISGPipeline"]
