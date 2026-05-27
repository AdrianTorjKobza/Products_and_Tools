# LLVM RISC-V Backend Integration Guide

## Overview

CISG generates LLVM TableGen (`.td`) files that plug directly into the
LLVM RISC-V backend. This guide walks through building LLVM, integrating
the definitions, and verifying the output.

---

## Prerequisites

```bash
# LLVM 16+ with RISC-V target enabled
git clone https://github.com/llvm/llvm-project
cd llvm-project

mkdir build && cd build
cmake ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="RISCV" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -GNinja

ninja -j$(nproc) llc clang
```

---

## Integrating the Custom Instructions

### 1. Copy generated files

```bash
LLVM_RISCV=path/to/llvm-project/llvm/lib/Target/RISCV

cp cisg_output/tablegen/RISCVInstrInfoCustom.td      $LLVM_RISCV/
cp cisg_output/tablegen/RISCVCustomInstrPatterns.td  $LLVM_RISCV/
```

### 2. Include in the main instruction info file

Edit `$LLVM_RISCV/RISCVInstrInfo.td` and add at the bottom:

```tablegen
// CISG custom instructions
include "RISCVInstrInfoCustom.td"
```

### 3. Rebuild

```bash
cd llvm-project/build
ninja llc
```

### 4. Verify TableGen compiles

```bash
# Run TableGen directly to check for syntax errors
./bin/llvm-tblgen -I ../llvm/include -I ../llvm/lib/Target/RISCV \
  ../llvm/lib/Target/RISCV/RISCV.td -gen-instr-info > /dev/null
echo "TableGen OK: $?"
```

---

## Using the Instructions in Assembly

After integration, `llc` and the assembler recognize the new mnemonics:

```asm
# test_custom.s
.text
.global main
main:
    # mmtile: tiled 8×8 matrix multiply
    # mmtile rd, rs1, rs2
    mmtile  a0, a1, a2

    # sfmax: fused softmax
    # sfmax rd, rs1, rs2 (rs2 = length)
    sfmax   a3, a4, t0

    # lnorm: fused layer norm
    lnorm   a5, a6, a7

    ret
```

Assemble with the custom target features:

```bash
./bin/clang --target=riscv64-unknown-elf \
  -march=rv64imfd \
  -mllvm -riscv-enable-custom \
  -c test_custom.s -o test_custom.o

./bin/llvm-objdump -d test_custom.o
# Should show the custom mnemonic and encoding
```

---

## Defining LLVM Intrinsics (optional — for C/C++ codegen)

To allow the compiler to automatically emit custom instructions from C code,
define LLVM intrinsics and ISel patterns.

### 1. Add intrinsic to `IntrinsicsRISCV.td`

```tablegen
// In llvm/include/llvm/IR/IntrinsicsRISCV.td

let TargetPrefix = "riscv" in {
  // mmtile: tiled 8x8 matmul
  // Arguments: output_ptr, A_ptr, B_ptr
  def int_riscv_mmtile : Intrinsic<
    [llvm_ptr_ty],              // return: output ptr
    [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty],  // A ptr, B ptr, output ptr
    [IntrWriteMem, IntrReadMem]
  >;

  // sfmax: fused softmax
  def int_riscv_sfmax : Intrinsic<
    [llvm_ptr_ty],
    [llvm_ptr_ty, llvm_ptr_ty, llvm_i64_ty],  // out, in, length
    [IntrWriteMem, IntrReadMem]
  >;
}
```

### 2. Add ISel pattern to `RISCVCustomInstrPatterns.td`

```tablegen
// Uncomment and fill in the stubs from the generated file:

def : Pat<
  (int_riscv_mmtile GPR:$rd, GPR:$rs1, GPR:$rs2),
  (MMTILE GPR:$rd, GPR:$rs1, GPR:$rs2)
>;

def : Pat<
  (int_riscv_sfmax GPR:$rd, GPR:$rs1, GPR:$rs2),
  (SFMAX GPR:$rd, GPR:$rs1, GPR:$rs2)
>;
```

### 3. Use from C code

```c
// my_kernel.c
#include <stdint.h>

// Declare the intrinsic
void* __builtin_riscv_mmtile(void* out, void* a, void* b);

void matmul_8x8(float* C, float* A, float* B) {
    __builtin_riscv_mmtile(C, A, B);
}
```

Compile:
```bash
./bin/clang --target=riscv64-unknown-elf \
  -march=rv64imfd \
  -O2 -c my_kernel.c -o my_kernel.o

./bin/llvm-objdump -d my_kernel.o | grep -A2 "mmtile"
# 0: 2b ..  mmtile a0, a1, a2
```

---

## Encoding Verification

Verify the encoding matches your Spike extension:

```bash
# Encode mmtile a0(=10), a1(=11), a2(=12)
# opcode=0x2B, funct3=0x1, funct7=0x01
# Encoding: [funct7(7)][rs2(5)][rs1(5)][funct3(3)][rd(5)][opcode(7)]
python3 -c "
opcode = 0x2B  # custom-1
funct3 = 0x1
funct7 = 0x01
rd, rs1, rs2 = 10, 11, 12  # a0, a1, a2
enc = (funct7<<25)|(rs2<<20)|(rs1<<15)|(funct3<<12)|(rd<<7)|opcode
print(f'Encoding: 0x{enc:08X}')
print(f'Binary:   {enc:032b}')
"
# Encoding: 0x00C5956B
```

Cross-check that the byte sequence in your `.o` file matches this encoding.

---

## Troubleshooting

### `error: unknown token in expression` in TableGen
Usually a missing `include` or a typo in a register class name.
Check that `RISCVInstrFormats.td` is included before your custom file.

### `LLVM ERROR: Cannot select: ...`
The intrinsic is defined but no ISel pattern matches it.
Add the corresponding `def : Pat<...>` entry.

### Custom instruction emitted with wrong encoding
Compare the `let Inst{...}` field assignments in your TableGen definition
against the expected binary encoding above.
The bit ranges must be exact — off-by-one shifts are a common mistake.
