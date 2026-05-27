# Spike ISA Simulator Integration Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Spike](https://github.com/riscv-software-src/riscv-isa-sim) | ≥ 1.1.0 | ISA simulation |
| [riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain) | any | Compile test programs |
| GCC or Clang | ≥ 9 / ≥ 10 | Build the extension plugin |
| CMake | ≥ 3.16 | Build system |

---

## Step 1: Install Spike

```bash
# Clone and build Spike
git clone https://github.com/riscv-software-src/riscv-isa-sim
cd riscv-isa-sim

# Install device tree compiler if missing
sudo apt-get install -y device-tree-compiler  # Ubuntu/Debian
brew install dtc                               # macOS

mkdir build && cd build
../configure --prefix=$HOME/.local
make -j$(nproc)
make install

# Verify
spike --version
```

---

## Step 2: Build the CISG Extension

```bash
# Generate the extension (CISG does this automatically)
riscv-cisg analyze --workload transformer --output-dir ./cisg_output

# Build
cd cisg_output/spike_extension/
mkdir build && cd build
cmake .. -DSPIKE_ROOT=$HOME/.local
make -j$(nproc)

# You should now have: libcisg_extension.so
ls -la libcisg_extension.so
```

---

## Step 3: Write a Test Program

The generated `tests/` directory contains assembly tests for each instruction.
Here is an annotated example for `mmtile`:

```asm
# tests/test_mmtile.S
.section .data
# 8x8 matrix A (row-major, float32)
mat_a: .float 1,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0, ...
# 8x8 matrix B
mat_b: .float 1,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0, ...
# Output buffer
mat_c: .skip 256   # 8×8×4 bytes

.section .text
.global _start
_start:
    la   a0, mat_c        # rd  = output ptr
    la   a1, mat_a        # rs1 = A ptr
    la   a2, mat_b        # rs2 = B ptr

    # Emit mmtile using .insn directive:
    # .insn r opcode, funct3, funct7, rd, rs1, rs2
    .insn r 0x2B, 0x1, 0x01, a0, a1, a2

    # Exit via htif
    li   a7, 93
    li   a0, 0
    ecall
```

Build and run:
```bash
riscv64-unknown-elf-gcc \
  -march=rv64imfd -mabi=lp64d \
  -nostdlib -static \
  -o tests/test_mmtile.elf tests/test_mmtile.S

spike \
  --extension=./build/libcisg_extension.so \
  --isa=rv64imfd \
  tests/test_mmtile.elf
```

---

## Step 4: Verify Correctness

Spike's `--log-commits` flag shows every instruction retirement:

```bash
spike \
  --extension=./build/libcisg_extension.so \
  --log-commits \
  tests/test_mmtile.elf 2>&1 | grep -A3 "mmtile\|custom"
```

To verify output values, write the result to a known memory location and
read it back via the Spike HTIF interface, or use Spike's interactive debug mode:

```bash
spike -d \
  --extension=./build/libcisg_extension.so \
  tests/test_mmtile.elf

# In the Spike debug REPL:
(spike) until pc 0 <address_after_mmtile>
(spike) mem 0 <mat_c_address> 256   # dump 256 bytes
```

---

## Step 5: Measure Cycle Counts

Spike supports reading CSR registers (mcycle, minstret) from assembly:

```asm
# Read cycle count before instruction
csrr  t0, mcycle

# Execute custom instruction
.insn r 0x2B, 0x1, 0x01, a0, a1, a2

# Read cycle count after
csrr  t1, mcycle
sub   t2, t1, t0    # t2 = cycles consumed
```

---

## Step 6: Compare Baseline vs. Custom Instruction

Write two versions of the test:

**Baseline** (scalar loop):
```asm
baseline_mmtile:
    # Standard nested loop: 8×8×K FMA operations
    li   t0, 0          # i = 0
outer_loop:
    ...                 # standard matmul loop body
```

**Custom instruction** (single `mmtile`):
```asm
custom_mmtile:
    .insn r 0x2B, 0x1, 0x01, a0, a1, a2
```

Compare cycle counts to get empirical speedup.

---

## Troubleshooting

### `spike: Illegal instruction`
The extension was not loaded, or the encoding doesn't match. Check:
1. `--extension=./libcisg_extension.so` is present in the command
2. The `MATCH_*` and `MASK_*` macros in `extension.cc` match your TableGen encoding
3. The `funct3`/`funct7` values in `.insn r` match the instruction definition

### `undefined symbol: p->get_mem`
Spike header version mismatch. Ensure `SPIKE_ROOT` points to your actual Spike install:
```bash
find $HOME -name "decode.h" 2>/dev/null | grep spike
# Use the directory containing this file as SPIKE_ROOT
cmake .. -DSPIKE_ROOT=/path/to/spike/install
```

### `Cannot open shared library`
Add the build directory to `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
spike --extension=./build/libcisg_extension.so ...
```
