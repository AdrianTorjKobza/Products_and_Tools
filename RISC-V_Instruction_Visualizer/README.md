# RISC-V Instruction Visualizer

An interactive, browser-based RISC-V assembly simulator and educational debugger. Write assembly code, step through execution one instruction at a time, and watch register state update in real time; with optional AI-powered natural-language explanations powered by Claude.

---

## Features

### Assembly Editor
- Syntax-aware editor with full comment support (`#`)
- Active instruction highlighted in **yellow** as you step through code
- Highlight stays in sync when scrolling long programs
- Editable at any time; the simulator resets automatically on change

### Step-Through Execution
- **Step** — execute one instruction at a time
- **Step Back** — undo the last instruction and restore previous register state
- **Run All** — animate through all instructions with a 1-second delay between steps, so you can watch the machine evolve
- **Stop** — interrupt Run All mid-execution at any point
- **Reset** — return all registers to zero and restart from the top

### Register State Panel
Three register banks displayed side by side, with changed registers flashing **green** on each step:

| Bank | Registers | Notes |
|---|---|---|
| Integer | `x0`–`x31` | Shown with ABI aliases (`zero`, `ra`, `sp`, `a0`…) |
| Float | `f0`–`f15` | Single-precision (F extension) |
| Vector | `v0`–`v7` | 4 × 32-bit element lanes (VLEN=128, SEW=32) |

Non-zero vector elements are highlighted in **blue** for quick scanning.

### Display Format Toggle
Switch register values between three formats at any time:
- `HEX` — `0x0000000a`
- `DEC` — `10`
- `BIN` — `0b00000000…`

### Execution Log
- Every executed instruction is logged with its computed result
- Errors (unknown mnemonics, bad arguments) shown in red
- One-click **Clear** button

### AI-Powered Explanations (optional)
Paste an [Anthropic API key](https://console.anthropic.com/) into the key bar at the bottom. After each instruction executes, Claude generates a one-sentence plain-English explanation of what just happened and why it matters — ideal for learners.

---

## Getting Started

* Open the HTML file directly in your favorite browser.
* The entire application is a single self-contained HTML file.
* No build tools, no dependencies, no server required.

---

## Supported ISA Extensions

### Base Integer — RV32I / RV64I

| Instruction | Operation |
|---|---|
| `addi rd, rs1, imm` | `rd = rs1 + imm` |
| `add rd, rs1, rs2` | `rd = rs1 + rs2` |
| `sub rd, rs1, rs2` | `rd = rs1 - rs2` |
| `mul rd, rs1, rs2` | `rd = rs1 × rs2` |
| `and / or / xor` | Bitwise ops |
| `slli / srli / srai` | Logical and arithmetic shifts |
| `lui rd, imm` | Load upper immediate |
| `li rd, imm` | Pseudo: load immediate |
| `mv rd, rs` | Pseudo: register copy |
| `neg rd, rs` | Pseudo: negate |
| `nop` | No operation |

### Vector Extension — RVV (VLEN=128, SEW=32, VL=4)

| Instruction | Operation |
|---|---|
| `vsetvli / vsetvl` | Configure vector length and element width |
| `vadd.vv / vadd.vx / vadd.vi` | Vector add (vector, scalar, immediate) |
| `vsub.vv / vsub.vx` | Vector subtract |
| `vmul.vv / vmul.vx` | Vector multiply |
| `vand.vv / vor.vv / vxor.vv` | Vector bitwise ops |
| `vmv.v.x / vmv.v.i` | Broadcast scalar/immediate to all lanes |
| `vmv.x.s / vmv.s.x` | Move between vector element and scalar register |
| `vrsub.vi` | Reverse subtract with immediate |
| `vsll.vi / vsrl.vi` | Vector shift left/right by immediate |

### Float Extension — F/D (single-precision)

| Instruction | Operation |
|---|---|
| `fadd.s / fsub.s / fmul.s / fdiv.s` | Float arithmetic |
| `fsqrt.s` | Square root |
| `fmadd.s / fmsub.s` | Fused multiply-add/subtract |
| `fmv.w.x / fmv.x.w` | Bit-reinterpret between float and integer registers |
| `fcvt.w.s / fcvt.s.w` | Convert between float and integer |

---

## AI Explanations Setup

1. Go to [console.anthropic.com](https://console.anthropic.com/) and create an API key
2. Open the visualizer and paste the key (`sk-ant-...`) into the key bar at the top
3. Click **Save** - the status indicator turns green. Key validation is perfomed during runtime.
4. Step through or run your program - each instruction gets a plain-English explanation in the log

The key is stored only in memory for the current session and never sent anywhere other than the Anthropic API. Close the tab and it's gone.

---

## Project Structure

```
riscv-visualizer.html    # The entire application — single self-contained file
README.md                # This file
```

The application is intentionally kept as a single HTML file for maximum portability. No npm, no bundler, no runtime dependencies.

**Internal architecture (all within the HTML file):**

```
├── CSS                  # Dark theme, layout grid, register/vector styling
├── HTML                 # Editor pane, register pane, console pane
└── JavaScript
    ├── Simulator state  # x[], f[], v[] register banks
    ├── parseAsm()       # Tokeniser — strips comments, maps to line numbers
    ├── execInstr()      # Instruction dispatcher (switch on mnemonic)
    ├── stepForward()    # Execute + highlight + log
    ├── stepBack()       # History-based undo
    ├── runAll()         # Animated loop with 1s delay
    ├── buildOverlay()   # Line highlight rendering
    ├── renderRegs()     # Register panel renderer
    └── explainInstr()   # Anthropic API call for AI explanation
```

---

## Acknowledgements

- [RISC-V International](https://riscv.org/) - open ISA specification
- [RISC-V Vector Extension Specification](https://github.com/riscv/riscv-v-spec) - RVV reference
