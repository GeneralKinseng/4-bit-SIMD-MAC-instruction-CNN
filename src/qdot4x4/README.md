# `qdot4x4`: RI-MAC-Inspired 4-Lane Unit vs. `qvdot4` on CV32E40P

This directory contains an independent clean-room SystemVerilog reimplementation of a 4-lane signed 4-bit dot-product unit (`qdot4x4`), inspired by the 4-bit C-MUL44 mode of RI-MAC ([Longchar et al., AIMLSystems 2025](https://doi.org/10.1109/AIMLSYSTEMS67835.2025.11330892)). It is **not** the authors' RTL. The unit is integrated alongside the 8-lane `qvdot4` unit in OpenHW [CV32E40P](https://github.com/openhwgroup/cv32e40p) (commit `fcd5968`) and benchmarked on LeNet-5 in [core-v-verif](https://github.com/openhwgroup/core-v-verif) with Verilator.

## Directory Contents

- `rtl/cv32e40p_qdot4x4.sv` — Combinational 4-lane signed 4-bit dot-product unit using RI-MAC's 2x2 partial-product decomposition on `operand[15:0]`.
- `rtl/cv32e40p_qvdot4.sv` — Existing 8-lane signed 4-bit dot-product unit.
- `tb/tb_dot_units.sv`, `tb/tb_dot_units.cpp` — Standalone Verilator unit testbench for both units (18,794,633 checks).
- `cv32e40p_fcd5968_qvdot4_qdot4x4.patch` — Self-contained git patch adding both units to a clean CV32E40P (`fcd5968`) checkout.
- `scripts/apply_cv32e40p_integration.py` — Anchor-checked Python alternative to applying the patch.
- `sw/lenet5-qdot4x4.c` — Three-way bare-metal LeNet-5 benchmark (`SW` vs. `qvdot4` vs. paired `2x qdot4x4`) with per-layer, packing, and end-to-end cycle timers.

## Instruction Encodings

Both instructions use RISC-V `custom-0` (`opcode = 7'h0b`, R-type) and execute in a single combinational EX cycle:

| Unit | Assembly Encoding | Operands Used | Lanes | Result |
| --- | --- | --- | --- | --- |
| `qvdot4` | `.insn r 0x0b, 0, 0, rd, rs1, rs2` | `rs1[31:0]`, `rs2[31:0]` | 8 x signed int4 | Sign-extended 32-bit sum |
| `qdot4x4` | `.insn r 0x0b, 1, 1, rd, rs1, rs2` | `rs1[15:0]`, `rs2[15:0]` | 4 x signed int4 | Sign-extended 32-bit sum |

- `qdot4x4` reuses RI-MAC's C-MUL44 function fields (`funct3 = 001`, `funct7 = 0000001`).
- Computing an 8-lane dot product with `qdot4x4` (`Q4X2` mode) executes `qdot4x4(a, b) + qdot4x4(a >> 16, b >> 16)`.

## Reproduction Guide

All commands below assume you are inside `src/qdot4x4/` unless stated otherwise.

### 1. Run the Standalone Verilator Unit Test

Requires Verilator (`verilator --version`):

```bash
verilator --cc --exe --build -O2 -Wall -Wno-UNUSEDSIGNAL \
  -Mdir /tmp/obj_unit --top-module tb_dot_units \
  rtl/cv32e40p_qvdot4.sv rtl/cv32e40p_qdot4x4.sv \
  tb/tb_dot_units.sv tb/tb_dot_units.cpp -o tb_unit
/tmp/obj_unit/tb_unit
```

Expected result: `checks=18794633 failures=0` and `UNIT RESULT PASS` (includes the RI-MAC Fig. 7 vector `0xe583 . 0xc937 = -30`, max `+256`, min `-224`, exhaustive per-lane tests, and 2M random 32-bit equivalence checks verifying `lo + hi == qvdot4`).

### 2. Apply the CV32E40P Integration Patch

Start from a clean CV32E40P checkout at commit `fcd5968` inside `core-v-verif` (note: if your checkout already contains `qvdot4` edits, stash them or create a fresh branch from `fcd5968` first):

```bash
cd $CORE_V_VERIF/core-v-cores/cv32e40p
git apply --check /path/to/repo/src/qdot4x4/cv32e40p_fcd5968_qvdot4_qdot4x4.patch
git apply /path/to/repo/src/qdot4x4/cv32e40p_fcd5968_qvdot4_qdot4x4.patch
```

(Alternatively, run `python3 scripts/apply_cv32e40p_integration.py $CORE_V_VERIF/core-v-cores/cv32e40p rtl`.)

### 3. Build and Run the Three-Way LeNet-5 Benchmark

1. Copy the benchmark into `core-v-verif` and create `test.yaml`:

```bash
mkdir -p $CORE_V_VERIF/cv32e40p/tests/programs/custom/lenet5-qdot4x4
cp sw/lenet5-qdot4x4.c $CORE_V_VERIF/cv32e40p/tests/programs/custom/lenet5-qdot4x4/lenet5-qdot4x4.c
cat > $CORE_V_VERIF/cv32e40p/tests/programs/custom/lenet5-qdot4x4/test.yaml << 'EOF'
name: lenet5-qdot4x4
uvm_test: uvmt_cv32e40p_firmware_test_c
description: >
    Three-way LeNet-5 benchmark comparing SW, qvdot4, and 2x qdot4x4.
EOF
```

2. Clean the Verilator build and run the simulation:

```bash
export CV_SW_TOOLCHAIN=/path/to/riscv-toolchain
export CV_SW_PREFIX=riscv64-unknown-elf-   # or riscv-none-elf- / riscv32-unknown-elf-
export CV_SW_MARCH=rv32imc_zicsr
export CV_SW_CC=gcc

cd $CORE_V_VERIF/cv32e40p/sim/core
make veri-clean
make veri-test TEST=lenet5-qdot4x4
```

Expected result: all 10 output logits match across `SW`, `Q8`, and `Q4X2` (`checksum=-588`), `RESULT PASS failures=0`, and `EXIT SUCCESS`.

## Reference Results (CV32E40P, Verilator)

Measured from the single three-way `lenet5-qdot4x4` ELF (`checksum = -588`, `failures = 0`):

| Stage | SW Cycles | Q8 (`qvdot4`) | Q4X2 (`2x qdot4x4`) | SW / Q8 | SW / Q4X2 | Q4X2 / Q8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Conv1` | `2,246,132` | `386,478` | `461,013` | `5.81x` | `4.87x` | `1.19x` |
| `Pool1` | `52,172` | `56,874` | `61,606` | `0.92x` | `0.85x` | `1.08x` |
| `Conv2` | `3,477,592` | `386,466` | `519,095` | `9.00x` | `6.70x` | `1.34x` |
| `Pool2` | `26,541` | `25,754` | `27,890` | `1.03x` | `0.95x` | `1.08x` |
| `FC1` | `682,914` | `71,379` | `101,932` | `9.57x` | `6.70x` | `1.43x` |
| `FC2` | `149,873` | `18,925` | `24,439` | `7.92x` | `6.13x` | `1.29x` |
| `FC3` | `13,758` | `1,330` | `1,939` | `10.34x` | `7.10x` | `1.46x` |
| **MAC Layers Subtotal** | **`6,570,269`** | **`864,578`** | **`1,108,418`** | **`7.60x`** | **`5.93x`** | **`1.28x`** |
| **Layer Sum (All 7 Layers)** | **`6,648,982`** | **`947,206`** | **`1,197,914`** | **`7.02x`** | **`5.55x`** | **`1.26x`** |
| `Pack` (`Conv1` + `Conv2`) | `916,419` | `916,419` | `916,417` | `1.00x` | `1.00x` | `1.00x` |
| **End-to-End Inference** | **`7,565,460`** | **`1,863,677`** | **`2,114,379`** | **`4.06x`** | **`3.58x`** | **`1.13x`** |

- Across the 56,586 packed 8-lane dot products in the MAC layers, `qvdot4` averages `15.28` cycles/dot4 vs. `19.59` cycles/dot4 for paired `qdot4x4` (`+4.31` cycles for the second instruction, two 16-bit shifts, and add).
- Unaccelerated software activation packing (`916,419` cycles) accounts for `49.2%` of total `qvdot4` inference time (`43.3%` for `Q4X2`).

## Yosys Gate-Count Summary (Yosys 0.52)

| Metric | `cv32e40p_qvdot4` (8-lane) | `cv32e40p_qdot4x4` (4-lane) | Ratio |
| --- | ---: | ---: | ---: |
| Primitive 2-input gates (`synth -flatten -noabc`) | `1,296` | `643` | `2.02x` |
| Primitive logic depth (`ltp`) | `30` levels | `31` levels | `0.97x` |
| NAND2 + NOT (`abc -fast -g NAND`) | `3,019` (`~2,584` GE) | `1,481` (`~1,256` GE) | `2.04x` |
| CMOS2 (`abc -fast -g cmos2`) | `3,855` | `1,676` | `2.30x` |

## Caveats

1. `cv32e40p_qdot4x4.sv` is an independent RI-MAC-inspired reimplementation, not the authors' RTL.
2. Compare `SW`, `Q8`, and `Q4X2` only within the same three-way ELF to avoid cross-binary compiler inlining variance.
3. Because `OPCODE_CUSTOM_0` shares `7'h0b` with `OPCODE_LOAD_POST`, the decoder patch applies only to `PULP_XPULP=0` configurations.
4. Verilator reports cycle counts and Yosys reports technology-independent gate counts; post-synthesis ASIC/FPGA timing and power remain future work.
