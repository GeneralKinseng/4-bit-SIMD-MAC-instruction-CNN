# 4-bit SIMD MAC Instruction for TinyML on RISC-V

This repository supports the research project **"Hardware-Efficient Implementation of TinyML Algorithms: Accelerating 4-bit Quantized Neural Networks on RISC-V Architecture"**.

The current baseline runs a TensorFlow Lite Micro LeNet-style MNIST inference workload as a static RV32 ELF binary under gem5 syscall emulation. The baseline statistics are used to measure the software-only execution cost before implementing a future custom RISC-V `qvdot4` 4-bit SIMD MAC instruction and evaluating it on a CV32E40P-style flow.

## Project Layout

Recommended local layout:

```text
master_research_wa/
├── README.md
├── src/
│   ├── main.cc
│   ├── model_full.cc
│   └── run_inference.py
├── scripts/
│   └── parse_gem5_stats.py
├── tflite-micro/
└── out/
    ├── baseline_inference.elf
    └── m5out_baseline/
```

The `out/` directory is used for generated files and should not be committed.

## Prerequisites

Install or build the following tools before running the baseline.

### RISC-V GNU Toolchain

The baseline is compiled with the bare-metal RISC-V cross compiler:

```bash
riscv64-unknown-elf-g++ --version
```

The compiler must support RV32 with the `ilp32` ABI.

### gem5

Build gem5 with RISC-V support. The current workflow has been tested with gem5 25.x:

```bash
/home/kinseng/gem5/build/ALL/gem5.opt --version
```

### TensorFlow Lite Micro

TensorFlow Lite Micro should be available under:

```text
tflite-micro/
```

The RV32 static library should exist at:

```text
tflite-micro/gen/riscv32_generic_rv32im_default_gcc/lib/libtensorflow-microlite.a
```

You can check with:

```bash
find tflite-micro -name "libtensorflow-microlite.a"
```

## Build the Baseline ELF

Run all commands from the repository root:

```bash
mkdir -p out
```

Compile the TensorFlow Lite Micro baseline program:

```bash
riscv64-unknown-elf-g++ \
  -march=rv32im -mabi=ilp32 -O2 -DNDEBUG \
  -DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG \
  -fno-rtti -fno-exceptions -fno-threadsafe-statics \
  -I./tflite-micro \
  -I./tflite-micro/tensorflow \
  -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
  -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/ruy \
  src/main.cc \
  src/model_full.cc \
  tflite-micro/gen/riscv32_generic_rv32im_default_gcc/lib/libtensorflow-microlite.a \
  -lm -static \
  -o out/baseline_inference.elf
```

If your locally built TFLite Micro library was compiled with compressed-instruction support, you may use:

```bash
-march=rv32imc
```

instead of:

```bash
-march=rv32im
```

Verify that the output is a 32-bit RISC-V ELF:

```bash
file out/baseline_inference.elf
```

## Run gem5 Simulation

Run the RV32 baseline simulation:

```bash
/home/kinseng/gem5/build/ALL/gem5.opt \
  --outdir=out/m5out_baseline \
  src/run_inference.py \
  --binary out/baseline_inference.elf
```

Optional: capture the simulated program output into gem5's output directory:

```bash
/home/kinseng/gem5/build/ALL/gem5.opt \
  --outdir=out/m5out_baseline \
  --redirect-stdout \
  src/run_inference.py \
  --binary out/baseline_inference.elf
```

Important argument order:

- `--outdir` and `--redirect-stdout` are gem5 options and must appear before `src/run_inference.py`.
- `--binary` is a script option and must appear after `src/run_inference.py`.

## Generated Output Files

After a successful run, gem5 writes results to:

```text
out/m5out_baseline/
```

Important files include:

```text
stats.txt       # main performance statistics
config.ini      # gem5 configuration
config.json     # gem5 configuration in JSON format
citations.bib   # gem5 citation information
simout          # generated only when --redirect-stdout is used
```

The run is successful when the program output reaches:

```text
Checkpoint 5: Inference Complete!
```

## Extract Baseline Metrics

You can generate a CSV summary for the metrics via `scripts/parse_gem5_stats.py`:

```bash
python3 scripts/parse_gem5_stats.py out/m5out_baseline/stats.txt \
  --label baseline_lenet_rv32 \
  --output out/gem5_summary.csv
```

For a quick manual check:

```bash
grep -E "simInsts|simTicks|numCycles|cpi|ipc|demandMissRate" out/m5out_baseline/stats.txt
```

Key metrics for the thesis baseline include:

- `simInsts`
- `simTicks`
- `board.processor.cores.core.numCycles`
- `board.processor.cores.core.cpi`
- `board.processor.cores.core.ipc`
- L1 instruction/data cache miss rates
- L2 cache miss rate

## dot4 Baseline Comparison: gem5 vs CV32E40P

The `dot4_baseline.c` microbenchmark is the software-only reference for the future
`qvdot4` instruction. It computes a 4-element signed 4-bit dot product by unpacking
nibbles and accumulating with plain RV32IM instructions. The same source is run on two
platforms so that functional correctness and cycle cost can be separated.

| Aspect | gem5 (syscall emulation) | CV32E40P (core-v-verif) |
| --- | --- | --- |
| Platform | Generic RV32 `TimingSimpleCPU` board | CV32E40P RTL core in `tb_top_verilator` |
| Simulator | gem5 25.x, `gem5.opt` | Verilator |
| Purpose | Preliminary profiling and software validation | Official core cycle baseline |
| Correctness | PASS (all 7 deterministic vectors) | PASS (all 7 deterministic vectors) |
| Checksum | `-247872` | `-247872` |
| Iterations | `DOT4_ITERATIONS=100000` | `DOT4_ITERATIONS=100000` |
| Failures | `0` | `0` |
| Cycle metric | `stats.txt` / `numCycles` — not yet recorded | `mcycle` delta = `11300011` |
| Cycles per dot4 | not yet recorded | `113.000` |
| Termination | `RESULT PASS` | `EXIT SUCCESS` |

### Deterministic test vectors

Both platforms produce identical results for the fixed vectors:

```text
zero            0x00000000 * 0x11111111 =    0
all_ones        0x11111111 * 0x11111111 =    8
all_minus_one   0xFFFFFFFF * 0x11111111 =   -8
max_positive    0x77777777 * 0x11111111 =   56
min_negative    0x88888888 * 0x11111111 =  -64
min_squared     0x88888888 * 0x88888888 =  512
mixed           0x89ABCDEF * 0x01234567 =  -84
```

### CV32E40P cycle measurement

The CV32E40P run reads the `mcycle` CSR around the benchmark loop:

```text
MCYCLE start      = 39054
MCYCLE end        = 11339065
MCYCLE delta      = 11300011
CYCLES_PER_DOT4   = 113.000
TOP.tb_top_verilator @ 329536850: EXIT SUCCESS
```

That is 11,300,011 core cycles for 100,000 iterations, i.e. 113.000 cycles per scalar
dot4 operation. This is the number the `qvdot4` speedup will be measured against.

### Which platform is authoritative

The gem5 dot4 run was functionally PASS with the same `-247872` checksum, which confirms
that the nibble unpacking, sign extension and accumulation logic are correct and portable.
gem5 remains useful for preliminary profiling — it is fast to iterate on, and its cache and
memory statistics help locate hot spots before any RTL work. However, gem5's generic
`TimingSimpleCPU` model does not reproduce the CV32E40P pipeline, so its cycle counts are
indicative only. No gem5 `mcycle` or `stats.txt` cycle figures for the dot4 run are recorded
here yet, and they are intentionally left blank rather than estimated. **Final hardware
comparison must use the CV32E40P core-v-verif `mcycle` measurement, because CV32E40P is the
selected core for this project.**

### Rerunning the dot4 baseline

gem5 (preliminary profiling), using the same layout and script as the LeNet baseline:

```bash
riscv64-unknown-elf-gcc \
  -march=rv32im -mabi=ilp32 -O2 -DNDEBUG \
  -DDOT4_ITERATIONS=100000 \
  src/dot4_baseline.c \
  -lm -static \
  -o out/dot4_baseline.elf

/home/kinseng/gem5/build/ALL/gem5.opt \
  --outdir=out/m5out_dot4 \
  --redirect-stdout \
  src/run_inference.py \
  --binary out/dot4_baseline.elf

grep -E "simInsts|numCycles|cpi|ipc" out/m5out_dot4/stats.txt
```

CV32E40P (official baseline), from a local `core-v-verif` checkout with the program placed
in `cv32e40p/tests/programs/custom/dot4_baseline/`:

```bash
cd core-v-verif/cv32e40p/sim/uvmt
make test TEST=dot4_baseline SIMULATOR=verilator
```

The run is correct when the log reports `RESULT PASS`, `failures=0`, checksum `-247872`,
and the testbench ends with `EXIT SUCCESS`.

