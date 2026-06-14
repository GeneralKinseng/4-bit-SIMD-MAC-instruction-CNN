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

