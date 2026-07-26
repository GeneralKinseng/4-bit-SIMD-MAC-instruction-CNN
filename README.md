# 4-bit SIMD MAC Instruction for TinyML on RISC-V

Software-only baselines for a planned custom RISC-V instruction `qvdot4`, a packed
signed 4-bit SIMD multiply-accumulate.

Two workloads:

- `src/main.cc` + `src/model_full.cc` — TensorFlow Lite Micro LeNet-style MNIST
  inference, built as a static RV32 ELF and run under gem5 syscall emulation.
- `src/dot4_baseline.c` — microbenchmark of the scalar `qvdot4` equivalent. Run on
  gem5 for correctness and on the CV32E40P RTL core for the official cycle count.

All commands are run from the repository root unless stated otherwise. `out/` holds
generated artifacts.

## Prerequisites

```bash
export RISCV_PREFIX=riscv64-unknown-elf-   # RV32 / ilp32 capable
export GEM5_ROOT=/path/to/gem5             # built with RISC-V support, gem5 25.x
export CORE_V_VERIF=/path/to/core-v-verif  # CV32E40P + Verilator flow
```

Also required:

- TensorFlow Lite Micro checked out at `tflite-micro/`, with the RV32 static library
  at `tflite-micro/gen/riscv32_generic_rv32im_default_gcc/lib/libtensorflow-microlite.a`.
- Python with `numpy` and a TFLite interpreter (`tflite-runtime`, `ai-edge-litert`, or
  `tensorflow`) for the reference validation.

## TFLM baseline on gem5

Build:

```bash
mkdir -p out
${RISCV_PREFIX}g++ \
  -march=rv32im -mabi=ilp32 -O2 -DNDEBUG \
  -DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG \
  -fno-rtti -fno-exceptions -fno-threadsafe-statics \
  -I./tflite-micro \
  -I./tflite-micro/tensorflow \
  -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
  -I./tflite-micro/tensorflow/lite/micro/tools/make/downloads/ruy \
  src/main.cc src/model_full.cc \
  tflite-micro/gen/riscv32_generic_rv32im_default_gcc/lib/libtensorflow-microlite.a \
  -lm -static -o out/baseline_inference.elf
```

Use `-march=rv32imc` if the TFLM library was built with compressed instructions.

Run:

```bash
$GEM5_ROOT/build/ALL/gem5.opt \
  --outdir=out/m5out_baseline --redirect-stdout \
  src/run_inference.py --binary out/baseline_inference.elf
```

gem5 options go before `src/run_inference.py`; `--binary` goes after it.

The run succeeded when the output reaches `Checkpoint 5: Inference Complete!`.
Results land in `out/m5out_baseline/` (`stats.txt`, `config.ini`, `simout`).

Extract metrics:

```bash
python3 scripts/parse_gem5_stats.py out/m5out_baseline/stats.txt \
  --label baseline_lenet_rv32 --output out/gem5_summary.csv
```

## Validate TFLite output reference

Compares the RV32 scores against a desktop TFLite interpreter running the same model
bytes extracted from `src/model_full.cc`, with the same zero-filled input.

```bash
python3 scripts/validate_tflite_reference.py --gem5-log out/m5out_baseline/simout.txt
```

Exit status 0 means PASS. Reference scores (zero input):
`19,39,97,246,11,292,19,15,203,46`.

## dot4 baseline on gem5

```bash
${RISCV_PREFIX}gcc -march=rv32im -mabi=ilp32 -O2 -DNDEBUG \
  -DDOT4_ITERATIONS=100000 \
  src/dot4_baseline.c -lm -static -o out/dot4_baseline.elf

$GEM5_ROOT/build/ALL/gem5.opt \
  --outdir=out/m5out_dot4_baseline --redirect-stdout \
  src/run_inference.py --binary out/dot4_baseline.elf
```

Expect `RESULT PASS`, `failures=0`, checksum `-247872`. gem5 uses a generic
`TimingSimpleCPU`, so it is used for correctness and profiling only — no cycle figure
for dot4 is claimed from gem5.

## dot4 baseline on CV32E40P

Place the program at `$CORE_V_VERIF/cv32e40p/tests/programs/custom/dot4_baseline/`,
then:

```bash
cd $CORE_V_VERIF/cv32e40p/sim/uvmt
make test TEST=dot4_baseline SIMULATOR=verilator
```

The benchmark reads the `mcycle` CSR around the loop. The run is correct when the log
reports `RESULT PASS`, `failures=0`, checksum `-247872`, and `EXIT SUCCESS`.

## Baseline results

CV32E40P (core-v-verif, Verilator) — the authoritative cycle baseline:

| Metric | Value |
| --- | --- |
| Iterations | 100000 |
| Checksum | -247872 |
| Failures | 0 |
| MCYCLE delta | 11300011 |
| Cycles per dot4 | 113.000 |
| Result | PASS |

gem5 dot4: PASS, same checksum `-247872`, 0 failures. Cycle counts not recorded.

Deterministic vectors, identical on both platforms:

```text
zero            0x00000000 * 0x11111111 =    0
all_ones        0x11111111 * 0x11111111 =    8
all_minus_one   0xFFFFFFFF * 0x11111111 =   -8
max_positive    0x77777777 * 0x11111111 =   56
min_negative    0x88888888 * 0x11111111 =  -64
min_squared     0x88888888 * 0x88888888 =  512
mixed           0x89ABCDEF * 0x01234567 =  -84
```

`qvdot4` speedup will be measured against the 113.000 cycles/dot4 CV32E40P figure.
