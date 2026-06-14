# 4-bit-SIMD-MAC-instruction-CNN

RISCV Lightweight 4-bit SIMD MAC instruction for LeNet5-MNIST.

## Build & Run

Run all commands from the repository root. Build artifacts live under `out/`.

```bash
# 1. Create the output directory
mkdir -p out

# 2. Compile the baseline workload to out/baseline_inference.elf
#    Build src/main.cc + src/model_full.cc with your RV32 cross-compiler /
#    TFLite-Micro toolchain and emit a static RV32 ELF at:
#        out/baseline_inference.elf

# 3. Run the gem5 RV32 simulation
/home/kinseng/gem5/build/ALL/gem5.opt \
    --outdir=out/m5out_baseline \
    src/run_inference.py \
    --binary out/baseline_inference.elf
```

### Options

- `--outdir` is a **gem5** option (placed before the script path) and controls
  where gem5 writes its output, e.g. `out/m5out_baseline`.
- `--binary` is a **script** option (placed after `src/run_inference.py`). It
  defaults to `out/baseline_inference.elf` and accepts any relative or absolute
  path. The script validates that the binary exists before starting.
- `--redirect-stdout` is an optional gem5 option to capture the simulated
  program's stdout, for example:

```bash
/home/kinseng/gem5/build/ALL/gem5.opt \
    --outdir=out/m5out_baseline \
    --redirect-stdout \
    src/run_inference.py \
    --binary out/baseline_inference.elf
```
