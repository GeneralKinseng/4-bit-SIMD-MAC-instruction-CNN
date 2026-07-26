#!/usr/bin/env python3
"""
Validate the gem5 / TFLite Micro baseline output against a desktop Python
TensorFlow Lite reference run of the very same model.

The model is not read from a .tflite file: it is extracted directly from the
embedded C array in src/model_full.cc, so the reference run is guaranteed to
use the exact bytes that were compiled into the RV32 ELF.

The reference interpreter is fed the same zero-filled input that src/main.cc
uses, and its output is scaled by 1000 and truncated the same way main.cc does
((int)(score * 1000.0f)) before being compared class by class.

Examples:
  # Compare against the known RV32 baseline scores (defaults)
  python3 scripts/validate_tflite_reference.py

  # Compare against a gem5 run log directly
  python3 scripts/validate_tflite_reference.py \
      --gem5-log out/m5out_baseline/simout.txt

  # Explicit scores and a tighter tolerance
  python3 scripts/validate_tflite_reference.py \
      --gem5-scores 19,39,97,246,11,292,19,15,203,46 \
      --tolerance-x1000 2

Exit status is 0 on PASS and 1 on FAIL.

Requirements:
  numpy, plus one TensorFlow Lite interpreter implementation. Install the
  smallest one that works on your machine:

    pip install tflite-runtime          # preferred, small wheel
    pip install ai-edge-litert          # successor of tflite-runtime
    pip install tensorflow              # fallback, uses tensorflow.lite

  On platforms without a tflite-runtime wheel (recent Python versions, macOS
  arm64), use ai-edge-litert or the full tensorflow package.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


DEFAULT_MODEL_CC = "src/model_full.cc"

# gem5 RV32 baseline output (zero-filled input), see README.
DEFAULT_GEM5_SCORES = [19, 39, 97, 246, 11, 292, 19, 15, 203, 46]

TFLITE_MAGIC = b"TFL3"

INSTALL_HINT = (
    "No TensorFlow Lite interpreter found. Install one of:\n"
    "  pip install tflite-runtime\n"
    "  pip install ai-edge-litert\n"
    "  pip install tensorflow"
)


def load_interpreter_class():
    """Return a TFLite Interpreter class from whichever backend is installed."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite_runtime"
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai_edge_litert"
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter
        return Interpreter, "tensorflow.lite"
    except ImportError:
        raise SystemExit(INSTALL_HINT)


def extract_model_bytes(model_cc):
    """Extract the TFLite flatbuffer from a C source file holding a byte array."""
    text = Path(model_cc).read_text()

    body = re.search(r"\[\s*\]\s*=\s*\{(.*?)\}\s*;", text, re.DOTALL)
    if body is None:
        raise SystemExit(f"{model_cc}: no C byte-array initializer found")

    data = bytes(int(b, 16) for b in re.findall(r"0[xX]([0-9a-fA-F]{1,2})", body.group(1)))
    if not data:
        raise SystemExit(f"{model_cc}: byte array is empty")

    declared = re.search(r"_len\s*=\s*(\d+)\s*;", text)
    if declared and int(declared.group(1)) != len(data):
        raise SystemExit(
            f"{model_cc}: extracted {len(data)} bytes but the file declares "
            f"{declared.group(1)}"
        )

    # The flatbuffer file identifier sits at offset 4 of a .tflite buffer.
    if data[4:8] != TFLITE_MAGIC:
        raise SystemExit(
            f"{model_cc}: extracted data is not a TFLite flatbuffer "
            f"(expected {TFLITE_MAGIC!r} at offset 4, got {data[4:8]!r})"
        )
    return data


def parse_gem5_log(path):
    """Read 'Class N score(x1000)=V' lines from a gem5 stdout/simout log."""
    pattern = re.compile(r"^Class\s+(\d+)\s+score\(x1000\)=(-?\d+)", re.MULTILINE)
    scores = {int(cls): int(value) for cls, value in pattern.findall(Path(path).read_text())}
    if not scores:
        raise SystemExit(f"{path}: no 'Class N score(x1000)=' lines found")
    missing = sorted(set(range(max(scores) + 1)) - scores.keys())
    if missing:
        raise SystemExit(f"{path}: missing scores for classes {missing}")
    return [scores[i] for i in sorted(scores)]


def parse_score_list(text):
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise SystemExit(f"--gem5-scores must be comma-separated integers, got {text!r}")


def run_reference(model_bytes):
    """Invoke the model on a zero-filled input and return the output vector."""
    interpreter_cls, backend = load_interpreter_class()
    interpreter = interpreter_cls(model_content=model_bytes)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    zeros = np.zeros(input_detail["shape"], dtype=input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], zeros)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"]).flatten()

    return output, backend, input_detail, output_detail


def main():
    parser = argparse.ArgumentParser(
        description="Compare gem5/TFLite Micro scores against a Python TFLite reference run."
    )
    parser.add_argument(
        "--model-cc",
        default=DEFAULT_MODEL_CC,
        help=f"C source file with the embedded model (default: {DEFAULT_MODEL_CC})",
    )
    parser.add_argument(
        "--gem5-scores",
        default=",".join(str(s) for s in DEFAULT_GEM5_SCORES),
        help="Comma-separated x1000 scores from gem5 (default: known RV32 baseline)",
    )
    parser.add_argument(
        "--gem5-log",
        help="gem5 stdout/simout file to read the scores from instead of --gem5-scores",
    )
    parser.add_argument(
        "--tolerance-x1000",
        type=int,
        default=2,
        help="Allowed absolute difference in x1000 units (default: 2)",
    )
    parser.add_argument(
        "--save-tflite",
        help="Also write the extracted flatbuffer to this path",
    )
    args = parser.parse_args()

    model_bytes = extract_model_bytes(args.model_cc)
    if args.save_tflite:
        Path(args.save_tflite).write_bytes(model_bytes)
        print(f"Extracted model written to {args.save_tflite}")

    gem5_scores = (
        parse_gem5_log(args.gem5_log) if args.gem5_log else parse_score_list(args.gem5_scores)
    )

    output, backend, input_detail, output_detail = run_reference(model_bytes)

    print(f"Model:      {args.model_cc} ({len(model_bytes)} bytes)")
    print(f"Backend:    {backend}")
    print(f"Input:      shape={[int(d) for d in input_detail['shape']]} "
          f"dtype={np.dtype(input_detail['dtype']).name} (zero-filled)")
    print(f"Output:     shape={[int(d) for d in output_detail['shape']]} "
          f"dtype={np.dtype(output_detail['dtype']).name}")
    print(f"Tolerance:  +/-{args.tolerance_x1000} (x1000 units)")
    print()

    if len(output) != len(gem5_scores):
        raise SystemExit(
            f"score count mismatch: reference has {len(output)} classes, "
            f"gem5 input has {len(gem5_scores)}"
        )

    # main.cc prints (int)(score * 1000.0f), which truncates toward zero.
    reference_x1000 = [int(value * 1000.0) for value in output]

    print(f"{'class':>5} {'gem5':>8} {'reference':>10} {'diff':>6}  status")
    failures = []
    for cls, (gem5, ref) in enumerate(zip(gem5_scores, reference_x1000)):
        diff = ref - gem5
        ok = abs(diff) <= args.tolerance_x1000
        if not ok:
            failures.append(cls)
        print(f"{cls:>5} {gem5:>8} {ref:>10} {diff:>+6}  {'ok' if ok else 'MISMATCH'}")

    gem5_top = max(range(len(gem5_scores)), key=gem5_scores.__getitem__)
    ref_top = max(range(len(reference_x1000)), key=reference_x1000.__getitem__)
    print()
    print(f"Top class:  gem5={gem5_top} reference={ref_top}")

    if failures or gem5_top != ref_top:
        reason = []
        if failures:
            reason.append(f"classes {failures} outside tolerance")
        if gem5_top != ref_top:
            reason.append("top-1 class differs")
        print(f"FAIL: {'; '.join(reason)}")
        return 1

    print("PASS: gem5 output matches the Python TFLite reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
