#!/usr/bin/env python3
"""
Parse gem5 stats.txt into a compact CSV row for thesis/result tables.

Example:
  python3 scripts/parse_gem5_stats.py out/m5out_baseline/stats.txt \
      --label baseline_lenet_rv32 \
      --output out/baseline_results.csv

For multiple runs:
  python3 scripts/parse_gem5_stats.py out/m5out_baseline_run*/stats.txt \
      --output out/baseline_results.csv
"""

import argparse
import csv
import math
import re
from pathlib import Path


METRICS = {
    "sim_seconds": "simSeconds",
    "sim_ticks": "simTicks",
    "sim_insts": "simInsts",
    "sim_ops": "simOps",
    "host_seconds": "hostSeconds",
    "host_inst_rate": "hostInstRate",
    "cpu_cycles": "board.processor.cores.core.numCycles",
    "cpu_cpi": "board.processor.cores.core.cpi",
    "cpu_ipc": "board.processor.cores.core.ipc",
    "l1d_accesses": "board.cache_hierarchy.l1d-cache-0.demandAccesses::total",
    "l1d_misses": "board.cache_hierarchy.l1d-cache-0.demandMisses::total",
    "l1d_miss_rate": "board.cache_hierarchy.l1d-cache-0.demandMissRate::total",
    "l1i_accesses": "board.cache_hierarchy.l1i-cache-0.demandAccesses::total",
    "l1i_misses": "board.cache_hierarchy.l1i-cache-0.demandMisses::total",
    "l1i_miss_rate": "board.cache_hierarchy.l1i-cache-0.demandMissRate::total",
    "l2_accesses": "board.cache_hierarchy.l2-cache-0.demandAccesses::total",
    "l2_misses": "board.cache_hierarchy.l2-cache-0.demandMisses::total",
    "l2_miss_rate": "board.cache_hierarchy.l2-cache-0.demandMissRate::total",
}


def parse_value(text):
    """Convert gem5 stat value text into int/float/string."""
    if text.lower() == "nan":
        return ""
    if text.lower() == "inf":
        return "inf"
    try:
        value = float(text)
    except ValueError:
        return text
    if math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def parse_stats(path):
    stats = {}
    line_re = re.compile(r"^(\S+)\s+([^\s#]+)")

    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-"):
                continue
            match = line_re.match(line)
            if not match:
                continue
            name, value = match.groups()
            stats[name] = parse_value(value)

    return stats


def make_row(stats_path, label):
    stats = parse_stats(stats_path)
    row = {
        "label": label,
        "stats_file": str(stats_path),
    }

    for csv_name, gem5_name in METRICS.items():
        row[csv_name] = stats.get(gem5_name, "")

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Extract selected gem5 stats.txt metrics into CSV."
    )
    parser.add_argument(
        "stats_files",
        nargs="+",
        help="One or more gem5 stats.txt files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="gem5_summary.csv",
        help="Output CSV path. Default: gem5_summary.csv",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label for a single run. For multiple files, directory names are used.",
    )
    args = parser.parse_args()

    rows = []
    for stats_file in args.stats_files:
        stats_path = Path(stats_file)
        if len(args.stats_files) == 1 and args.label:
            label = args.label
        else:
            label = stats_path.parent.name
        rows.append(make_row(stats_path, label))

    fieldnames = ["label", "stats_file"] + list(METRICS.keys())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {output_path}")


if __name__ == "__main__":
    main()
