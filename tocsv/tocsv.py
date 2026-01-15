#!/usr/bin/env python3
"""

python3 tocsv.py --output fitarea.csv


"""

import argparse
import csv
from pathlib import Path
import sys
import re

import numpy as np

from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaIO import specfilewrapper as specfile
from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory

from xrf_analysis_pymca import DEFAULT_CFG


ELEMENTS = ["Si", "Al", "Fe", "Ca", "Mg", "P", "K", "Na", "Ti", "V", "Mn", "S"]


def load_config():
    config = ConfigDict.ConfigDict()
    config.readfp(DEFAULT_CFG.splitlines().__iter__())
    return config


def find_first_mca(path):
    sf = specfile.Specfile(path)
    for scan in range(len(sf)):
        if sf[scan].nbmca():
            return sf[scan].mca(1)
    return None


def fit_file(path, config):
    y = find_first_mca(path)
    if y is None:
        return None
    x = np.arange(y.size, dtype=float)
    cfg = ConfigDict.ConfigDict()
    cfg.update(config)
    cfg["fit"]["xmin"] = int(cfg["fit"].get("xmin", 0))
    cfg["fit"]["xmax"] = int(cfg["fit"].get("xmax", y.size - 1))
    mcaFit = ClassMcaTheory.ClassMcaTheory()
    cfg = mcaFit.configure(cfg)
    mcaFit.setData(x, y, xmin=cfg["fit"]["xmin"], xmax=cfg["fit"]["xmax"])
    mcaFit.estimate()
    _, result = mcaFit.startfit(digest=1)
    return result


def extract_fitareas(result):
    areas = {el: 0.0 for el in ELEMENTS}
    for group in result.get("groups", []):
        parts = group.split()
        if not parts:
            continue
        el = parts[0]
        if el in areas:
            g = result.get(group, {})
            areas[el] += float(g.get("fitarea", 0.0))
    return areas


def extract_coal_id(filename):
    stem = Path(filename).stem
    match = re.search(r"_(\d+)", stem)
    return match.group(1) if match else stem


def main():
    parser = argparse.ArgumentParser(description="Batch XRF fitarea export using PyMca")
    parser.add_argument(
        "--input-dir",
        default="Data/Original_data",
        help="Directory containing .mca files",
    )
    parser.add_argument(
        "--output",
        default="fitarea.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    files = sorted(input_dir.glob("*.mca"))
    if not files:
        print("No .mca files found.", file=sys.stderr)
        return 1

    config = load_config()

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ELEMENTS + ["File"])
        total = len(files)
        for idx, path in enumerate(files, start=1):
            result = fit_file(str(path), config)
            if result is None:
                row = [""] * len(ELEMENTS) + [path.name]
                w.writerow(row)
            else:
                areas = extract_fitareas(result)
                coal_id = extract_coal_id(path.name)
                row = [f"{areas[el]:.6f}".rstrip("0").rstrip(".") for el in ELEMENTS] + [coal_id]
                w.writerow(row)
            remaining = total - idx
            print(f"Processed {idx}/{total}: {path.name} (remaining {remaining})")

    print(f"Saved CSV to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
