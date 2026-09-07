#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the tau-sensitivity experiment on the 13 remaining real datasets.

Assumes pca_fdr_sensitivity_tau_alpha.py is in the same directory.

Fixed sensitivity design:
    tau   = 0.80, 0.90, 0.95
    alpha = 0.05

Already completed and therefore omitted here:
    PC-GITA AWPE4
    clean1 / OpenML 40665
    Spambase
    UCI174 Parkinson
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


DATASETS = [
    ("Banknote", ["--dataset", "banknote"]),
    ("PC-GITA AWPE3", ["--dataset", "pcgita", "--signal", "AWPE", "--session", "3"]),
    ("PC-GITA F0WPE3", ["--dataset", "pcgita", "--signal", "F0WPE", "--session", "3"]),
    ("PC-GITA F0WPE4", ["--dataset", "pcgita", "--signal", "F0WPE", "--session", "4"]),
    ("Breast Cancer", ["--dataset", "generic"]),
    ("Ionosphere", ["--dataset", "ionosphere"]),
    ("OpenML 718", ["--openml-id", "718"]),
    ("OpenML 55", ["--openml-id", "55"]),
    ("Sonar", ["--dataset", "sonar"]),
    ("OpenML 851", ["--openml-id", "851"]),
    ("OpenML 1017", ["--openml-id", "1017"]),
    ("OpenML 828", ["--openml-id", "828"]),
    ("OpenML 1563", ["--openml-id", "1563"]),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        required=True,
        help="Root directory used by the PC-GITA loader."
    )
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument(
        "--outdir",
        default="results_tau_17datasets",
        help="Common output directory. Dataset filenames remain separate."
    )
    ap.add_argument(
        "--script",
        default=None,
        help="Path to pca_fdr_sensitivity_tau_alpha.py. "
             "Default: same directory as this runner."
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later datasets if one command fails."
    )
    return ap.parse_args()


def main():
    args = parse_args()

    here = Path(__file__).resolve().parent
    target_script = (
        Path(args.script).expanduser().resolve()
        if args.script
        else here / "pca_fdr_sensitivity_tau_alpha.py"
    )

    if not target_script.exists():
        raise FileNotFoundError(
            f"Cannot find sensitivity script: {target_script}"
        )

    common = [
        "--tau-grid", "0.80", "0.90", "0.95",
        "--alpha-grid", "0.05",
        "--n-jobs", str(args.n_jobs),
        "--outdir", args.outdir,
    ]

    total_start = time.time()
    failed = []

    print("=" * 80)
    print("Tau sensitivity battery: 13 remaining real datasets")
    print("tau = {0.80, 0.90, 0.95}; alpha_FDR = 0.05")
    print(f"n_jobs = {args.n_jobs}")
    print(f"outdir = {args.outdir}")
    print("=" * 80)

    for i, (name, ds_args) in enumerate(DATASETS, start=1):
        cmd = [sys.executable, str(target_script)] + ds_args + common

        if "pcgita" in ds_args:
            cmd += ["--data-root", args.data_root]

        print()
        print("#" * 80)
        print(f"[{i:02d}/{len(DATASETS)}] {name}")
        print("#" * 80)
        print("Command:")
        print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        print(flush=True)

        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start

        if result.returncode != 0:
            failed.append((name, result.returncode))
            print(
                f"\nERROR: {name} failed with return code "
                f"{result.returncode} after {elapsed/60:.1f} min."
            )
            if not args.continue_on_error:
                print("Stopping battery. Use --continue-on-error to keep going.")
                return result.returncode
        else:
            print(f"\nCompleted {name} in {elapsed/60:.1f} min.")

    total_elapsed = time.time() - total_start

    print()
    print("=" * 80)
    print("BATTERY FINISHED")
    print(f"Total elapsed: {total_elapsed/3600:.2f} h")
    if failed:
        print("Failed datasets:")
        for name, code in failed:
            print(f"  - {name}: return code {code}")
        return 1

    print("All 13 datasets completed successfully.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
