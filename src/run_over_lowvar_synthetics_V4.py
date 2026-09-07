#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run pca_fdr_nested_cv_V4_tailFDR.py over the low-variance synthetic datasets.

Expected synthetic filenames:
    synthetic_lowvar_d0_seed01.csv
    synthetic_lowvar_d0p5_seed01.csv
    synthetic_lowvar_d1_seed01.csv
    synthetic_lowvar_d1p5_seed01.csv
    synthetic_lowvar_d2_seed01.csv
    ...

Default experiment:
    d in {0, 0.5, 1.0}
    seeds 1..10

Nested-CV protocol:
    outer = 5 repeats x 5 folds
    inner = 5 folds
    C grid = {0.001, 0.01, 0.1, 1, 10}

The stronger levels d=1.5 and d=2 are deliberately omitted because
d=1 already gives a strong, near-systematic rescue regime.

Each dataset gets its own output directory:
    <outdir>/<dataset_stem>/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def fmt_float(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def main():
    ap = argparse.ArgumentParser(
        description="Run V4 tail-only ANOVA/BH over low-variance synthetic datasets."
    )

    ap.add_argument("--indir", required=True,
                    help="Directory containing generated synthetic CSV files.")
    ap.add_argument(
        "--v4-script",
        default="pca_fdr_nested_cv_V4_tailFDR.py",
        help="Path to the definitive V4 tail-only ANOVA/BH script."
    )
    ap.add_argument("--outdir", default="results_lowvar_nested",
                    help="Root directory for nested-CV results.")
    ap.add_argument(
        "--signal-d", type=float, nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Signal d levels to run. Default: 0, 0.5, 1."
    )
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(1, 11)),
                    help="Seeds to run. Default: 1..10.")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use. Default: current interpreter.")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Continue if one V4 call fails.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing them.")

    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    v4_script = Path(args.v4_script).resolve()
    outroot = Path(args.outdir).resolve()

    if not indir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {indir}")
    if not v4_script.exists():
        raise FileNotFoundError(f"V4 script does not exist: {v4_script}")

    outroot.mkdir(parents=True, exist_ok=True)

    jobs = []
    missing = []

    for d in args.signal_d:
        dtag = fmt_float(d)
        for seed in args.seeds:
            fname = f"synthetic_lowvar_d{dtag}_seed{seed:02d}.csv"
            csv_path = indir / fname

            if not csv_path.exists():
                missing.append(csv_path)
                continue

            dataset_outdir = outroot / csv_path.stem

            cmd = [
                args.python,
                str(v4_script),
                "--synthetic-csv", str(csv_path),
                "--outdir", str(dataset_outdir),
                "--outer-splits", "5",
                "--outer-repeats", "5",
                "--inner-splits", "5",
                "--c-grid", "0.001", "0.01", "0.1", "1", "10",
            ]
            jobs.append((d, seed, csv_path, dataset_outdir, cmd))

    if missing:
        print("\nWARNING: missing expected files:")
        for p in missing:
            print("  ", p)

    if not jobs:
        raise RuntimeError("No runnable datasets found.")

    print("\nLow-variance synthetic nested-CV run")
    print("------------------------------------")
    print(f"Input directory : {indir}")
    print(f"V4 script       : {v4_script}")
    print(f"Output root     : {outroot}")
    print("Signal d levels : 0, 0.5, 1 (defaults)")
    print("Outer CV        : 5 repeats x 5 folds")
    print("Inner CV        : 5 folds")
    print("C grid          : 0.001, 0.01, 0.1, 1, 10")
    print(f"Datasets found  : {len(jobs)}")
    print(f"Missing files   : {len(missing)}")
    print()

    failures = []

    for i, (d, seed, csv_path, dataset_outdir, cmd) in enumerate(jobs, start=1):
        print("=" * 78)
        print(f"[{i:03d}/{len(jobs):03d}] d={d:g} seed={seed:02d} dataset={csv_path.name}")
        print(f"Output: {dataset_outdir}")
        print("Command:")
        print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
        print("=" * 78, flush=True)

        if args.dry_run:
            continue

        dataset_outdir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((d, seed, csv_path, exc.returncode))
            print(
                f"\nERROR: V4 failed for d={d:g}, seed={seed:02d} "
                f"(return code {exc.returncode}).",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                sys.exit(exc.returncode)

    print("\n" + "=" * 78)
    print("RUN OVER FINISHED")
    print(f"Completed jobs : {len(jobs) - len(failures)}")
    print(f"Failures       : {len(failures)}")

    if failures:
        print("\nFailed datasets:")
        for d, seed, csv_path, rc in failures:
            print(f"  d={d:g}, seed={seed:02d}, returncode={rc}, dataset={csv_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
