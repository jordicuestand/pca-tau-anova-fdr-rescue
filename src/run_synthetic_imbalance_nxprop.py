#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Executa seqüencialment la bateria factorial sintètica "
            "n x class-imbalance amb pca_fdr_nested_cv_V4_tailFDR.py."
        )
    )

    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("synthetic_imbalance_nxprop"),
    )
    ap.add_argument(
        "--script",
        type=Path,
        default=Path("pca_fdr_nested_cv_V4_tailFDR.py"),
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_synthetic_imbalance_nxprop"),
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=10)
    ap.add_argument(
        "--c-grid",
        nargs="+",
        default=["0.001", "0.01", "0.1", "1", "10"],
    )
    ap.add_argument(
        "--pattern",
        default="synthetic_n*_p100_b*_seed*.csv",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")

    return ap.parse_args()


def build_command(csv_path: Path, outdir: Path, args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.script),
        "--synthetic-csv", str(csv_path),
        "--outer-splits", str(args.outer_splits),
        "--outer-repeats", str(args.outer_repeats),
        "--inner-splits", str(args.inner_splits),
        "--c-grid", *map(str, args.c_grid),
        "--n-jobs", str(args.n_jobs),
        "--outdir", str(outdir),
    ]


def main() -> int:
    args = parse_args()

    if not args.script.exists():
        print(
            f"ERROR: no existeix el script principal: {args.script}",
            file=sys.stderr,
        )
        return 2

    if not args.data_dir.exists():
        print(
            f"ERROR: no existeix el directori de dades: {args.data_dir}",
            file=sys.stderr,
        )
        return 2

    csv_files = sorted(args.data_dir.glob(args.pattern))

    if not csv_files:
        print(
            f"ERROR: no s'han trobat CSV amb el patró "
            f"{args.pattern!r} a {args.data_dir}",
            file=sys.stderr,
        )
        return 2

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("Experiment sintètic factorial: n x class imbalance")
    print("============================================================")
    print(f"Script principal : {args.script}")
    print(f"Dades             : {args.data_dir}")
    print(f"Resultats         : {args.results_dir}")
    print(f"Datasets trobats  : {len(csv_files)}")
    print(f"Outer folds       : {args.outer_splits}")
    print(f"Outer repeats     : {args.outer_repeats}")
    print(f"Inner folds       : {args.inner_splits}")
    print(f"C-grid            : {' '.join(map(str, args.c_grid))}")
    print(f"n_jobs            : {args.n_jobs}")
    print(f"Patró             : {args.pattern}")
    print("============================================================")
    print()

    failures = []

    for i, csv_path in enumerate(csv_files, start=1):
        outdir = args.results_dir / csv_path.stem
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(csv_path, outdir, args)

        print("------------------------------------------------------------")
        print(f"[{i:03d}/{len(csv_files):03d}] {csv_path.name}")
        print(f"Outdir       : {outdir}")
        print("Command:")
        print(" ".join(cmd))
        print("------------------------------------------------------------")

        if args.dry_run:
            print("DRY RUN: no s'executa.\n")
            continue

        completed = subprocess.run(cmd)

        if completed.returncode != 0:
            failures.append((csv_path, completed.returncode))
            print(
                f"\nERROR: {csv_path.name} ha acabat amb "
                f"codi {completed.returncode}.\n",
                file=sys.stderr,
            )

            if args.stop_on_error:
                print("S'atura la bateria per --stop-on-error.")
                return completed.returncode
        else:
            print(f"\nOK: {csv_path.name}\n")

    print("============================================================")

    if failures:
        print(f"Bateria acabada amb {len(failures)} error(s):")
        for path, code in failures:
            print(f"  - {path.name}: return code {code}")
        print("============================================================")
        return 1

    print("Bateria acabada correctament.")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
