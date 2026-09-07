#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic binary-classification datasets in which the true class
signal lies deliberately in a low-variance PCA direction.

The design is compatible with pipelines that STANDARDIZE FEATURES before PCA.

Core construction
-----------------
1) A nuisance block of strongly correlated variables creates dominant PCs
   carrying high variance but NO class information.

2) A pair of strongly positively correlated variables creates:
       - a high-variance "sum" direction
       - a very-low-variance "contrast" direction

   The binary class means differ ONLY along the low-variance contrast
   direction. Thus the Bayes-relevant direction is intentionally a tail PC.

3) Remaining variables are independent Gaussian noise.

The parameter --signal-d controls the *within-class Cohen's d* along the
true discriminative direction. This is preferable to a raw mean shift because
it keeps the discriminative direction low-variance even when the class signal
is strong.

Example theoretical signal block (rho_signal = 0.98):
    eigenvalue(sum)      = 1 + rho = 1.98
    eigenvalue(contrast) = 1 - rho = 0.02

The class shift is placed along contrast vector:
    v = (1, -1) / sqrt(2)

For a requested Cohen's d = d:
    mean difference along v = d * sqrt(1 - rho_signal)

Outputs
-------
For every generated dataset:
    synthetic_lowvar_d<...>_seed<...>.csv

Columns:
    y, x001, x002, ...

Also:
    synthetic_manifest.csv
    theoretical_pc_structure.csv

The manifest records the generator parameters and the exact true signal
direction in feature coordinates.

This script ONLY generates data. It does not run PCA, ANOVA/FDR, SVM or CV.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_within_covariance(p, nuisance_block_size, rho_nuisance, rho_signal):
    """
    Block diagonal covariance with unit marginal variances.

    Features:
      0 ... nuisance_block_size-1 : equicorrelated nuisance block
      nuisance_block_size,
      nuisance_block_size+1      : signal pair
      remaining                  : independent noise
    """
    if nuisance_block_size < 2:
        raise ValueError("nuisance_block_size must be >= 2")
    if nuisance_block_size + 2 > p:
        raise ValueError("Need at least nuisance_block_size + 2 total features.")
    if not (0 <= rho_nuisance < 1):
        raise ValueError("rho_nuisance must be in [0,1).")
    if not (0 <= rho_signal < 1):
        raise ValueError("rho_signal must be in [0,1).")

    Sigma = np.eye(p, dtype=float)

    # Equicorrelated nuisance block
    m = nuisance_block_size
    Sigma[:m, :m] = rho_nuisance
    np.fill_diagonal(Sigma[:m, :m], 1.0)

    # Highly correlated signal pair
    a, b = m, m + 1
    Sigma[a, b] = rho_signal
    Sigma[b, a] = rho_signal

    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() <= 0:
        raise ValueError(
            f"Within-class covariance is not positive definite; min eigenvalue={eigvals.min()}"
        )

    return Sigma


def true_signal_vector(p, nuisance_block_size):
    """
    Low-variance contrast direction of the signal pair:
        (x_a - x_b) / sqrt(2)
    """
    v = np.zeros(p, dtype=float)
    a, b = nuisance_block_size, nuisance_block_size + 1
    v[a] = 1.0 / np.sqrt(2.0)
    v[b] = -1.0 / np.sqrt(2.0)
    return v


def generate_dataset(
    n,
    p,
    minority_fraction,
    nuisance_block_size,
    rho_nuisance,
    rho_signal,
    signal_d,
    seed,
):
    rng = np.random.default_rng(seed)

    Sigma = build_within_covariance(
        p=p,
        nuisance_block_size=nuisance_block_size,
        rho_nuisance=rho_nuisance,
        rho_signal=rho_signal,
    )

    v = true_signal_vector(p, nuisance_block_size)

    # Exact class counts
    n1 = int(round(n * minority_fraction))
    n1 = max(1, min(n - 1, n1))
    n0 = n - n1

    # Within-class variance along the signal direction is exactly 1-rho_signal.
    lambda_signal_within = 1.0 - rho_signal

    # Requested within-class Cohen's d along the true discriminative direction.
    mean_difference = signal_d * np.sqrt(lambda_signal_within)

    mu0 = -0.5 * mean_difference * v
    mu1 = +0.5 * mean_difference * v

    X0 = rng.multivariate_normal(mu0, Sigma, size=n0)
    X1 = rng.multivariate_normal(mu1, Sigma, size=n1)

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    # Shuffle rows jointly
    idx = rng.permutation(n)
    X = X[idx]
    y = y[idx]

    return X, y, Sigma, v, mean_difference


def theoretical_structure(p, nuisance_block_size, rho_nuisance, rho_signal):
    """
    Theoretical within-class eigenvalues before class mean shift.
    Useful for documentation. Eigenvectors are simple because covariance is block diagonal.
    """
    m = nuisance_block_size

    rows = []

    # Nuisance block: one sum direction and m-1 contrasts
    rows.append({
        "component_family": "nuisance_block_sum",
        "multiplicity": 1,
        "within_class_eigenvalue": 1.0 + (m - 1) * rho_nuisance,
        "contains_true_class_signal": False,
    })

    rows.append({
        "component_family": "nuisance_block_contrasts",
        "multiplicity": m - 1,
        "within_class_eigenvalue": 1.0 - rho_nuisance,
        "contains_true_class_signal": False,
    })

    # Signal pair
    rows.append({
        "component_family": "signal_pair_sum",
        "multiplicity": 1,
        "within_class_eigenvalue": 1.0 + rho_signal,
        "contains_true_class_signal": False,
    })

    rows.append({
        "component_family": "signal_pair_contrast",
        "multiplicity": 1,
        "within_class_eigenvalue": 1.0 - rho_signal,
        "contains_true_class_signal": True,
    })

    # Independent noise
    n_ind = p - m - 2
    if n_ind > 0:
        rows.append({
            "component_family": "independent_noise",
            "multiplicity": n_ind,
            "within_class_eigenvalue": 1.0,
            "contains_true_class_signal": False,
        })

    return pd.DataFrame(rows)


def fmt_float(x):
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def main():
    ap = argparse.ArgumentParser(
        description="Generate binary synthetic datasets with class signal in a low-variance PCA direction."
    )

    ap.add_argument("--outdir", default="synthetic_low_variance_pc")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--p", type=int, default=20)

    ap.add_argument(
        "--signal-d",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Within-class Cohen's d along the true low-variance discriminative direction."
    )

    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Independent dataset seeds."
    )

    ap.add_argument(
        "--minority-fraction",
        type=float,
        default=0.50,
        help="Fraction assigned to class 1. Keep 0.50 for the proof-of-mechanism experiment."
    )

    ap.add_argument("--nuisance-block-size", type=int, default=10)
    ap.add_argument("--rho-nuisance", type=float, default=0.70)
    ap.add_argument("--rho-signal", type=float, default=0.98)

    args = ap.parse_args()

    if args.n < 20:
        raise ValueError("n should be at least 20.")
    if args.p < 4:
        raise ValueError("p should be at least 4.")
    if not (0 < args.minority_fraction < 1):
        raise ValueError("minority_fraction must be in (0,1).")
    if any(d < 0 for d in args.signal_d):
        raise ValueError("signal-d values must be >= 0.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feature_names = [f"x{i:03d}" for i in range(1, args.p + 1)]

    # Save theoretical structure once
    theoretical = theoretical_structure(
        p=args.p,
        nuisance_block_size=args.nuisance_block_size,
        rho_nuisance=args.rho_nuisance,
        rho_signal=args.rho_signal,
    )
    theoretical.to_csv(outdir / "theoretical_pc_structure.csv", index=False)

    manifests = []

    for d in args.signal_d:
        for seed in args.seeds:
            X, y, Sigma, v, mean_diff = generate_dataset(
                n=args.n,
                p=args.p,
                minority_fraction=args.minority_fraction,
                nuisance_block_size=args.nuisance_block_size,
                rho_nuisance=args.rho_nuisance,
                rho_signal=args.rho_signal,
                signal_d=d,
                seed=seed,
            )

            df = pd.DataFrame(X, columns=feature_names)
            df.insert(0, "y", y)

            fname = f"synthetic_lowvar_d{fmt_float(d)}_seed{seed:02d}.csv"
            df.to_csv(outdir / fname, index=False)

            signal_features = np.flatnonzero(np.abs(v) > 0)
            signal_feature_desc = ";".join(
                f"{feature_names[i]}:{v[i]:+.8f}" for i in signal_features
            )

            manifests.append({
                "file": fname,
                "n": args.n,
                "p": args.p,
                "class0_n": int((y == 0).sum()),
                "class1_n": int((y == 1).sum()),
                "minority_fraction_requested": args.minority_fraction,
                "seed": seed,
                "signal_d_requested": d,
                "rho_signal": args.rho_signal,
                "signal_within_variance": 1.0 - args.rho_signal,
                "signal_within_sd": np.sqrt(1.0 - args.rho_signal),
                "signal_mean_difference_along_true_direction": mean_diff,
                "rho_nuisance": args.rho_nuisance,
                "nuisance_block_size": args.nuisance_block_size,
                "nuisance_sum_eigenvalue": 1.0 + (args.nuisance_block_size - 1) * args.rho_nuisance,
                "nuisance_contrast_eigenvalue": 1.0 - args.rho_nuisance,
                "signal_sum_eigenvalue": 1.0 + args.rho_signal,
                "signal_contrast_eigenvalue_within": 1.0 - args.rho_signal,
                "true_signal_direction": signal_feature_desc,
            })

    manifest = pd.DataFrame(manifests)
    manifest.to_csv(outdir / "synthetic_manifest.csv", index=False)

    # Human-readable design note
    note = f"""Synthetic low-variance discriminative-PC experiment

n = {args.n}
p = {args.p}
minority fraction = {args.minority_fraction}

Nuisance correlated block:
  size = {args.nuisance_block_size}
  rho = {args.rho_nuisance}
  leading eigenvalue = {1.0 + (args.nuisance_block_size - 1) * args.rho_nuisance:.6f}
  contrast eigenvalues = {1.0 - args.rho_nuisance:.6f}

Signal pair:
  rho = {args.rho_signal}
  high-variance sum eigenvalue = {1.0 + args.rho_signal:.6f}
  low-variance contrast eigenvalue = {1.0 - args.rho_signal:.6f}

True discriminative direction:
  (x{args.nuisance_block_size+1:03d} - x{args.nuisance_block_size+2:03d}) / sqrt(2)

Class means differ ONLY along this low-variance contrast direction.

Requested within-class Cohen d values:
  {args.signal_d}

Seeds:
  {args.seeds}

Important:
  This design is intended for a downstream pipeline that standardizes
  original features before PCA. The low-variance direction arises from
  correlation structure, not from unequal marginal feature scales.
"""
    (outdir / "DESIGN.txt").write_text(note, encoding="utf-8")

    print("Generated", len(manifest), "datasets")
    print("Output directory:", outdir.resolve())
    print()
    print(theoretical.to_string(index=False))
    print()
    print("Files:")
    print("  synthetic_manifest.csv")
    print("  theoretical_pc_structure.csv")
    print("  DESIGN.txt")
    print("  + one CSV per (signal_d, seed)")


if __name__ == "__main__":
    main()
