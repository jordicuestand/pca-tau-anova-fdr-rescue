#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(
    r"synthetic_n(?P<n>\d+)_p(?P<p>\d+)_b(?P<maj>\d+)_(?P<min>\d+)_seed(?P<seed>\d+)$"
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Resumeix l'experiment factorial n x proporció minoritària."
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_synthetic_imbalance_nxprop"),
        help="Directori arrel amb un subdirectori per dataset.",
    )
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("synthetic_imbalance_nxprop"),
        help="Prefix dels CSV de sortida.",
    )
    return ap.parse_args()


def dataset_meta(name: str):
    m = NAME_RE.fullmatch(name)
    if not m:
        return None
    d = {k: int(v) for k, v in m.groupdict().items()}
    d["minority_prop"] = d["min"] / 100.0
    d["majority_prop"] = d["maj"] / 100.0
    d["n_minority"] = int(round(d["n"] * d["minority_prop"]))
    d["n_majority"] = d["n"] - d["n_minority"]
    d["p_over_n"] = d["p"] / d["n"]
    return d


def load_all(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.rglob("*_nested_results.csv"))
    if not files:
        raise SystemExit(f"No s'han trobat *_nested_results.csv a {results_dir}")

    frames = []
    bad = []
    for f in files:
        df = pd.read_csv(f)
        if "dataset" not in df.columns:
            bad.append(str(f))
            continue
        frames.append(df)

    if bad:
        print("AVÍS: fitxers ignorats sense columna dataset:")
        for f in bad:
            print("  ", f)

    if not frames:
        raise SystemExit("No hi ha fitxers de resultats vàlids.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(
        subset=["dataset", "repeat", "outer_fold", "method"], keep="last"
    )
    return all_df


def validate_design(df: pd.DataFrame):
    expected_methods = {"ALL", "PCA", "HYBRID", "PLSDA"}
    problems = []
    for dataset, g in df.groupby("dataset"):
        meta = dataset_meta(str(dataset))
        if meta is None:
            continue
        methods = set(g["method"].astype(str))
        missing = expected_methods - methods
        if missing:
            problems.append(f"{dataset}: falten mètodes {sorted(missing)}")

        # 5 outer folds x 10 repeats = 50 files per method in the final experiment.
        counts = g.groupby("method").size().to_dict()
        for method in expected_methods:
            if counts.get(method, 0) != 50:
                problems.append(
                    f"{dataset}: {method} té {counts.get(method,0)} files, esperades 50"
                )
    return problems


def make_fold_level(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only factorial datasets.
    rows = []
    for dataset, g in df.groupby("dataset"):
        meta = dataset_meta(str(dataset))
        if meta is None:
            continue

        keys = ["dataset", "repeat", "outer_fold"]
        pca = g[g.method == "PCA"][keys + ["f1", "auc", "mcc"]].rename(
            columns={"f1": "f1_pca", "auc": "auc_pca", "mcc": "mcc_pca"}
        )
        hyb_cols = keys + ["f1", "auc", "mcc", "n_rescued", "n_selected", "n_pca_tau"]
        hyb = g[g.method == "HYBRID"][hyb_cols].rename(
            columns={
                "f1": "f1_hybrid",
                "auc": "auc_hybrid",
                "mcc": "mcc_hybrid",
                "n_selected": "n_selected_hybrid",
            }
        )
        paired = hyb.merge(pca, on=keys, how="inner", validate="one_to_one")
        paired["delta_f1"] = paired.f1_hybrid - paired.f1_pca
        paired["delta_auc"] = paired.auc_hybrid - paired.auc_pca
        paired["delta_mcc"] = paired.mcc_hybrid - paired.mcc_pca
        paired["rescued_any"] = (paired.n_rescued.fillna(0) > 0).astype(int)

        for k, v in meta.items():
            paired[k] = v
        rows.append(paired)

    if not rows:
        raise SystemExit("No s'han trobat datasets amb el patró factorial esperat.")
    return pd.concat(rows, ignore_index=True)


def make_seed_summary(fold: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset", "n", "p", "maj", "min", "seed", "minority_prop",
        "n_minority", "n_majority", "p_over_n"
    ]
    return (
        fold.groupby(group_cols, as_index=False)
        .agg(
            n_outer_folds=("rescued_any", "size"),
            rescue_frequency=("rescued_any", "mean"),
            mean_n_rescued=("n_rescued", "mean"),
            mean_n_pca_tau=("n_pca_tau", "mean"),
            mean_n_selected_hybrid=("n_selected_hybrid", "mean"),
            delta_auc_mean=("delta_auc", "mean"),
            delta_auc_sd=("delta_auc", "std"),
            delta_mcc_mean=("delta_mcc", "mean"),
            delta_mcc_sd=("delta_mcc", "std"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_sd=("delta_f1", "std"),
        )
        .sort_values(["n", "minority_prop", "seed"], ascending=[True, False, True])
    )


def make_condition_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    # IMPORTANT: uncertainty is across the 10 independent generated datasets/seeds,
    # not across the dependent repeated-CV folds.
    group_cols = ["n", "p", "minority_prop", "n_minority", "n_majority", "p_over_n"]
    return (
        seed_df.groupby(group_cols, as_index=False)
        .agg(
            n_datasets=("seed", "size"),
            rescue_frequency_mean=("rescue_frequency", "mean"),
            rescue_frequency_sd=("rescue_frequency", "std"),
            mean_n_rescued_mean=("mean_n_rescued", "mean"),
            mean_n_rescued_sd=("mean_n_rescued", "std"),
            delta_auc_mean=("delta_auc_mean", "mean"),
            delta_auc_sd_across_datasets=("delta_auc_mean", "std"),
            delta_mcc_mean=("delta_mcc_mean", "mean"),
            delta_mcc_sd_across_datasets=("delta_mcc_mean", "std"),
            delta_f1_mean=("delta_f1_mean", "mean"),
            delta_f1_sd_across_datasets=("delta_f1_mean", "std"),
        )
        .sort_values(["n", "minority_prop"], ascending=[True, False])
    )


def make_abs_minority_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_df.groupby("n_minority", as_index=False)
        .agg(
            n_conditions=("dataset", lambda s: s.str.extract(r"synthetic_n(\d+)_p100_b(\d+)_(\d+)_seed").drop_duplicates().shape[0]),
            n_datasets=("dataset", "size"),
            rescue_frequency_mean=("rescue_frequency", "mean"),
            rescue_frequency_sd=("rescue_frequency", "std"),
            mean_n_rescued_mean=("mean_n_rescued", "mean"),
            delta_auc_mean=("delta_auc_mean", "mean"),
            delta_mcc_mean=("delta_mcc_mean", "mean"),
            delta_f1_mean=("delta_f1_mean", "mean"),
        )
        .sort_values("n_minority")
    )


def main():
    args = parse_args()
    raw = load_all(args.results_dir)

    problems = validate_design(raw)
    if problems:
        print("\nAVÍS: comprovació de completitud:")
        for p in problems:
            print(" -", p)
    else:
        print("Comprovació de completitud: OK (50 outer folds per mètode i dataset).")

    fold = make_fold_level(raw)
    seed = make_seed_summary(fold)
    cond = make_condition_summary(seed)
    absmin = make_abs_minority_summary(seed)

    prefix = args.out_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fold_path = Path(str(prefix) + "_fold_level.csv")
    seed_path = Path(str(prefix) + "_seed_summary.csv")
    cond_path = Path(str(prefix) + "_condition_summary.csv")
    abs_path = Path(str(prefix) + "_absolute_minority_summary.csv")

    fold.to_csv(fold_path, index=False)
    seed.to_csv(seed_path, index=False)
    cond.to_csv(cond_path, index=False)
    absmin.to_csv(abs_path, index=False)

    print("\n=== CONDITION SUMMARY ===")
    show_cols = [
        "n", "minority_prop", "n_minority", "n_datasets",
        "rescue_frequency_mean", "rescue_frequency_sd",
        "mean_n_rescued_mean", "delta_auc_mean", "delta_mcc_mean", "delta_f1_mean"
    ]
    print(cond[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCSV escrits:")
    for p in [fold_path, seed_path, cond_path, abs_path]:
        print(" ", p)


if __name__ == "__main__":
    main()
