#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATTERN = re.compile(r"^synthetic_lowvar_d(?P<d>[0-9]+(?:p[0-9]+)?)_seed(?P<seed>[0-9]+)$")

def parse_dir_name(name):
    m = PATTERN.match(name)
    if not m:
        return None
    return float(m.group("d").replace("p", ".")), int(m.group("seed"))

def normalize_method(x):
    s = str(x).strip().upper()
    if s in {"PCA", "PCA90", "PCA-TAU", "PCA_TAU"}:
        return "PCA"
    if s in {"HYBRID", "PCA+FDR", "PCA_FDR"}:
        return "HYBRID"
    if s in {"ALL", "PCA_ALL", "PCA-ALL"}:
        return "ALL"
    if s in {"PLSDA", "PLS-DA", "PLS_DA", "PLS"}:
        return "PLSDA"
    return s

def first_present(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns found: {candidates}")
    return None

def read_summary(summary_path):
    df = pd.read_csv(summary_path)
    method_col = first_present(df, ["method", "Method"], required=True)
    df = df.copy()
    df["_method"] = df[method_col].map(normalize_method)

    colmap = {
        "auc": ["auc_mean", "mean_auc", "AUC_mean", "auc"],
        "mcc": ["mcc_mean", "mean_mcc", "MCC_mean", "mcc"],
        "f1": ["f1_mean", "mean_f1", "F1_mean", "f1"],
        "auc_sd": ["auc_sd", "sd_auc", "AUC_sd"],
        "mcc_sd": ["mcc_sd", "sd_mcc", "MCC_sd"],
        "f1_sd": ["f1_sd", "sd_f1", "F1_sd"],
        "avg_rescued_pcs": [
            "avg_rescued_pcs", "mean_rescued_pcs",
            "rescued_pc_count_mean", "avg_rescue", "n_rescued_mean"
        ],
        "rescue_fraction": [
            "rescue_fold_fraction", "fraction_folds_with_rescue",
            "rescue_rate", "prop_folds_with_rescue"
        ],
        "avg_selected_pcs": [
            "avg_selected_pcs", "mean_selected_pcs", "n_selected_pcs_mean"
        ],
    }

    rows = []
    for _, r in df.iterrows():
        method = r["_method"]
        if method not in {"ALL", "PCA", "HYBRID", "PLSDA"}:
            continue
        out = {"method": method}
        for key, candidates in colmap.items():
            col = first_present(df, candidates, required=False)
            out[key] = r[col] if col is not None else np.nan
        rows.append(out)

    if not rows:
        raise ValueError(f"{summary_path}: no recognized methods found.")
    return pd.DataFrame(rows)

def collect_results(root):
    by_seed_rows = []
    long_rows = []
    incomplete = []
    skipped = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_dir_name(subdir.name)
        if parsed is None:
            continue
        d, seed = parsed

        exact = subdir / f"{subdir.name}_summary.csv"
        summary_files = sorted(subdir.glob("*_summary.csv"))
        if exact.exists():
            summary_path = exact
        elif summary_files:
            summary_path = summary_files[0]
        else:
            incomplete.append(subdir.name)
            continue

        try:
            summ = read_summary(summary_path)
        except Exception as exc:
            skipped.append((subdir.name, str(exc)))
            continue

        wide_row = {"dataset": subdir.name, "signal_d": d, "seed": seed}

        for _, r in summ.iterrows():
            method = r["method"]
            for metric in [
                "auc", "mcc", "f1",
                "auc_sd", "mcc_sd", "f1_sd",
                "avg_rescued_pcs", "rescue_fraction", "avg_selected_pcs"
            ]:
                wide_row[f"{method}_{metric}"] = r.get(metric, np.nan)

            long_rows.append({
                "dataset": subdir.name,
                "signal_d": d,
                "seed": seed,
                "method": method,
                "auc": r.get("auc", np.nan),
                "mcc": r.get("mcc", np.nan),
                "f1": r.get("f1", np.nan),
                "auc_sd": r.get("auc_sd", np.nan),
                "mcc_sd": r.get("mcc_sd", np.nan),
                "f1_sd": r.get("f1_sd", np.nan),
                "avg_rescued_pcs": r.get("avg_rescued_pcs", np.nan),
                "rescue_fraction": r.get("rescue_fraction", np.nan),
                "avg_selected_pcs": r.get("avg_selected_pcs", np.nan),
            })

        for metric in ["auc", "mcc", "f1"]:
            h = wide_row.get(f"HYBRID_{metric}", np.nan)
            p = wide_row.get(f"PCA_{metric}", np.nan)
            a = wide_row.get(f"ALL_{metric}", np.nan)
            q = wide_row.get(f"PLSDA_{metric}", np.nan)
            if pd.notna(h) and pd.notna(p):
                wide_row[f"delta_{metric}_HYBRID_minus_PCA"] = h - p
            if pd.notna(h) and pd.notna(a):
                wide_row[f"delta_{metric}_HYBRID_minus_ALL"] = h - a
            if pd.notna(h) and pd.notna(q):
                wide_row[f"delta_{metric}_HYBRID_minus_PLSDA"] = h - q

        by_seed_rows.append(wide_row)

    if by_seed_rows:
        by_seed = pd.DataFrame(by_seed_rows)
        by_seed = by_seed.sort_values(["signal_d", "seed"]).reset_index(drop=True)
    else:
        by_seed = pd.DataFrame(columns=["dataset", "signal_d", "seed"])

    if long_rows:
        long_df = pd.DataFrame(long_rows)
        long_df = long_df.sort_values(["signal_d", "seed", "method"]).reset_index(drop=True)
    else:
        long_df = pd.DataFrame(columns=[
            "dataset", "signal_d", "seed", "method",
            "auc", "mcc", "f1",
            "auc_sd", "mcc_sd", "f1_sd",
            "avg_rescued_pcs", "rescue_fraction", "avg_selected_pcs"
        ])

    return by_seed, long_df, incomplete, skipped

def aggregate_by_d_method(long_df):
    if long_df.empty:
        return pd.DataFrame(columns=["signal_d", "method", "n_seeds"])
    rows = []
    for (d, method), sub in long_df.groupby(["signal_d", "method"]):
        row = {"signal_d": d, "method": method, "n_seeds": len(sub)}
        for metric in ["auc", "mcc", "f1", "avg_rescued_pcs", "rescue_fraction", "avg_selected_pcs"]:
            vals = sub[metric].dropna()
            row[f"mean_{metric}"] = vals.mean() if len(vals) else np.nan
            row[f"sd_{metric}"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["signal_d", "method"])

def aggregate_hybrid_deltas(by_seed):
    if by_seed.empty:
        return pd.DataFrame(columns=["signal_d", "n_seeds"])
    rows = []
    for d, sub in by_seed.groupby("signal_d"):
        row = {"signal_d": d, "n_seeds": len(sub)}
        for metric in ["auc", "mcc", "f1"]:
            for target in ["PCA", "ALL", "PLSDA"]:
                col = f"delta_{metric}_HYBRID_minus_{target}"
                if col in sub.columns:
                    vals = sub[col].dropna()
                    row[f"mean_{col}"] = vals.mean() if len(vals) else np.nan
                    row[f"sd_{col}"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_d")

def plot_metric_by_d(agg, metric, outpath):
    plt.figure(figsize=(7, 5))
    for method in ["PCA", "HYBRID", "ALL", "PLSDA"]:
        sub = agg[agg["method"] == method].sort_values("signal_d")
        if sub.empty:
            continue
        plt.errorbar(
            sub["signal_d"].to_numpy(),
            sub[f"mean_{metric}"].to_numpy(),
            yerr=sub[f"sd_{metric}"].to_numpy(),
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xlabel("Signal d")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs signal d")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_hybrid_rescue(agg, outpath):
    plt.figure(figsize=(7, 5))
    sub = agg[agg["method"] == "HYBRID"].sort_values("signal_d")
    if not sub.empty:
        plt.errorbar(
            sub["signal_d"].to_numpy(),
            sub["mean_avg_rescued_pcs"].to_numpy(),
            yerr=sub["sd_avg_rescued_pcs"].to_numpy(),
            marker="o",
            capsize=3,
        )
    plt.xlabel("Signal d")
    plt.ylabel("Average rescued PCs")
    plt.title("HYBRID rescued PCs vs signal d")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_delta_by_d(delta_df, metric, outpath):
    plt.figure(figsize=(7, 5))
    plotted = False
    for target in ["PCA", "ALL", "PLSDA"]:
        mean_col = f"mean_delta_{metric}_HYBRID_minus_{target}"
        sd_col = f"sd_delta_{metric}_HYBRID_minus_{target}"
        if mean_col not in delta_df.columns:
            continue
        plt.errorbar(
            delta_df["signal_d"].to_numpy(),
            delta_df[mean_col].to_numpy(),
            yerr=delta_df[sd_col].to_numpy() if sd_col in delta_df.columns else None,
            marker="o",
            capsize=3,
            label=f"HYBRID - {target}",
        )
        plotted = True
    if plotted:
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Signal d")
        plt.ylabel(f"Δ {metric.upper()}")
        plt.title(f"HYBRID advantage vs signal d")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory with 30 result folders.")
    ap.add_argument("--outdir", default=None, help="Default: <root>/final_summary")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else root / "final_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    candidate_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and PATTERN.match(p.name)
    ]

    print(f"Root: {root}")
    print(f"Synthetic result folders recognized by name: {len(candidate_dirs)}")

    by_seed, long_df, incomplete, skipped = collect_results(root)

    if by_seed.empty or long_df.empty:
        print("\nDIAGNOSTIC")
        print(f"Recognized synthetic folders: {len(candidate_dirs)}")
        print(f"Folders without *_summary.csv: {len(incomplete)}")
        if incomplete:
            print("  " + ", ".join(incomplete[:20]))
        print(f"Folders skipped while reading summary: {len(skipped)}")
        for name, msg in skipped[:20]:
            print(f"  {name}: {msg}")
        raise RuntimeError(
            "No completed readable runs were collected. "
            "Check --root and the *_summary.csv files listed above."
        )

    print(f"Completed readable datasets collected: {len(by_seed)}")

    by_d_method = aggregate_by_d_method(long_df)
    hybrid_deltas = aggregate_hybrid_deltas(by_seed)

    by_seed.to_csv(outdir / "lowvar_by_seed_wide.csv", index=False)
    long_df.to_csv(outdir / "lowvar_long_metrics.csv", index=False)
    by_d_method.to_csv(outdir / "lowvar_by_d_method.csv", index=False)
    hybrid_deltas.to_csv(outdir / "lowvar_hybrid_deltas_by_d.csv", index=False)

    plot_metric_by_d(by_d_method, "auc", outdir / "auc_vs_d.png")
    plot_metric_by_d(by_d_method, "mcc", outdir / "mcc_vs_d.png")
    plot_metric_by_d(by_d_method, "f1", outdir / "f1_vs_d.png")
    plot_hybrid_rescue(by_d_method, outdir / "hybrid_rescued_pcs_vs_d.png")
    plot_delta_by_d(hybrid_deltas, "auc", outdir / "hybrid_delta_auc_vs_d.png")
    plot_delta_by_d(hybrid_deltas, "mcc", outdir / "hybrid_delta_mcc_vs_d.png")
    plot_delta_by_d(hybrid_deltas, "f1", outdir / "hybrid_delta_f1_vs_d.png")

    display_cols = [
        "signal_d", "method", "n_seeds",
        "mean_auc", "sd_auc",
        "mean_mcc", "sd_mcc",
        "mean_f1", "sd_f1",
        "mean_avg_rescued_pcs"
    ]
    display_cols = [c for c in display_cols if c in by_d_method.columns]

    print("\n=== AGGREGATED BY SIGNAL d AND METHOD ===")
    print(by_d_method[display_cols].to_string(index=False))

    print("\n=== HYBRID DELTAS BY SIGNAL d ===")
    print(hybrid_deltas.to_string(index=False))

    print(f"\nCompleted datasets: {len(by_seed)}")
    if incomplete:
        print(f"Incomplete folders (no final summary): {len(incomplete)}")
        print("  " + ", ".join(incomplete))
    if skipped:
        print(f"Skipped folders due to parse/read issues: {len(skipped)}")
        for name, msg in skipped:
            print(f"  {name}: {msg}")

    print(f"\nSaved outputs to: {outdir}")

if __name__ == "__main__":
    main()
