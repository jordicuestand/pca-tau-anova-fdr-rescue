#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometric diagnostics for PCA + ANOVA/FDR experiments.

This module is observational only: it reads the already-fitted HYBRID pipeline
and the corresponding outer-training fold. It does not refit or modify any
model, threshold, split, PCA, ANOVA/FDR decision, or SVM hyperparameter.
"""

from pathlib import Path
import numpy as np
import pandas as pd


ALL_PC_DIAG_COLS = [
    "dataset", "repeat", "outer_fold",
    "pc_1based", "pc_0based", "n_pca_tau",
    "distance_from_pca_tau", "inside_pca_tau",
    "is_anova_significant", "is_rescued",
    "explained_variance", "explained_variance_ratio",
    "cumulative_variance_ratio",
    "eigengap_prev", "eigengap_next", "local_eigengap",
    "relative_eigengap",
    "class0_n", "class1_n",
    "class0_mean", "class1_mean",
    "centroid_difference", "abs_centroid_difference",
    "class0_sd", "class1_sd", "pooled_sd",
    "cohens_d", "abs_cohens_d", "hedges_g",
    "anova_f", "p_value", "q_value",
]

ALL_PC_LOADING_COLS = [
    "dataset", "repeat", "outer_fold",
    "pc_1based", "pc_0based", "n_pca_tau",
    "inside_pca_tau", "is_rescued",
    "feature", "feature_index_1based", "loading",
]

SPECTRUM_COLS = [
    "dataset", "repeat", "outer_fold",
    "pc_1based", "pc_0based", "n_pca_tau",
    "explained_variance", "explained_variance_ratio",
    "cumulative_variance_ratio",
    "eigengap_prev", "eigengap_next", "local_eigengap",
    "relative_eigengap",
]

FOLD_GEOMETRY_COLS = [
    "dataset", "repeat", "outer_fold",
    "n_train", "p", "n_pca_tau",
    "pca_tau_variance_ratio",
    "n_anova_significant", "n_rescued",
    "spectral_entropy",
    "effective_rank",
    "tail_variance_after_tau",
    "min_relative_eigengap_after_tau",
    "median_relative_eigengap_after_tau",
]


def _append_df(path, df, columns):
    if df is None or len(df) == 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, columns].to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def _hedges_correction(n0, n1):
    df = int(n0 + n1 - 2)
    if df <= 1:
        return np.nan
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def _eigengap_data(ev):
    ev = np.asarray(ev, dtype=float)
    m = len(ev)
    prev = np.full(m, np.nan)
    nxt = np.full(m, np.nan)

    if m > 1:
        prev[1:] = np.abs(ev[:-1] - ev[1:])
        nxt[:-1] = np.abs(ev[:-1] - ev[1:])

    local = np.fmin(
        np.where(np.isfinite(prev), prev, np.inf),
        np.where(np.isfinite(nxt), nxt, np.inf),
    )
    local[np.isinf(local)] = np.nan

    # Scale-free version: local eigengap relative to the eigenvalue itself.
    rel = np.divide(
        local, np.abs(ev),
        out=np.full_like(local, np.nan),
        where=np.abs(ev) > 0,
    )
    return prev, nxt, local, rel


def _spectral_entropy(evr):
    p = np.asarray(evr, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan, np.nan
    p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    effective_rank = float(np.exp(h))
    return h, effective_rank


def collect_pca_geometry(best_pipe, X_train_raw, y_train, feature_names,
                         dataset, repeat, outer_fold):
    """
    Extract full PCA/ANOVA geometry from one already-fitted HYBRID estimator.

    All quantities are computed on the outer-training fold only.
    """
    rep = best_pipe.named_steps["representation"]

    X_imp = best_pipe.named_steps["imputer"].transform(X_train_raw)
    X_std = best_pipe.named_steps["scaler"].transform(X_imp)
    Z = rep.pca_.transform(X_std)

    y = np.asarray(y_train)
    c0 = (y == 0)
    c1 = (y == 1)
    n0, n1 = int(c0.sum()), int(c1.sum())

    ev = np.asarray(rep.pca_.explained_variance_, dtype=float)
    evr = np.asarray(rep.pca_.explained_variance_ratio_, dtype=float)
    cum = np.cumsum(evr)
    prev_gap, next_gap, local_gap, rel_gap = _eigengap_data(ev)

    sig = set(np.asarray(rep.anova_all_indices_, dtype=int).tolist())
    rescued = set(np.asarray(rep.rescued_indices_, dtype=int).tolist())
    n_tau = int(rep.n_pca_tau_)

    diag_rows = []
    loading_rows = []
    spectrum_rows = []

    for j in range(len(ev)):
        z0, z1 = Z[c0, j], Z[c1, j]
        m0 = float(np.mean(z0)) if n0 else np.nan
        m1 = float(np.mean(z1)) if n1 else np.nan
        sd0 = float(np.std(z0, ddof=1)) if n0 > 1 else np.nan
        sd1 = float(np.std(z1, ddof=1)) if n1 > 1 else np.nan

        df = n0 + n1 - 2
        if df > 0 and n0 > 1 and n1 > 1:
            pv = ((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / df
            pooled = float(np.sqrt(max(pv, 0.0)))
        else:
            pooled = np.nan

        delta = m1 - m0 if np.isfinite(m0) and np.isfinite(m1) else np.nan
        if np.isfinite(pooled) and pooled > 0:
            d = float(delta / pooled)
            J = _hedges_correction(n0, n1)
            g = float(J * d) if np.isfinite(J) else np.nan
        else:
            d = np.nan
            g = np.nan

        base = {
            "dataset": dataset,
            "repeat": int(repeat),
            "outer_fold": int(outer_fold),
            "pc_1based": int(j + 1),
            "pc_0based": int(j),
            "n_pca_tau": n_tau,
            "distance_from_pca_tau": int(j - n_tau + 1),
            "inside_pca_tau": bool(j < n_tau),
            "is_anova_significant": bool(j in sig),
            "is_rescued": bool(j in rescued),
            "explained_variance": float(ev[j]),
            "explained_variance_ratio": float(evr[j]),
            "cumulative_variance_ratio": float(cum[j]),
            "eigengap_prev": float(prev_gap[j]) if np.isfinite(prev_gap[j]) else np.nan,
            "eigengap_next": float(next_gap[j]) if np.isfinite(next_gap[j]) else np.nan,
            "local_eigengap": float(local_gap[j]) if np.isfinite(local_gap[j]) else np.nan,
            "relative_eigengap": float(rel_gap[j]) if np.isfinite(rel_gap[j]) else np.nan,
        }

        diag_rows.append({
            **base,
            "class0_n": n0,
            "class1_n": n1,
            "class0_mean": m0,
            "class1_mean": m1,
            "centroid_difference": delta,
            "abs_centroid_difference": abs(delta) if np.isfinite(delta) else np.nan,
            "class0_sd": sd0,
            "class1_sd": sd1,
            "pooled_sd": pooled,
            "cohens_d": d,
            "abs_cohens_d": abs(d) if np.isfinite(d) else np.nan,
            "hedges_g": g,
            "anova_f": float(rep.f_values_[j]),
            "p_value": float(rep.p_values_[j]),
            "q_value": float(rep.q_values_[j]),
        })

        spectrum_rows.append({
            k: base[k] for k in SPECTRUM_COLS
        })

        for feat_idx, loading in enumerate(rep.pca_.components_[j]):
            loading_rows.append({
                "dataset": dataset,
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "pc_1based": int(j + 1),
                "pc_0based": int(j),
                "n_pca_tau": n_tau,
                "inside_pca_tau": bool(j < n_tau),
                "is_rescued": bool(j in rescued),
                "feature": str(feature_names[feat_idx]),
                "feature_index_1based": int(feat_idx + 1),
                "loading": float(loading),
            })

    h, erank = _spectral_entropy(evr)
    tail = evr[n_tau:] if n_tau < len(evr) else np.array([], dtype=float)
    tail_rel_gaps = rel_gap[n_tau:] if n_tau < len(rel_gap) else np.array([], dtype=float)
    finite_tail_gaps = tail_rel_gaps[np.isfinite(tail_rel_gaps)]

    fold_row = {
        "dataset": dataset,
        "repeat": int(repeat),
        "outer_fold": int(outer_fold),
        "n_train": int(len(y)),
        "p": int(len(ev)),
        "n_pca_tau": n_tau,
        "pca_tau_variance_ratio": float(evr[:n_tau].sum()),
        "n_anova_significant": int(len(sig)),
        "n_rescued": int(len(rescued)),
        "spectral_entropy": h,
        "effective_rank": erank,
        "tail_variance_after_tau": float(tail.sum()) if len(tail) else 0.0,
        "min_relative_eigengap_after_tau": (
            float(np.min(finite_tail_gaps)) if len(finite_tail_gaps) else np.nan
        ),
        "median_relative_eigengap_after_tau": (
            float(np.median(finite_tail_gaps)) if len(finite_tail_gaps) else np.nan
        ),
    }

    return (
        pd.DataFrame(diag_rows, columns=ALL_PC_DIAG_COLS),
        pd.DataFrame(loading_rows, columns=ALL_PC_LOADING_COLS),
        pd.DataFrame(spectrum_rows, columns=SPECTRUM_COLS),
        pd.DataFrame([fold_row], columns=FOLD_GEOMETRY_COLS),
    )


def save_pca_geometry(best_pipe, X_train_raw, y_train, feature_names,
                      dataset, repeat, outer_fold, outdir):
    """
    Append full-fold PCA geometry to four CSV files.
    """
    outdir = Path(outdir)
    diag, loads, spectrum, fold = collect_pca_geometry(
        best_pipe=best_pipe,
        X_train_raw=X_train_raw,
        y_train=y_train,
        feature_names=feature_names,
        dataset=dataset,
        repeat=repeat,
        outer_fold=outer_fold,
    )

    paths = {
        "all_pc_diagnostics": outdir / f"{dataset}_all_pc_diagnostics.csv",
        "all_pc_loadings": outdir / f"{dataset}_all_pc_loadings.csv",
        "eigen_spectrum": outdir / f"{dataset}_eigen_spectrum.csv",
        "fold_geometry": outdir / f"{dataset}_fold_geometry.csv",
    }

    _append_df(paths["all_pc_diagnostics"], diag, ALL_PC_DIAG_COLS)
    _append_df(paths["all_pc_loadings"], loads, ALL_PC_LOADING_COLS)
    _append_df(paths["eigen_spectrum"], spectrum, SPECTRUM_COLS)
    _append_df(paths["fold_geometry"], fold, FOLD_GEOMETRY_COLS)

    return paths
