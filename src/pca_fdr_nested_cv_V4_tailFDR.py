#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repeated nested CV for PCA-tau + tail-only ANOVA/FDR rescue.

Methods:
  1) ALL + linear SVM
  2) PCA-tau + linear SVM
  3) PCA-tau + ANOVA/BH-FDR rescue on discarded PCs only + linear SVM
  4) PLS-DA (PLSRegression as binary classifier)

Design:
  - Outer repeated 5-fold CV (10 repeats by default)
  - Inner 5-fold CV
  - Inner selection by MCC
  - Every data-dependent step fitted on training data only
  - Group-aware CV when --group is supplied
  - Outer loop sequential; inner GridSearchCV parallel (default n_jobs=3)
  - Native BLAS/OpenMP threads limited to 1 per process
  - Checkpoint after every method / outer fold; resumable

Examples:
  python pca_fdr_nested_cv.py --quick
  python pca_fdr_nested_cv.py --file data.csv --target diagnosis --positive 1
  python pca_fdr_nested_cv.py --file pcgita.csv --target class --group subject_id --positive PD
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import argparse
import json
import math
import time
import warnings
import sys
from pathlib import Path

from parkinson_loader import load_pcgita_excel, load_uci_174

# Give custom sklearn objects a stable importable module name for joblib/loky.
# The module name must match the real filename so worker processes can import it.
_JOBLIB_MODULE = "pca_fdr_nested_cv_V4_tailFDR"

if __name__ == "__main__":
    sys.modules.setdefault(_JOBLIB_MODULE, sys.modules[__name__])

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

SEED = 20260808
OUTER_SPLITS = 5
OUTER_REPEATS = 10
INNER_SPLITS = 5
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3]
PLS_GRID = [1, 2, 3, 5, 8, 10, 15, 20]
TAU = 0.90
ALPHA = 0.05
N_JOBS = 3
MCC_SCORER = make_scorer(matthews_corrcoef)


from pathlib import Path


def load_synthetic_csv(path):
    """
    Carrega els datasets sintètics de l'experiment de desequilibri.

    Format esperat:
        x001, x002, ..., x100, target

    target ha de contenir exclusivament 0 i 1.
    """

    path = Path(path)

    df = pd.read_csv(path)

    if "target" in df.columns:
        target_col = "target"
    elif "y" in df.columns:
        target_col = "y"
    else:
        raise ValueError(
            f"{path}: no existeix ni la columna 'target' ni la columna 'y'."
        )

    y = df[target_col].to_numpy()

    X = df.drop(columns=[target_col]).copy()

    # Comprovació de target binari
    classes = np.unique(y)

    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(
            f"{path}: target ha de ser binari {{0,1}}; "
            f"s'han trobat {classes.tolist()}."
        )

    # Totes les features han de ser numèriques
    bad = [
        c for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c])
    ]

    if bad:
        raise ValueError(
            f"{path}: predictors no numèrics: {bad}"
        )

    if X.isna().any().any():
        raise ValueError(
            f"{path}: el dataset sintètic conté NaN."
        )

    dataset_name = path.stem

    target_meta = {
        "target_name": target_col,
        "classes": [0, 1],
        "negative_label": 0,
        "positive_label": 1,
        "source": "synthetic"
    }

    groups = None

    print()
    print(f"Synthetic dataset: {dataset_name}")
    print(f"X shape: {X.shape}")
    print(
        "Class counts:",
        dict(zip(*np.unique(y, return_counts=True)))
    )

    return X, y, groups, dataset_name, target_meta

def subject_stratified_splits(y, groups, n_splits, random_state):
    groups = np.asarray(groups)
    y = np.asarray(y)

    unique_groups = np.unique(groups)

    # Una etiqueta per subjecte
    group_y = np.array([
        y[groups == g][0]
        for g in unique_groups
    ])

    # Comprovació: cada subjecte ha de tenir una sola classe
    for g in unique_groups:
        if len(np.unique(y[groups == g])) != 1:
            raise ValueError(f"El grup {g} conté més d'una classe")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for train_g_idx, test_g_idx in skf.split(unique_groups, group_y):

        train_groups = unique_groups[train_g_idx]
        test_groups  = unique_groups[test_g_idx]

        train_idx = np.flatnonzero(np.isin(groups, train_groups))
        test_idx  = np.flatnonzero(np.isin(groups, test_groups))

        yield train_idx, test_idx


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.ones_like(p)
    finite = np.isfinite(p)
    if not np.any(finite):
        return q
    pv = p[finite]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = adjusted
    q[finite] = out
    return q


def encode_binary(y, positive_label=None):
    s = pd.Series(y)
    classes = list(pd.unique(s.dropna()))
    if len(classes) != 2:
        raise ValueError(f"Binary target required; found {classes}")
    if positive_label is not None:
        match = [c for c in classes if c == positive_label or str(c) == str(positive_label)]
        if len(match) != 1:
            raise ValueError(f"--positive={positive_label!r} does not uniquely match {classes}")
        pos = match[0]
    else:
        try:
            pos = sorted(classes)[1]
        except Exception:
            pos = classes[1]
    neg = [c for c in classes if c != pos][0]
    mapping = {neg: 0, pos: 1}
    return s.map(mapping).to_numpy(dtype=int), {
        "negative_label": str(neg), "positive_label": str(pos)
    }


def sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return sens, spec, int(tn), int(fp), int(fn), int(tp)


def evaluate(y_true, y_pred, scores):
    sens, spec, tn, fp, fn, tp = sens_spec(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = np.nan
    return {
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc": auc,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def jdump(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def local_eigengaps(vals, indices):
    vals = np.asarray(vals, dtype=float)
    gaps = []
    for j in indices:
        c = []
        if j > 0:
            c.append(abs(vals[j - 1] - vals[j]))
        if j < len(vals) - 1:
            c.append(abs(vals[j] - vals[j + 1]))
        gaps.append(min(c) if c else np.nan)
    return np.asarray(gaps)


class PCATau(BaseEstimator, TransformerMixin):
    def __init__(self, tau=0.90):
        self.tau = tau

    def fit(self, X, y=None):
        self.pca_ = PCA(svd_solver="full")
        self.pca_.fit(X)
        cum = np.cumsum(self.pca_.explained_variance_ratio_)
        self.n_selected_ = int(np.searchsorted(cum, self.tau, side="left") + 1)
        self.selected_indices_ = np.arange(self.n_selected_, dtype=int)
        self.selected_explained_variance_ = float(
            self.pca_.explained_variance_ratio_[:self.n_selected_].sum()
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        return self.pca_.transform(X)[:, :self.n_selected_]


class PCAFDRRescue(BaseEstimator, TransformerMixin):
    """
    PCA-tau plus supervised rescue of discarded PCs.

    Important methodological rule:
      - PCs 1..k_tau form the unsupervised PCA core and are always retained.
      - ANOVA is performed ONLY on PCs beyond k_tau.
      - Benjamini-Hochberg FDR correction is therefore applied ONLY to the
        family of PCs that are actual candidates for rescue.
    """
    def __init__(self, tau=0.90, alpha=0.05):
        self.tau = tau
        self.alpha = alpha

    def fit(self, X, y):
        self.pca_ = PCA(svd_solver="full")
        Z = self.pca_.fit_transform(X)

        cum = np.cumsum(self.pca_.explained_variance_ratio_)
        self.n_pca_tau_ = int(
            np.searchsorted(cum, self.tau, side="left") + 1
        )

        n_components = Z.shape[1]
        baseline = np.arange(self.n_pca_tau_, dtype=int)
        tail = np.arange(self.n_pca_tau_, n_components, dtype=int)

        # Full-length diagnostic vectors. Core PCs are deliberately untested,
        # hence NaN there. Only tail entries receive F/p/q values.
        fvals_full = np.full(n_components, np.nan, dtype=float)
        pvals_full = np.full(n_components, np.nan, dtype=float)
        qvals_full = np.full(n_components, np.nan, dtype=float)

        if len(tail) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_tail, p_tail = f_classif(Z[:, tail], y)

            f_tail = np.asarray(f_tail, dtype=float)
            p_tail = np.asarray(p_tail, dtype=float)

            p_tail[~np.isfinite(p_tail)] = 1.0
            f_tail[~np.isfinite(f_tail)] = 0.0

            # CRITICAL V4 CHANGE:
            # BH is applied only to the candidate family (the PCA tail).
            q_tail = bh_fdr(p_tail)

            rescued = tail[q_tail <= self.alpha]

            fvals_full[tail] = f_tail
            pvals_full[tail] = p_tail
            qvals_full[tail] = q_tail
        else:
            f_tail = np.array([], dtype=float)
            p_tail = np.array([], dtype=float)
            q_tail = np.array([], dtype=float)
            rescued = np.array([], dtype=int)

        selected = np.sort(
            np.unique(np.concatenate([baseline, rescued]))
        )

        # Diagnostics / fitted state
        self.tail_indices_ = tail
        self.n_tail_candidates_ = len(tail)

        self.f_values_ = fvals_full
        self.p_values_ = pvals_full
        self.q_values_ = qvals_full

        self.tail_f_values_ = f_tail
        self.tail_p_values_ = p_tail
        self.tail_q_values_ = q_tail

        # Kept for backward compatibility with older diagnostics code:
        # in V4 these are significant TESTED tail PCs, not significant PCs
        # from an ANOVA performed on the entire PCA basis.
        self.anova_all_indices_ = rescued.astype(int)

        self.rescued_indices_ = rescued.astype(int)
        self.selected_indices_ = selected.astype(int)
        self.n_rescued_ = len(rescued)
        self.n_selected_ = len(selected)

        self.selected_explained_variance_ = float(
            self.pca_.explained_variance_ratio_[selected].sum()
        )
        self.rescued_variance_ratio_ = (
            self.pca_.explained_variance_ratio_[rescued]
        )
        self.rescued_eigengaps_ = local_eigengaps(
            self.pca_.explained_variance_, rescued
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        return self.pca_.transform(X)[:, self.selected_indices_]


class PLSDA(ClassifierMixin, BaseEstimator):
    """Binary PLS-DA via PLSRegression(y in {0,1}); threshold 0.5."""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.model_ = PLSRegression(
            n_components=self.n_components,
            scale=False, max_iter=500, tol=1e-6
        )
        self.model_.fit(X, np.asarray(y, dtype=float))
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict(self, X):
        return (self.decision_function(X) >= 0.5).astype(int)


# Stable, importable module names are needed by joblib/loky when this file
# is executed as a script and GridSearchCV uses multiple worker processes.
PCATau.__module__ = _JOBLIB_MODULE
PCAFDRRescue.__module__ = _JOBLIB_MODULE
PLSDA.__module__ = _JOBLIB_MODULE


UCI_ALIASES = {
    "ionosphere": {"id": 52, "name": "UCI52_Ionosphere", "positive": "g"},
    "sonar": {"id": 151, "name": "UCI151_Sonar", "positive": "M"},
    "banknote": {"id": 267, "name": "UCI267_Banknote", "positive": 1},
    "spambase": {"id": 94, "name": "UCI94_Spambase", "positive": 1},
    "qsar": {"id": 254, "name": "UCI254_QSAR_biodegradation", "positive": None},
}


def load_uci_generic(uci_id, dataset_name=None, positive_label=None):
    """Load a binary numeric UCI dataset through ucimlrepo.

    The target must consist of exactly one binary column. Predictors must be
    numeric; missing values are allowed because the main pipeline imputes them
    inside each training fold.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is required for UCI datasets. Install it with: "
            "pip install ucimlrepo"
        ) from exc

    ds = fetch_ucirepo(id=int(uci_id))
    X = ds.data.features.copy()
    targets = ds.data.targets.copy()

    if targets is None or targets.shape[1] != 1:
        n_targets = 0 if targets is None else targets.shape[1]
        raise ValueError(
            f"UCI id={uci_id}: expected exactly one target column; found {n_targets}"
        )

    bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        raise ValueError(
            f"UCI id={uci_id}: current protocol expects numeric predictors; "
            f"non-numeric columns: {bad[:15]}"
        )

    y_raw = targets.iloc[:, 0]
    y, meta = encode_binary(y_raw, positive_label)

    if dataset_name is None:
        dataset_name = f"UCI{uci_id}"

    print(
        f"Loaded UCI id={uci_id}: {dataset_name} | "
        f"n={len(X)}, p={X.shape[1]} | "
        f"target={targets.columns[0]!r} | "
        f"positive={meta['positive_label']}"
    )

    return X, y, None, dataset_name, meta

def make_column_names_unique(columns):
    counts = {}
    new_cols = []

    for col in map(str, columns):
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")

    return new_cols

def load_data(args):

    if args.dataset == "uci174":
        X, y, groups = load_uci_174()

        target_meta = {
            "negative_label": 0,
            "positive_label": 1
        }

        return X, y, groups, "UCI174_Parkinson", target_meta

    if args.dataset == "pcgita":
        X, y, groups = load_pcgita_excel(
            data_root=args.data_root,
            signal=args.signal,
            session=args.session
        )

        dataset_name = f"PCGITA_{args.signal}_{args.session}"

        target_meta = {
            "negative_label": "S",
            "positive_label": "E"
        }

        return X, y, groups, dataset_name, target_meta

    # Built-in UCI aliases: no local CSV needed.
    if args.dataset in UCI_ALIASES:
        spec = UCI_ALIASES[args.dataset]
        positive = args.positive if args.positive is not None else spec["positive"]
        return load_uci_generic(
            spec["id"], dataset_name=spec["name"], positive_label=positive
        )

    # Arbitrary UCI dataset by numeric repository id.
    if args.uci_id is not None:
        return load_uci_generic(
            args.uci_id,
            dataset_name=args.dataset_name or f"UCI{args.uci_id}_{args.dataset}",
            positive_label=args.positive,
        )

    if args.openml_id is not None:

        ds = fetch_openml(
            data_id=args.openml_id,
            as_frame=True
        )

        X = ds.data.copy()
        y_raw = ds.target.copy()


        constant_cols = [
            c for c in X.columns
            if X[c].nunique(dropna=True) <= 1
        ]

        if constant_cols:
            print("Columnes constants eliminades:", constant_cols)
            X = X.drop(columns=constant_cols)

        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))

        unique_classes = np.unique(y)
        if unique_classes.size != 2:
            raise ValueError(
                "El protocol requereix classificació binària. "
                f"S'han trobat {unique_classes.size} classes."
            )

        X.columns = make_column_names_unique(X.columns)

        bad = [
            c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c])
        ]
                
        for c in bad:
            # Preserve genuine missing values. They are handled later by the
            # train-only SimpleImputer inside the CV pipeline.
            original_notna = X[c].notna()
            vals = X.loc[original_notna, c].unique()

            if len(vals) != 2:
                raise ValueError(
                    f"La columna {c} no és binària: {vals}"
                )

            mapping = {vals[0]: 0, vals[1]: 1}
            converted = X[c].map(mapping)

            # Reject only NaNs newly created from originally observed values.
            introduced_nan = original_notna & converted.isna()
            if introduced_nan.any():
                bad_vals = X.loc[introduced_nan, c].unique()
                raise ValueError(
                    f"La codificació de {c} ha introduït valors NaN "
                    f"per als valors observats {bad_vals}"
                )

            X[c] = converted.astype(float)

        # Tornem a comprovar després de la conversió
        bad = [
            c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c])
        ]        
        
        if bad:
            raise ValueError(
            "   Aquest dataset OpenML conté predictors no numèrics: "
                f"{bad[:15]}"
            )

        dataset_name = getattr(ds, "DESCR", None)
        dataset_name = f"OpenML_{args.openml_id}"

        groups = None

        target_meta = {
            "negative_label": str(encoder.classes_[0]),
            "positive_label": str(encoder.classes_[1]),
        }

        
        return X, y, groups, dataset_name, target_meta

    if args.synthetic_csv is not None:
        return load_synthetic_csv(args.synthetic_csv)

    # fallback
    if args.file is None:
        if args.dataset != "generic":
            raise ValueError(
                f"Unknown dataset {args.dataset!r}. Use a built-in UCI alias "
                "(ionosphere, sonar, banknote, spambase, qsar), --uci-id ID, "
                "or provide --file and --target."
            )
        ds = load_breast_cancer(as_frame=True)
        X = ds.data.copy()
        y, meta = encode_binary(ds.target, 1)
        return X, y, None, "sklearn_breast_cancer", meta

    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=args.sheet)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Supported: .csv, .xlsx, .xls, .parquet")

    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target!r} not found")

    groups = None
    if args.group:
        if args.group not in df.columns:
            raise ValueError(f"Group column {args.group!r} not found")
        groups = df[args.group].to_numpy()

    drop = [args.target] + ([args.group] if args.group else []) + list(args.drop)
    X = df.drop(columns=list(dict.fromkeys(drop))).copy()

    bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        raise ValueError(
            "Current protocol expects numeric predictors after explicit encoding. "
            f"Non-numeric columns: {bad[:15]}"
        )

    y, meta = encode_binary(df[args.target], args.positive)
    return X, y, groups, args.dataset_name or path.stem, meta



def outer_splits(X, y, groups, k, repeats, seed, subject_level=False):
    for rep in range(repeats):
        this_seed = seed + rep * 1000

        if groups is None:
            cv = StratifiedKFold(k, shuffle=True, random_state=this_seed)
            iterator = cv.split(X, y)
        elif subject_level:
            iterator = subject_stratified_splits(
                y, groups, n_splits=k, random_state=this_seed
            )
        else:
            cv = StratifiedGroupKFold(k, shuffle=True, random_state=this_seed)
            iterator = cv.split(X, y, groups)

        for fold, (tr, te) in enumerate(iterator, 1):
            yield rep + 1, fold, tr, te


def inner_cv(y, groups, k, seed, subject_level=False):
    if groups is None:
        return StratifiedKFold(k, shuffle=True, random_state=seed)

    if subject_level:
        # GridSearchCV accepts an iterable/list of explicit (train_idx, test_idx) splits.
        return list(subject_stratified_splits(
            y, groups, n_splits=k, random_state=seed
        ))

    return StratifiedGroupKFold(k, shuffle=True, random_state=seed)


def svm_pipe(kind, tau, alpha, memory):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if kind == "PCA":
        steps.append(("representation", PCATau(tau)))
    elif kind == "HYBRID":
        steps.append(("representation", PCAFDRRescue(tau, alpha)))
    steps.append(("clf", SVC(kernel="linear", probability=False, cache_size=1024)))
    return Pipeline(steps, memory=memory)


def pls_pipe(memory):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", PLSDA(2)),
    ], memory=memory)


def valid_pls_grid(n_outer, p, inner_k, base):
    min_inner = math.floor(n_outer * (inner_k - 1) / inner_k)
    max_comp = max(1, min(p, min_inner - 1))
    g = sorted(set(int(v) for v in base if 1 <= int(v) <= max_comp))
    return g or [1]


def diagnostics(method, pipe):
    d = dict(
        n_selected=np.nan, n_pca_tau=np.nan, n_rescued=0,
        n_tail_candidates=np.nan,
        anova_tested_pcs_1based="[]",
        tail_p_values="[]", tail_q_values="[]",
        rescued_pcs_1based="[]", anova_sig_pcs_1based="[]",
        selected_pcs_1based="[]", selected_explained_variance=np.nan,
        rescued_var_ratio="[]", rescued_eigengaps="[]",
    )
    if method == "PCA":
        r = pipe.named_steps["representation"]
        d.update(
            n_selected=r.n_selected_, n_pca_tau=r.n_selected_,
            selected_pcs_1based=jdump((r.selected_indices_ + 1).tolist()),
            selected_explained_variance=r.selected_explained_variance_,
        )
    elif method == "HYBRID":
        r = pipe.named_steps["representation"]
        d.update(
            n_selected=r.n_selected_,
            n_pca_tau=r.n_pca_tau_,
            n_rescued=r.n_rescued_,
            n_tail_candidates=r.n_tail_candidates_,
            anova_tested_pcs_1based=jdump((r.tail_indices_ + 1).tolist()),
            tail_p_values=jdump([float(x) for x in r.tail_p_values_]),
            tail_q_values=jdump([float(x) for x in r.tail_q_values_]),
            rescued_pcs_1based=jdump((r.rescued_indices_ + 1).tolist()),
            anova_sig_pcs_1based=jdump((r.anova_all_indices_ + 1).tolist()),
            selected_pcs_1based=jdump((r.selected_indices_ + 1).tolist()),
            selected_explained_variance=r.selected_explained_variance_,
            rescued_var_ratio=jdump([float(x) for x in r.rescued_variance_ratio_]),
            rescued_eigengaps=jdump([
                float(x) if np.isfinite(x) else None
                for x in r.rescued_eigengaps_
            ]),
        )
    elif method == "PLSDA":
        d["n_selected"] = pipe.named_steps["clf"].n_components
    return d


COLS = [
    "dataset", "repeat", "outer_fold", "method", "tau", "alpha",
    "n_train", "n_test", "p", "best_param", "inner_best_mcc",
    "f1", "auc", "mcc", "sensitivity", "specificity", "tn", "fp", "fn", "tp",
    "fit_seconds", "n_selected", "n_pca_tau", "n_rescued",
    "n_tail_candidates", "anova_tested_pcs_1based",
    "tail_p_values", "tail_q_values",
    "rescued_pcs_1based", "anova_sig_pcs_1based", "selected_pcs_1based",
    "selected_explained_variance", "rescued_var_ratio", "rescued_eigengaps",
]


def checkpoint_done(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df.dataset.astype(str), df.repeat.astype(int),
                   df.outer_fold.astype(int), df.method.astype(str)))


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=COLS).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def run(args):
    Xdf, y, groups, name, target_meta = load_data(args)
    X = Xdf.to_numpy(dtype=float)
    n, p = X.shape
    counts = np.bincount(y, minlength=2)
    if counts.min() < args.outer_splits:
        raise ValueError("Too few minority-class observations for requested outer folds")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / f"{name}_nested_results.csv"
    summary_path = outdir / f"{name}_summary.csv"
    config_path = outdir / f"{name}_config.json"
    cache_path = outdir / "_joblib_cache"
    memory = joblib.Memory(str(cache_path), verbose=0) if args.cache else None

    cfg = vars(args).copy()
    cfg.update(
        dataset=name,
        n=n,
        p=p,
        class_counts=counts.tolist(),
        target_encoding=target_meta,
        native_threads_per_process=1,
        hybrid_version="V4_tail_only_ANOVA_BH",
        fdr_family="PCA_tail_candidates_only",
    )
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    done = checkpoint_done(result_path)
    print("=" * 78)
    print("PCA-tau + ANOVA/FDR rescue | repeated nested CV")
    print(f"Dataset: {name} | n={n}, p={p}, p/n={p/n:.4f}")
    print(f"Class counts: {counts.tolist()} | positive={target_meta['positive_label']}")
    print(f"Outer: {args.outer_repeats} x {args.outer_splits}-fold | Inner: {args.inner_splits}-fold")
    print(f"tau={args.tau}, alpha={args.alpha}, n_jobs={args.n_jobs}, BLAS threads=1")
    print("HYBRID V4: ANOVA + BH-FDR only on PCs beyond the PCA-tau cutoff")
    print(f"C grid: {args.c_grid}")
    print(f"Checkpoint: {result_path}")
    print("=" * 78, flush=True)

    methods = ["ALL", "PCA", "HYBRID", "PLSDA"]
    total_outer = args.outer_repeats * args.outer_splits
    global_start = time.time()

    # UCI 174 has repeated recordings per subject and a single class label per subject.
    # For this dataset, stratify directly at subject level so each fold preserves
    # class representation while keeping all recordings of a subject together.
    subject_level_cv = (args.dataset == "uci174")

    for outer_i, (rep, fold, tr, te) in enumerate(
            outer_splits(
                X, y, groups, args.outer_splits, args.outer_repeats,
                args.seed, subject_level=subject_level_cv
            ), 1):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            raise ValueError(
                f"Degenerate outer split at rep={rep}, fold={fold}: "
                f"train classes={np.unique(ytr).tolist()}, "
                f"test classes={np.unique(yte).tolist()}"
            )

        gtr = groups[tr] if groups is not None else None
        cv_inner = inner_cv(
            ytr, gtr, args.inner_splits,
            args.seed + rep * 1000 + fold,
            subject_level=subject_level_cv
        )
        print(f"\n[Outer {outer_i:02d}/{total_outer}] rep={rep} fold={fold} train={len(tr)} test={len(te)}")

        for method in methods:
            key = (name, rep, fold, method)
            if key in done:
                print(f"  {method:7s}: checkpoint -> skip")
                continue
            t0 = time.time()
            if method in {"ALL", "PCA", "HYBRID"}:
                pipe = svm_pipe(method, args.tau, args.alpha, memory)
                grid = {"clf__C": args.c_grid}
                pname = "C"
            else:
                pipe = pls_pipe(memory)
                gridvals = valid_pls_grid(len(tr), p, args.inner_splits, args.pls_grid)
                grid = {"clf__n_components": gridvals}
                pname = "n_components"

            search = GridSearchCV(
                pipe, grid, scoring=MCC_SCORER, cv=cv_inner,
                n_jobs=args.n_jobs, refit=True, error_score="raise",
                pre_dispatch=args.n_jobs, return_train_score=False,
            )
            kwargs = (
                {"groups": gtr}
                if gtr is not None and not subject_level_cv
                else {}
            )
            with threadpool_limits(limits=1):
                search.fit(Xtr, ytr, **kwargs)

            best = search.best_estimator_
            pred = best.predict(Xte)
            if hasattr(best, "decision_function"):
                scores = best.decision_function(Xte)
            elif hasattr(best, "predict_proba"):
                scores = best.predict_proba(Xte)[:, 1]
            else:
                scores = pred
            met = evaluate(yte, pred, scores)
            diag = diagnostics(method, best)
            bval = list(search.best_params_.values())[0]
            elapsed = time.time() - t0
            row = {
                "dataset": name, "repeat": rep, "outer_fold": fold, "method": method,
                "tau": args.tau if method in {"PCA", "HYBRID"} else np.nan,
                "alpha": args.alpha if method == "HYBRID" else np.nan,
                "n_train": len(tr), "n_test": len(te), "p": p,
                "best_param": f"{pname}={bval}",
                "inner_best_mcc": float(search.best_score_),
                **met, "fit_seconds": elapsed, **diag,
            }
            if method == "ALL":
                row["n_selected"] = p
            append_row(result_path, row)
            done.add(key)
            extra = f", rescued={row['n_rescued']}" if method == "HYBRID" else ""
            print(f"  {method:7s}: MCC={row['mcc']:+.4f}, F1={row['f1']:.4f}, AUC={row['auc']:.4f}, {row['best_param']}, {elapsed:.1f}s{extra}", flush=True)

        elapsed_all = time.time() - global_start
        eta = elapsed_all / outer_i * (total_outer - outer_i)
        print(f"  Rough ETA: {eta/3600:.2f} h | elapsed: {elapsed_all/3600:.2f} h", flush=True)

    res = pd.read_csv(result_path)
    summary = res.groupby("method", as_index=False).agg(
        folds=("mcc", "count"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
        mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
        sensitivity_mean=("sensitivity", "mean"), sensitivity_sd=("sensitivity", "std"),
        specificity_mean=("specificity", "mean"), specificity_sd=("specificity", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
        n_selected_mean=("n_selected", "mean"), n_rescued_mean=("n_rescued", "mean"),
    ).sort_values("mcc_mean", ascending=False)
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))

    pca = res[res.method == "PCA"].set_index(["repeat", "outer_fold"])
    hyb = res[res.method == "HYBRID"].set_index(["repeat", "outer_fold"])
    common = pca.index.intersection(hyb.index)
    if len(common):
        print("\nHYBRID - PCA paired outer-fold differences (descriptive only):")
        for metric in ["f1", "auc", "mcc"]:
            d = hyb.loc[common, metric] - pca.loc[common, metric]
            print(f"  Δ{metric.upper():3s}: mean={d.mean():+.5f}, sd={d.std(ddof=1):.5f}, n={len(d)}")
    print(f"\nResults: {result_path}\nSummary: {summary_path}\nConfig : {config_path}")


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument(
        "--synthetic-csv",
        type=str,
        default=None,
        help="CSV sintètic amb predictors numèrics i columna target binària."
    )


    ap.add_argument(
    "--openml-id",
    type=int,
    default=None
    )

    ap.add_argument(
        "--dataset",
        default="generic",
        help=(
            "Dataset identifier. Special loaders: pcgita, uci174; "
            "built-in UCI aliases: ionosphere, sonar, banknote, spambase, qsar. "
            "For any other UCI dataset, use --uci-id ID."
        )
    )
    ap.add_argument(
        "--uci-id", type=int, default=None,
        help="Fetch an arbitrary binary numeric UCI dataset by ucimlrepo id."
    )

    ap.add_argument("--data-root", type=str, default=None)

    ap.add_argument(
        "--signal",
        choices=["AWPE", "F0WPE"],
        default="AWPE"
    )

    ap.add_argument(
        "--session",
        type=int,
        choices=[3, 4],
        default=3
    )

    
    ap.add_argument("--file", default=None, help="CSV/XLSX/Parquet. Omit for sklearn breast-cancer smoke test.")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--target", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--positive", default=None)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--outer-splits", type=int, default=OUTER_SPLITS)
    ap.add_argument("--outer-repeats", type=int, default=OUTER_REPEATS)
    ap.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    ap.add_argument("--c-grid", nargs="+", type=float, default=C_GRID)
    ap.add_argument("--pls-grid", nargs="+", type=int, default=PLS_GRID)
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="results_pca_fdr")
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.set_defaults(cache=True)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 1x3 outer, 3 inner, small grids.")
    args = ap.parse_args()
    if args.file is not None and not args.target:
        ap.error("--target is required when --file is used")
    if args.quick:
        args.outer_splits = 3
        args.outer_repeats = 1
        args.inner_splits = 3
        args.c_grid = [0.1, 1.0, 10.0]
        args.pls_grid = [1, 2, 3, 5]
        args.outdir = str(Path(args.outdir) / "quick")
    if args.n_jobs < 1:
        ap.error("--n-jobs must be >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
