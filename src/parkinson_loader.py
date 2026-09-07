from pathlib import Path
import pandas as pd

def load_pcgita_excel(data_root, signal="AWPE", session=3):
    import pandas as pd
    import numpy as np

    data_root = Path(data_root).expanduser()

    file_path = data_root / f"{signal}_{session}.xlsx"

    df = pd.read_excel(file_path)

        
    required = {"Id", "Condición", "Fonación"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Falten columnes obligatòries: {required - set(df.columns)}"
        )

    # Target
    cond = df["Condición"].astype(str).str.strip()

    bad = set(cond.unique()) - {"E", "S"}
    if bad:
        raise ValueError(f"Valors inesperats a Condición: {bad}")

    y = (cond == "E").astype(int).to_numpy()

    # Grouping per subjecte
    groups = (
        df["Condición"].astype(str).str.strip()
        + "_"
        + df["Id"].astype(str)
    ).to_numpy()
    n_subjects = df.groupby(["Condición", "Id"]).ngroups

    print("DEBUG n_subjects =", n_subjects)
    print("DEBUG Id únics =", df["Id"].nunique())
    print("DEBUG groups únics =", len(np.unique(groups)))
    # Predictors: excloem identificadors i target
    exclude = {"Id", "Condición", "Fonación"}

    feature_cols = [
        c for c in df.columns
        if c not in exclude
    ]

    X = df[feature_cols].copy()

    # Verificació numèrica
    non_numeric = [
        c for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c])
    ]

    if non_numeric:
        raise ValueError(
            f"Features no numèriques: {non_numeric}"
        )

    print("PC-GITA carregat:")
    print(f"  Files           = {len(df)}")
    print(f"  Subjectes únics = {n_subjects}")
    print(f"  Features        = {X.shape[1]}")
    print(f"  PD (E)          = {(y == 1).sum()} files")
    print(f"  HC (S)          = {(y == 0).sum()} files")

    return X, y, groups

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def load_uci_174():
    ds = fetch_ucirepo(id=174)

    X = ds.data.features.copy()
    y = ds.data.targets["status"].astype(int).to_numpy()

    original = ds.data.original.copy()

    # Ex.: phon_R01_S01_1 -> subject = phon_R01_S01
    groups = (
        original["name"]
        .astype(str)
        .str.rsplit("_", n=1)
        .str[0]
        .to_numpy()
    )

    n_subjects = len(np.unique(groups))

    print("UCI 174 Parkinson carregat:")
    print(f"  Files           = {len(X)}")
    print(f"  Subjectes únics = {n_subjects}")
    print(f"  Features        = {X.shape[1]}")
    print(f"  PD (1)          = {(y == 1).sum()} files")
    print(f"  HC (0)          = {(y == 0).sum()} files")

    # Comprovacions bàsiques
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y i groups tenen longituds diferents")

    if n_subjects != 32:
        raise ValueError(
            f"S'esperaven 31 subjectes, però se n'han trobat {n_subjects}"
        )

    # Cada subjecte ha de pertànyer a una única classe
    tmp = pd.DataFrame({"group": groups, "y": y})
    nclasses_per_subject = tmp.groupby("group")["y"].nunique()

    if (nclasses_per_subject > 1).any():
        raise ValueError(
            "Hi ha subjectes amb més d'una classe assignada"
        )

    return X, y, groups
