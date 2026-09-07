#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

N_VALUES = [200, 400, 800]
MINORITY_PROPORTIONS = [0.40, 0.20, 0.10, 0.05]
P = 100

N_SEEDS = 10
BASE_SEED = 20260819

OUTDIR = Path("synthetic_imbalance_nxprop")

SIGNAL_PCS = [4, 39, 84]  # PC5, PC40, PC85
EFFECT_SIZE = 0.80
MEASUREMENT_NOISE_SD = 0.05


def make_eigenvalues(p):
    idx = np.arange(p)
    return 3.0 * np.exp(-idx / 28.0) + 0.08


def make_rotation(p, seed=123456):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    Q, _ = np.linalg.qr(A)
    return Q


EIGVALS = make_eigenvalues(P)
ROTATION = make_rotation(P)


def generate_dataset(n, minority_prop, seed):
    rng = np.random.default_rng(seed)

    n1 = int(round(n * minority_prop))
    n0 = n - n1

    y = np.array([0] * n0 + [1] * n1, dtype=int)
    rng.shuffle(y)

    Z = rng.normal(
        loc=0.0,
        scale=np.sqrt(EIGVALS),
        size=(n, P)
    )

    signed_class = 2 * y - 1

    for pc in SIGNAL_PCS:
        delta = EFFECT_SIZE * np.sqrt(EIGVALS[pc])
        Z[:, pc] += 0.5 * delta * signed_class

    X = Z @ ROTATION.T

    if MEASUREMENT_NOISE_SD > 0:
        X += rng.normal(
            0.0,
            MEASUREMENT_NOISE_SD,
            size=X.shape
        )

    columns = [f"x{i+1:03d}" for i in range(P)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Synthetic imbalance factorial design")
    print(f"n values: {N_VALUES}")
    print(f"p = {P}")
    print("minority proportions:", MINORITY_PROPORTIONS)
    print(f"seeds per condition = {N_SEEDS}")
    print("Signal PCs:", [pc + 1 for pc in SIGNAL_PCS])

    cum = np.cumsum(EIGVALS) / np.sum(EIGVALS)
    m90 = np.searchsorted(cum, 0.90) + 1
    print(f"Approximate population PCA90: {m90} PCs")
    print(f"Cumulative variance at PC{m90}: {cum[m90-1]:.4f}")
    print()

    manifest = []

    for n in N_VALUES:
        for minority_prop in MINORITY_PROPORTIONS:
            majority_pct = int(round((1.0 - minority_prop) * 100))
            minority_pct = int(round(minority_prop * 100))

            condition_code = n * 1000 + minority_pct * 10

            for i in range(N_SEEDS):
                seed = BASE_SEED + condition_code + i

                df = generate_dataset(
                    n=n,
                    minority_prop=minority_prop,
                    seed=seed
                )

                filename = (
                    f"synthetic_n{n}_p{P}_"
                    f"b{majority_pct}_{minority_pct}_"
                    f"seed{i+1:02d}.csv"
                )

                path = OUTDIR / filename
                df.to_csv(path, index=False)

                counts = df["target"].value_counts().sort_index()
                n0 = int(counts.get(0, 0))
                n1 = int(counts.get(1, 0))

                approx_outer_train_minority = 0.8 * n1
                approx_inner_train_minority = 0.8 * approx_outer_train_minority

                manifest.append({
                    "file": filename,
                    "seed": seed,
                    "replicate": i + 1,
                    "class_0": n0,
                    "class_1": n1,
                    "minority_prop": minority_prop,
                    "minority_pct": minority_pct,
                    "n_minority": n1,
                    "n_majority": n0,
                    "n": n,
                    "p": P,
                    "p_over_n": P / n,
                    "approx_outer_train_minority": approx_outer_train_minority,
                    "approx_inner_train_minority": approx_inner_train_minority,
                    "effect_size": EFFECT_SIZE,
                    "signal_pcs": ",".join(str(pc + 1) for pc in SIGNAL_PCS)
                })

                print(
                    f"{filename}: "
                    f"class 0={n0}, class 1={n1}, "
                    f"approx inner-train minority={approx_inner_train_minority:.1f}"
                )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(OUTDIR / "synthetic_manifest.csv", index=False)

    design = (
        manifest_df[
            ["n", "p", "minority_prop", "minority_pct",
             "n_minority", "n_majority", "p_over_n",
             "approx_outer_train_minority",
             "approx_inner_train_minority"]
        ]
        .drop_duplicates()
        .sort_values(["n", "minority_prop"], ascending=[True, False])
        .reset_index(drop=True)
    )
    design.to_csv(OUTDIR / "synthetic_design.csv", index=False)

    print()
    print(f"Created {len(manifest)} datasets.")
    print(f"Factorial conditions: {len(N_VALUES) * len(MINORITY_PROPORTIONS)}")
    print(f"Output directory: {OUTDIR.resolve()}")
    print()
    print(design.to_string(index=False))


if __name__ == "__main__":
    main()
