from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

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


def load_uci_by_id(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)

    X = dataset.data.features.copy()
    X.columns = make_column_names_unique(X.columns)

    y = prepare_target(
        dataset_id,
        dataset.data.targets,
    )

    groups = None
    dataset_name = dataset.metadata.name

    print(f"Dataset: {dataset_name}")
    print(f"UCI ID: {dataset_id}")
    print(f"N={len(X)}, p_original={X.shape[1]}")
    print(
        "Classes:",
        dict(zip(*np.unique(y, return_counts=True)))
    )

    return X, y, groups, dataset_name
