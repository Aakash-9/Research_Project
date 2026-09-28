"""
Step 2 - Federated client simulation.

Splits the Step 1 labeled outputs into per-site client folders, holding out
whole bearings/units for test (never rows) so a machine's life never leaks
across train and test.

federated_clients/<site>/train.csv
federated_clients/<site>/test.csv

Run directly: python scripts/02_federated_split.py
"""
import os
import numpy as np
import pandas as pd

FEMTO_FEATURES_CSV = "output/femto_features_labeled.csv"
CMAPSS_TRAIN_CSV = "output/cmapss_train_labeled.csv"
CMAPSS_TEST_CSV = "output/cmapss_test_labeled.csv"
OUT_DIR = "federated_clients"
TRAIN_FRACTION = 0.75
SEED = 42


def split_by_group(df, group_col, train_fraction, rng):
    """Assign whole groups (bearings/units) to train or test so no machine's
    life is split across both splits."""
    groups = sorted(df[group_col].unique())
    rng.shuffle(groups)
    n_train = max(1, round(len(groups) * train_fraction))
    # keep at least one group in test when there's more than one group to split
    if n_train == len(groups) and len(groups) > 1:
        n_train -= 1
    train_groups = set(groups[:n_train])
    test_groups = set(groups[n_train:])
    return df[df[group_col].isin(train_groups)], df[df[group_col].isin(test_groups)]


def main():
    rng = np.random.default_rng(SEED)
    summary = []

    # ---- FEMTO: split by bearing_id, already tagged with `site` ----
    femto = pd.read_csv(FEMTO_FEATURES_CSV)
    for site, g in femto.groupby("site"):
        train, test = split_by_group(g, "bearing_id", TRAIN_FRACTION, rng)
        site_dir = os.path.join(OUT_DIR, site)
        os.makedirs(site_dir, exist_ok=True)
        train.to_csv(os.path.join(site_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(site_dir, "test.csv"), index=False)
        summary.append((site, g["bearing_id"].nunique(), len(train), len(test)))

    # ---- C-MAPSS: already split into train/test files by the dataset itself.
    # We still resplit by unit_number within the *provided* train file so each
    # site's client gets its own held-out units (the official test_FD00X.txt
    # files are a separate held-out evaluation set, not used here). ----
    cmapss_train_all = pd.read_csv(CMAPSS_TRAIN_CSV)
    for site, g in cmapss_train_all.groupby("site"):
        train, test = split_by_group(g, "unit_number", TRAIN_FRACTION, rng)
        site_dir = os.path.join(OUT_DIR, site)
        os.makedirs(site_dir, exist_ok=True)
        train.to_csv(os.path.join(site_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(site_dir, "test.csv"), index=False)
        summary.append((site, g["unit_number"].nunique(), len(train), len(test)))

    print(f"{'client':<15}{'bearings/units':<16}{'train rows':<12}{'test rows':<12}")
    for site, n_groups, n_train, n_test in summary:
        print(f"{site:<15}{n_groups:<16}{n_train:<12}{n_test:<12}")


if __name__ == "__main__":
    main()
