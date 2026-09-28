"""
Step 1 - Data preprocessing.

PRONOSTIA/FEMTO bearing data -> femto_features_labeled.csv
NASA C-MAPSS turbofan data   -> cmapss_train_labeled.csv, cmapss_test_labeled.csv

Run directly: python scripts/01_preprocess.py
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

# ---- configuration (edit these paths if your layout differs) ----
FEMTO_ROOT = "data/femto"
CMAPSS_ROOT = "data/cmapss"
OUT_DIR = "output"

FEMTO_SITE_MAP = {"Bearing1": "condition_1", "Bearing2": "condition_2", "Bearing3": "condition_3"}
CMAPSS_SITES = ["FD001", "FD002", "FD003", "FD004"]
CMAPSS_COLS = (
    ["unit_number", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
RUL_CAP = 125  # standard C-MAPSS piecewise-linear RUL convention (Heimes 2008)


# ------------------------------------------------------------------
# FEMTO
# ------------------------------------------------------------------
def read_acc_file(path):
    """acc_XXXXX.csv: hour,minute,second,microsecond,horiz_accel,vert_accel.
    Most files are comma-delimited but a few (e.g. Full_Test_Set/Bearing1_4)
    ship with semicolons instead -- sniff the first line and pick the delimiter.
    """
    with open(path, "r") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    arr = np.loadtxt(path, delimiter=sep, usecols=(4, 5))
    return arr[:, 0], arr[:, 1]  # horiz, vert


def extract_features(horiz, vert):
    feats = {}
    for axis, sig in (("h", horiz), ("v", vert)):
        feats[f"rms_{axis}"] = float(np.sqrt(np.mean(sig ** 2)))
        feats[f"peak_{axis}"] = float(np.max(np.abs(sig)))
        feats[f"kurtosis_{axis}"] = float(kurtosis(sig))
        feats[f"skew_{axis}"] = float(skew(sig))
        feats[f"std_{axis}"] = float(np.std(sig))
    return feats


def process_femto():
    print("=== FEMTO preprocessing ===")
    rows = []
    for split in ("Learning_set", "Full_Test_Set"):
        split_dir = os.path.join(FEMTO_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"  (skipping missing {split_dir})")
            continue
        for bearing_dir in sorted(glob.glob(os.path.join(split_dir, "Bearing*"))):
            bearing_id = os.path.basename(bearing_dir)
            site = FEMTO_SITE_MAP[bearing_id.split("_")[0]]
            files = sorted(glob.glob(os.path.join(bearing_dir, "acc_*.csv")))
            n_files = len(files)
            print(f"  {split}/{bearing_id}: {n_files} files")
            for file_index, path in enumerate(files):
                horiz, vert = read_acc_file(path)
                feats = extract_features(horiz, vert)
                feats["bearing_id"] = bearing_id
                feats["site"] = site
                feats["file_index"] = file_index
                feats["RUL"] = n_files - file_index - 1
                rows.append(feats)
    df = pd.DataFrame(rows)
    cols = ["bearing_id", "site", "file_index"] + [c for c in df.columns if c not in ("bearing_id", "site", "file_index", "RUL")] + ["RUL"]
    df = df[cols]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "femto_features_labeled.csv")
    df.to_csv(out_path, index=False)
    print(f"  -> saved {out_path} ({len(df)} rows)")
    return df


# ------------------------------------------------------------------
# C-MAPSS
# ------------------------------------------------------------------
def load_cmapss_file(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLS)


def process_cmapss():
    print("\n=== C-MAPSS preprocessing ===")
    train_frames, test_frames = [], []
    sensor_cols = [c for c in CMAPSS_COLS if c.startswith("sensor_")]

    for site in CMAPSS_SITES:
        train_path = os.path.join(CMAPSS_ROOT, f"train_{site}.txt")
        test_path = os.path.join(CMAPSS_ROOT, f"test_{site}.txt")
        rul_path = os.path.join(CMAPSS_ROOT, f"RUL_{site}.txt")

        train = load_cmapss_file(train_path)
        test = load_cmapss_file(test_path)
        final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["final_rul"])

        # Train RUL: cycles remaining until the unit's last recorded cycle,
        # capped at RUL_CAP (early life is a flat "healthy" label).
        max_cycle = train.groupby("unit_number")["cycle"].transform("max")
        train["RUL"] = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)

        # Test RUL: RUL_FD00X.txt gives the true RUL at the truncation point
        # (each unit's last row). Earlier rows count backward from there,
        # capped the same way so train/test label distributions match.
        final_rul.index = final_rul.index + 1  # unit_number is 1-indexed
        test_max_cycle = test.groupby("unit_number")["cycle"].transform("max")
        unit_final_rul = test["unit_number"].map(final_rul["final_rul"])
        test["RUL"] = ((test_max_cycle - test["cycle"]) + unit_final_rul).clip(upper=RUL_CAP)

        train["site"] = site
        test["site"] = site
        train_frames.append(train)
        test_frames.append(test)

        print(f"  {site}: {train['unit_number'].nunique()} train units, "
              f"{test['unit_number'].nunique()} test units")

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    # Fit MinMaxScaler on train sensors only, apply to both splits (avoid test leakage).
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    os.makedirs(OUT_DIR, exist_ok=True)
    train_out = os.path.join(OUT_DIR, "cmapss_train_labeled.csv")
    test_out = os.path.join(OUT_DIR, "cmapss_test_labeled.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"  -> saved {train_out} ({len(train_df)} rows)")
    print(f"  -> saved {test_out} ({len(test_df)} rows)")
    return train_df, test_df


# ------------------------------------------------------------------
def verify(femto_df, cmapss_train_df, cmapss_test_df):
    print("\n=== Verification ===")
    print(f"FEMTO: {len(femto_df)} rows, {femto_df['bearing_id'].nunique()} bearings")
    print(femto_df.groupby("site")["bearing_id"].nunique().rename("bearings"))
    for site, g in femto_df.groupby("site"):
        example_bearing = g["bearing_id"].iloc[0]
        seq = g[g["bearing_id"] == example_bearing].sort_values("file_index")["RUL"]
        print(f"  {site} example ({example_bearing}) RUL: {seq.iloc[0]} -> {seq.iloc[-1]}")

    print(f"\nC-MAPSS train: {len(cmapss_train_df)} rows, "
          f"{cmapss_train_df.groupby('site')['unit_number'].nunique().to_dict()} units/site")
    print(f"C-MAPSS test: {len(cmapss_test_df)} rows, "
          f"{cmapss_test_df.groupby('site')['unit_number'].nunique().to_dict()} units/site")
    for site, g in cmapss_train_df.groupby("site"):
        example_unit = g["unit_number"].iloc[0]
        seq = g[g["unit_number"] == example_unit].sort_values("cycle")["RUL"]
        print(f"  {site} train example (unit {example_unit}) RUL: {seq.iloc[0]} -> {seq.iloc[-1]}")


if __name__ == "__main__":
    femto_df = process_femto()
    cmapss_train_df, cmapss_test_df = process_cmapss()
    verify(femto_df, cmapss_train_df, cmapss_test_df)
