"""
Step 4 - Train one independent digital twin model per federated site.

No cross-site mixing: each site's model only ever sees that site's own
train/test split (federated aggregation is out of scope for this phase).

Run directly: python scripts/04_train.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import importlib.util

# "03_model.py" starts with a digit so it can't be `import`-ed normally; load by path instead.
spec = importlib.util.spec_from_file_location("digital_twin_model", os.path.join(os.path.dirname(__file__), "03_model.py"))
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)
DigitalTwinCNNLSTM = model_mod.DigitalTwinCNNLSTM
WINDOW_SIZE = model_mod.WINDOW_SIZE

CLIENTS_DIR = "federated_clients"
CHECKPOINT_DIR = "digital_twins"
RESULTS_JSON = os.path.join(CHECKPOINT_DIR, "results.json")

BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEMTO_FEATURE_COLS = ["rms_h", "peak_h", "kurtosis_h", "skew_h", "std_h",
                       "rms_v", "peak_v", "kurtosis_v", "skew_v", "std_v"]
CMAPSS_FEATURE_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]


def get_site_config(train_df):
    """FEMTO sites have a bearing_id/file_index column, C-MAPSS sites have
    unit_number/cycle -- use whichever is present to pick columns."""
    if "bearing_id" in train_df.columns:
        return dict(group_col="bearing_id", order_col="file_index", feature_cols=FEMTO_FEATURE_COLS)
    return dict(group_col="unit_number", order_col="cycle", feature_cols=CMAPSS_FEATURE_COLS)


def build_windows(df, group_col, order_col, feature_cols, window_size=WINDOW_SIZE):
    """Slide a window over each group's (bearing/unit) own sequence only --
    windows never cross group boundaries."""
    X, y = [], []
    for _, g in df.groupby(group_col):
        g = g.sort_values(order_col)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        rul = g["RUL"].to_numpy(dtype=np.float32)
        if len(g) < window_size:
            continue
        for i in range(len(g) - window_size + 1):
            X.append(feats[i:i + window_size])
            y.append(rul[i + window_size - 1])  # label = RUL at the last step of the window
    if not X:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(X), np.array(y, dtype=np.float32)


def train_one_site(site_name, train_df, test_df):
    cfg = get_site_config(train_df)
    X_train, y_train = build_windows(train_df, **cfg)
    X_test, y_test = build_windows(test_df, **cfg)
    print(f"\n--- {site_name}: {len(X_train)} train windows, {len(X_test)} test windows ---")
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  SKIPPED: not enough rows per group to form a window of size {WINDOW_SIZE}")
        return None

    X_train_t = torch.from_numpy(X_train).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    X_test_t = torch.from_numpy(X_test).to(DEVICE)
    y_test_t = torch.from_numpy(y_test).to(DEVICE)

    model = DigitalTwinCNNLSTM(n_features=len(cfg["feature_cols"])).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    epochs_since_improve = 0
    n_train = len(X_train_t)

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            pred, _ = model(X_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_pred, _ = model(X_test_t)
            test_rmse = torch.sqrt(loss_fn(test_pred, y_test_t)).item()

        if test_rmse < best_rmse - 1e-4:
            best_rmse = test_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 5 == 0 or epochs_since_improve == 0:
            print(f"  epoch {epoch:3d}  test RMSE {test_rmse:8.3f}  (best {best_rmse:8.3f})")

        if epochs_since_improve >= PATIENCE:
            print(f"  early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{site_name}_model.pt")
    torch.save({"state_dict": model.state_dict(), "n_features": len(cfg["feature_cols"]),
                "feature_cols": cfg["feature_cols"], "window_size": WINDOW_SIZE}, ckpt_path)
    print(f"  saved {ckpt_path}  final test RMSE = {best_rmse:.3f}")

    return dict(site=site_name, n_train_windows=len(X_train), n_test_windows=len(X_test), test_rmse=best_rmse)


def main():
    results = []
    for site_dir in sorted(glob.glob(os.path.join(CLIENTS_DIR, "*"))):
        site_name = os.path.basename(site_dir)
        train_df = pd.read_csv(os.path.join(site_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(site_dir, "test.csv"))
        result = train_one_site(site_name, train_df, test_df)
        if result:
            results.append(result)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved training results to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
