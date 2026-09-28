"""
Step 5 - Verification summary across all trained digital twins.

Reads digital_twins/results.json (written by 04_train.py) and prints a
per-site summary table, flagging any site whose model looks like it
failed to converge (RMSE not meaningfully better than predicting the
mean RUL of that site's own test set).

Run directly: python scripts/05_verify.py
"""
import json
import os
import glob
import pandas as pd
import numpy as np

RESULTS_JSON = "digital_twins/results.json"
CLIENTS_DIR = "federated_clients"


def baseline_rmse(site_name):
    """RMSE of always predicting the mean test RUL -- the bar a model
    must clear to prove it learned something beyond a constant guess."""
    test_path = os.path.join(CLIENTS_DIR, site_name, "test.csv")
    df = pd.read_csv(test_path)
    mean_rul = df["RUL"].mean()
    return float(np.sqrt(np.mean((df["RUL"] - mean_rul) ** 2)))


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    print(f"{'site':<15}{'train windows':<16}{'test RMSE':<12}{'baseline RMSE':<15}{'status'}")
    any_failed = False
    for r in results:
        base = baseline_rmse(r["site"])
        # a model that isn't at least 20% better than the constant-mean
        # baseline hasn't meaningfully learned the degradation trend
        converged = r["test_rmse"] < base * 0.8
        status = "OK" if converged else "FAILED TO CONVERGE"
        if not converged:
            any_failed = True
        print(f"{r['site']:<15}{r['n_train_windows']:<16}{r['test_rmse']:<12.3f}{base:<15.3f}{status}")

    print()
    if any_failed:
        print("WARNING: one or more sites failed to converge (see FAILED rows above).")
    else:
        print("All sites converged: every model beats its own constant-mean baseline by >20%.")


if __name__ == "__main__":
    main()
