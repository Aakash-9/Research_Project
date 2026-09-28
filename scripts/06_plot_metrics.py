"""
Step 6 (reporting) - render the Step 5 verification metrics and Step 4
training curves as a single PNG summary figure.

Not part of the core pipeline (01-05); this is a one-off report generator
that reads digital_twins/results.json + federated_clients/*/test.csv and,
optionally, a captured training-log text file for per-epoch curves.

Run directly: python scripts/06_plot_metrics.py
"""
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_JSON = "digital_twins/results.json"
CLIENTS_DIR = "federated_clients"
TRAIN_LOG = "training_log.txt"  # plain-text capture of scripts/04_train.py stdout
OUT_PNG = "output/pipeline_metrics.png"

# Validated 7-slot categorical order (blue, orange, aqua, yellow, magenta, green, violet)
# from the dataviz skill's reference palette -- passes CVD + normal-vision checks in this order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
CMAPSS_SITES = ["FD001", "FD002", "FD003", "FD004"]
FEMTO_SITES = ["condition_1", "condition_2", "condition_3"]
SITE_ORDER = CMAPSS_SITES + FEMTO_SITES


def baseline_rmse(site_name):
    df = pd.read_csv(f"{CLIENTS_DIR}/{site_name}/test.csv")
    mean_rul = df["RUL"].mean()
    return float(np.sqrt(np.mean((df["RUL"] - mean_rul) ** 2)))


def parse_training_log(path):
    """Pull (site -> [(epoch, rmse), ...]) out of 04_train.py's printed log."""
    curves = {}
    current_site = None
    site_header = re.compile(r"^--- (\S+):")
    epoch_line = re.compile(r"^\s*epoch\s+(\d+)\s+test RMSE\s+([\d.]+)")
    with open(path) as f:
        for line in f:
            m = site_header.match(line)
            if m:
                current_site = m.group(1)
                curves[current_site] = []
                continue
            m = epoch_line.match(line)
            if m and current_site:
                curves[current_site].append((int(m.group(1)), float(m.group(2))))
    return curves


def main():
    with open(RESULTS_JSON) as f:
        results = {r["site"]: r for r in json.load(f)}
    sites = [s for s in SITE_ORDER if s in results]
    colors = dict(zip(sites, PALETTE))
    baselines = {s: baseline_rmse(s) for s in sites}

    try:
        curves = parse_training_log(TRAIN_LOG)
    except FileNotFoundError:
        curves = {}
        print(f"(no {TRAIN_LOG} found -- skipping training-curve panels; "
              f"bar chart will still be produced)")

    fig = plt.figure(figsize=(13, 9), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.28,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    # ---- Panel 1 (top, full width): final test RMSE vs constant-mean baseline ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor("#fcfcfb")
    x = np.arange(len(sites))
    width = 0.36
    model_vals = [results[s]["test_rmse"] for s in sites]
    base_vals = [baselines[s] for s in sites]

    bars_model = ax0.bar(x - width / 2, model_vals, width, color=[colors[s] for s in sites],
                          label="digital twin (test RMSE)", zorder=3)
    bars_base = ax0.bar(x + width / 2, base_vals, width, color="none",
                         edgecolor=MUTED, hatch="////", linewidth=1.2,
                         label="constant-mean baseline", zorder=3)

    y_range = max(base_vals) * 1.25  # headroom for value labels + status annotations
    label_gap = y_range * 0.015
    ax0.set_ylim(0, y_range)

    for b, v in zip(bars_model, model_vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + label_gap, f"{v:,.0f}", ha="center",
                  va="bottom", fontsize=9, color=INK)
    for b, v in zip(bars_base, base_vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + label_gap, f"{v:,.0f}", ha="center",
                  va="bottom", fontsize=9, color=MUTED)

    ax0.set_xticks(x)
    ax0.set_xticklabels(sites)
    ax0.set_ylabel("RUL RMSE")
    ax0.set_title("Step 5 verification — test RMSE vs. constant-mean baseline, per site",
                   fontsize=12, color=INK, loc="left", pad=12)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.spines[["left", "bottom"]].set_color(MUTED)
    ax0.tick_params(colors=MUTED)
    ax0.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax0.set_axisbelow(True)
    ax0.legend(frameon=False, loc="upper left", fontsize=9)

    # flag non-converged sites (model not >=20% better than baseline) with a fixed
    # headroom offset (axes fraction), independent of each group's own bar height,
    # so short bars (condition_3) don't collide with their own value labels.
    blended = ax0.get_xaxis_transform()  # x in data coords, y in axes fraction
    for xi, s in zip(x, sites):
        if results[s]["test_rmse"] >= baselines[s] * 0.8:
            ax0.text(xi, 0.94, "did not converge", transform=blended,
                      ha="center", fontsize=8, color="#d03b3b", style="italic")

    # ---- Panel 2 (bottom-left): C-MAPSS training curves ----
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor("#fcfcfb")
    for s in CMAPSS_SITES:
        if s not in curves:
            continue
        epochs, rmses = zip(*curves[s])
        ax1.plot(epochs, rmses, color=colors[s], linewidth=2, marker="o", markersize=3, label=s)
    ax1.set_title("C-MAPSS sites — test RMSE per epoch", fontsize=11, color=INK, loc="left")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test RMSE")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(MUTED)
    ax1.tick_params(colors=MUTED)
    ax1.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=9)

    # ---- Panel 3 (bottom-right): FEMTO training curves ----
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor("#fcfcfb")
    for s in FEMTO_SITES:
        if s not in curves:
            continue
        epochs, rmses = zip(*curves[s])
        ax2.plot(epochs, rmses, color=colors[s], linewidth=2, marker="o", markersize=3, label=s)
    ax2.set_title("FEMTO sites — test RMSE per epoch", fontsize=11, color=INK, loc="left")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test RMSE (raw file-count units)")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(MUTED)
    ax2.tick_params(colors=MUTED)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle("Phase 1 digital twin pipeline — training & verification metrics",
                 fontsize=14, color=INK, x=0.07, ha="left", y=0.97)

    fig.savefig(OUT_PNG, dpi=150, facecolor=fig.get_facecolor())
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
