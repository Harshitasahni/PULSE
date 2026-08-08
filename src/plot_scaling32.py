#!/usr/bin/env python3
"""Plot GSM coordination overhead for 32 concurrent consumers 
(median & p99 lock wait) vs concurrency.

Usage:
    python plot_scaling32.py /path/to/data
    python plot_scaling32.py /path/to/data --save graph.png
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- CONFIG ----
GLOB_PATTERN   = "active*_run*/instrumentation_logs/active*"
GSM_WAIT_SCALE = 1e6      # seconds -> µs

TS_COL, KIND_COL, N_ACTIVE_COL = "t", "kind", "n_active"
GSM_KIND, SNAP_KIND            = "gsm_lock", "active_snapshot"

RUN_RE = re.compile(r"active[_]?(\d+)_run[_]?(\d+)")


def parse_run_name(path):
    m = RUN_RE.search(path)
    if not m:
        return None, None, os.path.basename(os.path.dirname(os.path.dirname(path)))
    a, r = int(m.group(1)), int(m.group(2))
    return a, r, f"active{a}_run_{r}"


def find_productive_phase(snap, cap, tol=1):
    thresh = cap - tol
    at_cap = snap[snap[N_ACTIVE_COL] >= thresh]
    if at_cap.empty:
        return snap[TS_COL].iloc[0], snap[TS_COL].iloc[-1]
    return at_cap[TS_COL].iloc[0], at_cap[TS_COL].iloc[-1]


def gsm_stats(df, cap):
    """median & p99 GSM Per window GSM cost(µs)"""
    df = df.sort_values(TS_COL)
    snap = df[df[KIND_COL] == SNAP_KIND][[TS_COL, N_ACTIVE_COL]].dropna()
    if snap.empty:
        return np.nan, np.nan
    t0, t1 = find_productive_phase(snap, cap)
    g = df[(df[KIND_COL] == GSM_KIND) & (df[TS_COL] >= t0) & (df[TS_COL] <= t1)]
    w = g["wait"].dropna() * GSM_WAIT_SCALE
    if len(w) == 0:
        return np.nan, np.nan
    return float(w.median()), float(np.percentile(w, 99))


def main():
    parser = argparse.ArgumentParser(description="Plot GSM overhead vs concurrency.")
    parser.add_argument("--base_dir", help="Directory containing the active*_run* folders")
    parser.add_argument("--save", default="graph.png", help="Path to save the figure (PNG)")
    parser.add_argument("--pattern", default=GLOB_PATTERN, help="Glob pattern for CSV files")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.base_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched {args.pattern!r} under {args.base_dir!r}")

    rows = []
    for p in paths:
        active, rep, name = parse_run_name(p)
        try:
            raw = pd.read_csv(p, on_bad_lines="skip")
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        rs = raw[raw[KIND_COL] == "run_start"]
        cap = (int(rs["active_cap"].dropna().iloc[0])
               if (len(rs) and rs["active_cap"].notna().any()) else active)
        med, p99 = gsm_stats(raw, cap)
        rows.append({"active": active if active is not None else cap, "rep": rep,
                     "gsm_median_us": med, "gsm_p99_us": p99})

    per_run = pd.DataFrame(rows).sort_values(["active", "rep"]).reset_index(drop=True)

    _raw0 = pd.read_csv(paths[0], on_bad_lines="skip")
    print("GSM wait median (raw units):",
          _raw0[_raw0[KIND_COL] == GSM_KIND]["wait"].median(),
          "-> if ~15, set GSM_WAIT_SCALE=1.0")

    agg = per_run.groupby("active").agg(
        n=("active", "size"),
        gsm_median_us_mean=("gsm_median_us", "mean"),
        gsm_median_us_std=("gsm_median_us", "std"),
        gsm_p99_us_mean=("gsm_p99_us", "mean"),
        gsm_p99_us_std=("gsm_p99_us", "std"),
    ).reset_index()
    print(agg.to_string(index=False))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6, 4))
    x = agg["active"]
    ax.errorbar(x, agg["gsm_median_us_mean"], yerr=agg["gsm_median_us_std"].fillna(0),
                fmt='^-', capsize=3, label='median', color='#1b9e77')
    ax.errorbar(x, agg["gsm_p99_us_mean"], yerr=agg["gsm_p99_us_std"].fillna(0),
                fmt='s--', capsize=3, label='p99', color='#7fcdbb')
    ax.set_xlabel('Active trajectories')
    ax.set_ylabel('Per window GSM cost (µs)')
    ax.set_xticks(x)
    ax.set_ylim([0, 70])
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()