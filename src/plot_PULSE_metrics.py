import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TS_COL = "t"
KIND_COL = "kind"
TRAJ_COL = "traj_id"
N_ACTIVE_COL = "n_active"

GSM_KIND = "gsm_lock"
SNAP_KIND = "active_snapshot"
PROD_KIND = "window_produced"
DONE_KIND = "window_done"

GSM_WAIT_SCALE = 1e6
file_pattern = "../Results_pulse_metrics/active*_*/instrumentation_logs/run_*_events.csv"

def find_productive_phase(snap, cap, tol=1):
    thresh = cap - tol
    at_cap = snap[snap[N_ACTIVE_COL] >= thresh]

    if at_cap.empty:
        return snap[TS_COL].iloc[0], snap[TS_COL].iloc[-1], False

    return at_cap[TS_COL].iloc[0], at_cap[TS_COL].iloc[-1], True


def peak_sustained(snap, t0, t1, min_run=3):
    w = snap[(snap[TS_COL] >= t0) & (snap[TS_COL] <= t1)]

    if w.empty:
        return np.nan

    peak = int(w[N_ACTIVE_COL].max())
    isp = (w[N_ACTIVE_COL] == peak).astype(int).values

    best = run = 0

    for v in isp:
        run = run + 1 if v else 0
        best = max(best, run)

    return peak if best >= min_run else peak - 1


def tw_mean_conc(snap, t0, t1):
    w = snap[(snap[TS_COL] >= t0) & (snap[TS_COL] <= t1)]

    if len(w) < 2:
        return float(w[N_ACTIVE_COL].mean()) if len(w) else np.nan

    t = w[TS_COL].values
    n = w[N_ACTIVE_COL].values
    dt = np.diff(t)

    return float(np.sum(n[:-1] * dt) / np.sum(dt))


def med_p99(s):
    s = s.dropna().values

    if len(s) == 0:
        return np.nan, np.nan

    return float(np.median(s)), float(np.percentile(s, 99))


def drop_first_per_traj(d):
    return (
        d.sort_values(TS_COL)
        .groupby(TRAJ_COL, group_keys=False)
        .apply(lambda g: g.iloc[1:])
    )


def analyze_run_productive(df, cap, tol=1):
    df = df.sort_values(TS_COL)

    snap = (
        df[df[KIND_COL] == SNAP_KIND][[TS_COL, N_ACTIVE_COL]]
        .dropna()
    )

    t0, t1, reached = find_productive_phase(snap, cap, tol)
    dur = t1 - t0

    inp = lambda d: d[(d[TS_COL] >= t0) & (d[TS_COL] <= t1)]

    done = inp(
        drop_first_per_traj(
            df[df[KIND_COL] == DONE_KIND]
        )
    )

    an_med, an_p99 = med_p99(done["analysis_time"])
    q_med, _ = med_p99(done["queue_wait"])

    sim_med, _ = med_p99(
        inp(df[df[KIND_COL] == PROD_KIND])["sim_time"]
    )

    gsm = inp(df[df[KIND_COL] == GSM_KIND])
    gw = gsm["wait"] * GSM_WAIT_SCALE

    g_med, g_p99 = med_p99(gw)
    g_max = float(gw.max()) if len(gw) else np.nan

    mean_c = tw_mean_conc(snap, t0, t1)

    return {
        "cap": cap,
        "reached_cap": reached,
        "productive_dur_s": dur,
        "peak_sustained_concurrency": peak_sustained(snap, t0, t1),
        "mean_productive_concurrency": mean_c,
        "gsm_median_us": g_med,
        "gsm_p99_us": g_p99,
        "gsm_max_us": g_max,
        "analysis_median_s": an_med,
        "analysis_p99_s": an_p99,
        "queue_median_s": q_med,
        "sim_median_s": sim_med,
        "window_throughput": len(done) / dur if dur > 0 else np.nan,
        "agg_analysis_throughput": mean_c / an_med if an_med else np.nan,
        "n_windows_productive": len(done),
    }


def build_master_table(run_rows):
    df = pd.DataFrame(run_rows)

    metrics = [
        "mean_productive_concurrency",
        "gsm_median_us",
        "gsm_p99_us",
        "analysis_median_s",
        "sim_median_s",
        "agg_analysis_throughput",
        "queue_median_s",
    ]

    agg = df.groupby("active").agg(
        n=("active", "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    ).reset_index()

    return agg


def format_for_paper(agg):
    def pm(mean, std):
        return f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f} ± {std:.2f}"

    out = pd.DataFrame()

    out["active"] = agg["active"]
    out["n"] = agg["n"]

    out["concurrency"] = [
        pm(m, s)
        for m, s in zip(
            agg["mean_productive_concurrency_mean"],
            agg["mean_productive_concurrency_std"],
        )
    ]

    out["GSM med (µs)"] = [
        pm(m, s)
        for m, s in zip(
            agg["gsm_median_us_mean"],
            agg["gsm_median_us_std"],
        )
    ]

    out["GSM p99 (µs)"] = [
        pm(m, s)
        for m, s in zip(
            agg["gsm_p99_us_mean"],
            agg["gsm_p99_us_std"],
        )
    ]

    out["analysis med (s)"] = [
        pm(m, s)
        for m, s in zip(
            agg["analysis_median_s_mean"],
            agg["analysis_median_s_std"],
        )
    ]

    out["sim med (s)"] = [
        pm(m, s)
        for m, s in zip(
            agg["sim_median_s_mean"],
            agg["sim_median_s_std"],
        )
    ]

    out["throughput"] = [
        pm(m, s)
        for m, s in zip(
            agg["agg_analysis_throughput_mean"],
            agg["agg_analysis_throughput_std"],
        )
    ]

    out["queue_median_s (s)"] = [
        pm(m, s)
        for m, s in zip(
            agg["queue_median_s_mean"],
            agg["queue_median_s_std"],
        )
    ]

    return out




csv_list = []

for path in sorted(glob.glob(file_pattern)):
    match = re.search(r"active(\d+)_(\d+)", path)

    if match:
        active = int(match.group(1))
        runnum = int(match.group(2))
        csv_list.append((path, active))
    else:
      print("no active found", match)


run_rows = []

for path, active in csv_list:
    df = pd.read_csv(path)
    row = analyze_run_productive(df, active)
    row["active"] = active
    run_rows.append(row)


agg = build_master_table(run_rows)

print(format_for_paper(agg).to_string(index=False))


x = agg["active"]
labels = agg["active"]


fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar(
    x,
    agg["gsm_median_us_mean"],
    yerr=agg["gsm_median_us_std"].fillna(0),
    fmt="^-",
    capsize=3,
    label="median",
    color="#1b9e77",
)
ax.errorbar(
    x,
    agg["gsm_p99_us_mean"],
    yerr=agg["gsm_p99_us_std"].fillna(0),
    fmt="s--",
    capsize=3,
    label="p99",
    color="#7fcdbb",
)
ax.set_xlabel("Active trajectories")
ax.set_ylabel("GSM lock wait (µs)")
ax.set_ylim(0, 70)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("AAgsm_coordination.png", dpi=500)


fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar(
    x,
    agg["sim_median_s_mean"],
    yerr=agg["sim_median_s_std"].fillna(0),
    fmt="o-",
    capsize=3,
    label="OpenMM sim",
    color="#d95f02",
)
ax.errorbar(
    x,
    agg["analysis_median_s_mean"],
    yerr=agg["analysis_median_s_std"].fillna(0),
    fmt="s-",
    capsize=3,
    label="PULSE analysis",
    color="#7570b3",
)
ax.errorbar(
    x,
    agg["gsm_median_us_mean"] / 1e6,
    yerr=agg["gsm_median_us_std"].fillna(0) / 1e6,
    fmt="^-",
    capsize=3,
    label="PULSE GSM",
    color="#1b9e77",
)
ax.set_xlabel("Active trajectories")
ax.set_ylabel("Per-window cost")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("AAoverhead.png", dpi=300)



def count_events_in_intervals(events, intervals):
    """
    Count event rows whose timestamps fall inside any saturated interval.
    """
    if events.empty or intervals.empty:
        return 0

    count = 0
    t = events["t"].to_numpy()

    for _, row in intervals.iterrows():
        count += np.sum((t >= row["start"]) & (t < row["end"]))

    return int(count)


def saturated_intervals(df, active_cap, tol=1):
    snap = df[df["kind"] == "active_snapshot"].copy()

    if snap.empty:
        return pd.DataFrame(columns=["start", "end", "duration"])

    snap = snap.sort_values("t").reset_index(drop=True)
    snap["t_next"] = snap["t"].shift(-1)
    snap.loc[snap["t_next"].isna(), "t_next"] = df["t"].max()

    full = snap[snap["n_active"] >= active_cap - tol].copy()

    full["start"] = full["t"]
    full["end"] = full["t_next"]
    full["duration"] = full["end"] - full["start"]

    return full[
        full["duration"] > 0
    ][["start", "end", "duration"]]


def throughput_for_run(df, active_cap, min_seconds=30, min_windows=5):
    """
    Compute throughput during intervals where the requested concurrency
    condition is satisfied.
    """
    intervals = saturated_intervals(df, active_cap)
    done = df[df["kind"] == "window_done"].copy()

    sat_seconds = (
        intervals["duration"].sum()
        if not intervals.empty
        else 0.0
    )

    sat_windows = count_events_in_intervals(done, intervals)

    if sat_seconds <= 0:
        throughput = np.nan
    else:
        throughput = sat_windows / sat_seconds

    usable = (
        sat_seconds >= min_seconds
        and sat_windows >= min_windows
    )

    return {
        "active": active_cap,
        "throughput": throughput if usable else np.nan,
        "throughput_seconds": sat_seconds,
        "throughput_minutes": sat_seconds / 60.0,
        "throughput_windows": sat_windows,
        "usable": usable,
    }


def build_throughput_table(csv_list):
    rows = []

    for path, active in csv_list:
        df = pd.read_csv(path)

        result = throughput_for_run(
            df,
            active_cap=active,
            min_seconds=30,
            min_windows=5,
        )

        result["path"] = path
        rows.append(result)

    per_run = pd.DataFrame(rows)

    agg_throughput = per_run.groupby("active").agg(
        n=("throughput", "count"),
        throughput_mean=("throughput", "mean"),
        throughput_std=("throughput", "std"),
        throughput_minutes_mean=("throughput_minutes", "mean"),
        throughput_windows_mean=("throughput_windows", "mean"),
        usable_runs=("usable", "sum"),
    ).reset_index()

    return per_run, agg_throughput


per_run_throughput, agg_throughput = build_throughput_table(csv_list)

print("\nPer-run throughput:")
print(
    per_run_throughput[
        [
            "active",
            "throughput",
            "throughput_minutes",
            "throughput_windows",
            "usable",
        ]
    ].to_string(index=False)
)

print("\nAggregated throughput:")
print(agg_throughput.to_string(index=False))


def plot_throughput(agg_throughput, out="AAthroughput.png"):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    x = agg_throughput["active"].to_numpy()
    y = agg_throughput["throughput_mean"].to_numpy()
    yerr = agg_throughput["throughput_std"].fillna(0).to_numpy()

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
        linewidth=2,
    )

    ax.set_xlabel("Active trajectories")
    ax.set_ylabel("Aggregate analysis throughput (windows/s)")
    ax.grid(True, alpha=0.3)

    ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


plot_throughput(agg_throughput)