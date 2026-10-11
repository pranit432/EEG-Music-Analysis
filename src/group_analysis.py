"""Step 1: baseline vs music (all trials pooled), paired across subjects.

For every subject in music1/: baseline = the BS recording; music = mean over that subject's
music trials, each cropped to the span between the '5 | start' and '6 | end' annotations.

Usage: python src/group_analysis.py
Writes results/group_subject_means.csv, results/group_stats.csv, results/group_baseline_vs_music.png
"""
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from band_power import clean, summarise
from load_edf import load

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
MIN_MUSIC_S = 40  # skip trials with less music than this (truncated recordings)
MIN_WINDOWS = 8  # skip recordings with fewer clean 2 s windows than this
TRIAL = re.compile(r"^S\d+E\d+_filtered\.edf$")  # excludes duplicate parts like S05E02_0001
METRICS = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel", "beta_alpha", "theta_beta", "faa_F4_F3"]
LOG_METRICS = ["beta_alpha", "theta_beta"]  # ratios are log-transformed before averaging/testing


def music_span(raw):
    a = raw.annotations
    start = [o for o, d in zip(a.onset, a.description) if "start" in d]
    end = [o for o, d in zip(a.onset, a.description) if "end" in d and o > 0]
    return (start[0], end[-1]) if start and end and end[-1] - start[0] >= MIN_MUSIC_S else None


def measure(path, crop):
    raw = load(path)
    if crop:
        span = music_span(raw)
        if span is None:
            return None, "no usable music span"
        raw.crop(*span)
    raw, bad = clean(raw)
    row = summarise(raw, bad, path.stem)
    kept = int(row["win_kept"].split("/")[0])
    if kept < MIN_WINDOWS:
        return None, f"only {kept} clean windows"
    for m in LOG_METRICS:
        row[m] = np.log(row[m])
    return {m: row[m] for m in METRICS}, None


def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1)
    return adj


def main():
    rows, skipped = [], []
    for sdir in sorted((ROOT / "data/tempo/music1").glob("S*")):
        sub = sdir.name
        base, why = measure(sdir / f"{sub}BS_filtered.edf", crop=False)
        if base is None:
            skipped.append((sub, "BS", why))
            continue
        trials = []
        for f in sorted(sdir.glob("*_filtered.edf")):
            if "BS" in f.name or not TRIAL.match(f.name):
                continue
            m, why = measure(f, crop=True)
            if m is None:
                skipped.append((sub, f.name[:6], why))
            else:
                trials.append(m)
        if not trials:
            skipped.append((sub, "all music", "no usable trials"))
            continue
        music = pd.DataFrame(trials).mean()  # NaN-skipping mean (e.g. FAA when F3/F4 dropped)
        rows.append({"subject": sub, "n_trials": len(trials), **{f"{m}_base": base[m] for m in METRICS},
                     **{f"{m}_music": music[m] for m in METRICS}})
        print(f"{sub}: baseline + {len(trials)} trials")

    df = pd.DataFrame(rows).set_index("subject")
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "group_subject_means.csv")

    out = []
    for m in METRICS:
        pair = df[[f"{m}_base", f"{m}_music"]].dropna()
        d = pair[f"{m}_music"] - pair[f"{m}_base"]
        out.append({"metric": m, "n": len(d), "base_mean": pair[f"{m}_base"].mean(), "music_mean": pair[f"{m}_music"].mean(),
                    "mean_diff": d.mean(), "dz": d.mean() / d.std(ddof=1),
                    "t_p": stats.ttest_rel(pair[f"{m}_music"], pair[f"{m}_base"]).pvalue,
                    "wilcoxon_p": stats.wilcoxon(d).pvalue, "n_music_higher": int((d > 0).sum())})
    st = pd.DataFrame(out).set_index("metric")
    st["wilcoxon_p_holm"] = holm(st["wilcoxon_p"].to_numpy())
    st.to_csv(RES / "group_stats.csv")
    pd.set_option("display.width", 200)
    print("\n", st.round(4).to_string())
    print("\nSkipped:", *skipped, sep="\n  ") if skipped else None

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, m in zip(axes.flat, METRICS):
        pair = df[[f"{m}_base", f"{m}_music"]].dropna()
        for _, r in pair.iterrows():
            ax.plot([0, 1], r.values, color="grey", alpha=0.5, lw=1)
        ax.plot([0, 1], pair.mean().values, color="crimson", lw=3)
        ax.set_xticks([0, 1], ["baseline", "music"])
        ax.set_title(f"{m}\nWilcoxon p={st.loc[m, 'wilcoxon_p']:.3f} (Holm {st.loc[m, 'wilcoxon_p_holm']:.3f})", fontsize=10)
    fig.suptitle(f"Baseline vs music, paired by subject (n={len(df)}); red = group mean", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "group_baseline_vs_music.png", dpi=130)


if __name__ == "__main__":
    main()
