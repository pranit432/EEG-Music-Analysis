"""Pages and cached data access for the Streamlit app (entry point: app.py).

Importing this module has no side effects, so each page can be tested on its own."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from band_power import BAD_CH_MAD, BANDS, REJECT_UV, clean, summarise
from group_analysis import METRICS, TRIAL, music_span
from load_edf import load
from sonify import make_notes, render, subject_features, window_features

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/tempo/music1"
RES = ROOT / "results"
BAND_NAMES = list(BANDS)
ACCENT, GREY = "#d6336c", "#9aa5b1"
ZERO_MEANINGFUL = {"beta_alpha", "theta_beta", "faa_F4_F3"}  # ln(ratio) = 0 means equal power; asymmetry 0 means none
SUBJECT_LINE = "#6f9fd8"  # soft blue: readable on the dark background, distinct from the white mean line
DEFAULT_SUBJECT = "S09"  # the largest baseline-vs-music change of the 18 subjects
METRIC_LABELS = {
    "delta_rel": "Delta (1-4 Hz), relative power", "theta_rel": "Theta (4-8 Hz), relative power",
    "alpha_rel": "Alpha (8-13 Hz), relative power", "beta_rel": "Beta (13-30 Hz), relative power",
    "gamma_rel": "Gamma (30-45 Hz), relative power", "beta_alpha": "Beta/alpha ratio (log)",
    "theta_beta": "Theta/beta ratio (log)", "faa_F4_F3": "Frontal alpha asymmetry (F4 vs F3)",
}
FEATURES = {"Alpha (relative)": "alpha", "Beta/alpha (log)": "beta_alpha", "Amplitude (µV)": "amplitude"}

PAGE_EXPLORE = None  # set by app.py once the pages exist (used by switch_page)



# ------------------------------------------------------------------ data access (cached)
@st.cache_data
def subjects():
    return sorted(p.name for p in DATA.glob("S*") if p.is_dir())


@st.cache_data
def recordings(sub):
    return [f.name[len(sub):].split("_")[0] for f in sorted((DATA / sub).glob("*_filtered.edf"))
            if "BS" in f.name or TRIAL.match(f.name)]


@st.cache_data
def read_csv(path, mtime, **kw):
    """mtime is part of the cache key, so recomputed results show up immediately."""
    return pd.read_csv(path, **kw)


def load_results(name, **kw):
    p = RES / name
    return read_csv(str(p), p.stat().st_mtime, **kw) if p.exists() else None


@st.cache_data(show_spinner="Processing EEG...", max_entries=200)
def recording_row(sub, rec, reject_uv, bad_ch_mad):
    """Band-power summary of one recording (music trials are cropped to the music itself)."""
    raw = load(DATA / sub / f"{sub}{rec}_filtered.edf")
    if rec != "BS":
        span = music_span(raw)
        if span is None:
            return None
        raw.crop(*span)
    raw, bad = clean(raw, bad_ch_mad)
    row = summarise(raw, bad, rec, reject_uv)
    return row if "alpha_rel" in row else None


@st.cache_data(show_spinner="Computing time course...", max_entries=200)
def time_course(sub, rec, reject_uv, bad_ch_mad):
    f = window_features(DATA / sub / f"{sub}{rec}_filtered.edf", reject_uv, bad_ch_mad)
    return pd.DataFrame(f, columns=list(FEATURES.values())).assign(window=lambda d: d.index + 1)


@st.cache_data(show_spinner="Analysing this subject's recordings...")
def sonify_features(sub):
    return subject_features(sub)


@st.cache_data(show_spinner="Rendering audio...")
def render_cached(sub, rec):
    feats, ranges = sonify_features(sub)
    return render(sub, rec, feats, ranges)


# ------------------------------------------------------------------ shared state / URL
def focus_subject():
    """Subject shared between pages, initialised from ?subject=... in the URL."""
    if "focus_sub" not in st.session_state:
        q = st.query_params.get("subject")
        st.session_state.focus_sub = q if q in subjects() else DEFAULT_SUBJECT
    return st.session_state.focus_sub


def remember_subject():
    st.session_state.focus_sub = st.session_state.exp_sub
    st.query_params["subject"] = st.session_state.exp_sub


# ------------------------------------------------------------------ pages
def explore_page():
    head, pop = st.columns([3, 1], vertical_alignment="bottom")
    head.header("Explore a recording")
    with pop.popover("Preprocessing", icon=":material/tune:"):
        reject = st.slider("Reject windows above (µV peak-to-peak)", 100, 400, REJECT_UV, 10,
                           help="A 2 s window is dropped if any channel swings more than this.")
        mad = st.slider("Drop channels noisier than (× median)", 2.0, 6.0, float(BAD_CH_MAD), 0.5,
                        help="A channel is dropped if its standard deviation exceeds this multiple of the median channel.")
        st.caption(f"Defaults ({REJECT_UV} µV, {BAD_CH_MAD}×) match the group analysis. Changes here only affect this page.")

    subs = subjects()
    c1, c2 = st.columns([1, 4])
    sub = c1.selectbox("Subject", subs, index=subs.index(focus_subject()), key="exp_sub", on_change=remember_subject)
    st.query_params["subject"] = sub  # keeps the URL shareable, including after arriving from another page
    trials = [r for r in recordings(sub) if r != "BS"]
    rec = c2.pills("Music trial", trials, default=trials[0], key=f"exp_rec_{sub}") or trials[0]

    base, row = recording_row(sub, "BS", reject, mad), recording_row(sub, rec, reject, mad)
    if base is None or row is None:
        st.warning("This recording has too little clean data (or no usable music segment) at these settings.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Alpha (relative)", f"{row['alpha_rel']:.3f}", f"{row['alpha_rel'] - base['alpha_rel']:+.3f} vs baseline",
              delta_color="off", border=True)
    m2.metric("Beta / alpha", f"{row['beta_alpha']:.2f}", f"{row['beta_alpha'] - base['beta_alpha']:+.2f} vs baseline",
              delta_color="off", border=True, help="Higher means relatively more alert, lower means relatively more relaxed.")
    m3.metric("Gamma (relative)", f"{row['gamma_rel']:.3f}", f"{row['gamma_rel'] - base['gamma_rel']:+.3f} vs baseline",
              delta_color="off", border=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Power by band")
        bars = pd.DataFrame({"band": BAND_NAMES * 2, "recording": ["Baseline"] * 5 + [rec] * 5,
                             "relative power": [base[f"{b}_rel"] for b in BAND_NAMES] + [row[f"{b}_rel"] for b in BAND_NAMES]})
        st.altair_chart(alt.Chart(bars).mark_bar().encode(
            x=alt.X("band:N", sort=BAND_NAMES, title=None, axis=alt.Axis(labelAngle=0)), xOffset="recording:N",
            y=alt.Y("relative power:Q"), color=alt.Color("recording:N", scale=alt.Scale(range=[GREY, ACCENT]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["recording", "band", alt.Tooltip("relative power:Q", format=".3f")]), width="stretch")
    with right:
        st.subheader("Over the recording")
        feature = st.segmented_control("Feature", list(FEATURES), default="Alpha (relative)", key="exp_feature") or "Alpha (relative)"
        col = FEATURES[feature]
        series = pd.concat([time_course(sub, "BS", reject, mad).assign(recording="Baseline"),
                            time_course(sub, rec, reject, mad).assign(recording=rec)])
        st.altair_chart(alt.Chart(series).mark_line(point=True).encode(
            x=alt.X("window:Q", title="Clean 2 s window"), y=alt.Y(f"{col}:Q", title=feature, scale=alt.Scale(zero=False)),
            color=alt.Color("recording:N", scale=alt.Scale(range=[GREY, ACCENT]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["recording", "window", alt.Tooltip(f"{col}:Q", format=".3f", title=feature)]), width="stretch")

    st.dataframe(pd.DataFrame({"Baseline": [base["bad_ch"], base["win_kept"]], rec: [row["bad_ch"], row["win_kept"]]},
                              index=["Channels dropped as noisy", "Clean 2 s windows kept"]), width="stretch")
    st.caption("Relative power = share of 1-45 Hz power in each band. A single trial can differ from the group trend, "
               "so see *Baseline vs music* for all 18 subjects.")


def group_page():
    st.header("Baseline vs music")
    if st.session_state.pop("recomputed", False):
        st.toast("Group results updated", icon=":material/check_circle:")
    means, stats = load_results("group_subject_means.csv", index_col="subject"), load_results("group_stats.csv", index_col="metric")
    if means is None or stats is None:
        st.info("No group results yet. Open the *Recompute* section below to create them.")
    else:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        metric = c1.pills("Metric", METRICS, format_func=METRIC_LABELS.get, default="alpha_rel", key="grp_metric") or "alpha_rel"
        highlight = c2.selectbox("Highlight a subject", ["None"] + list(means.index), key="grp_highlight")
        s = stats.loc[metric]

        m1, m2, m3 = st.columns(3)
        m1.metric("Subjects higher in music", f"{int(s.n_music_higher)} of {int(s.n)}", border=True)
        m2.metric("Wilcoxon p", f"{s.wilcoxon_p:.4f}", border=True)
        m3.metric("Holm-corrected p", f"{s.wilcoxon_p_holm:.4f}", border=True,
                  help="Corrected for testing 8 metrics at once.")
        if s.wilcoxon_p_holm < 0.05:
            st.badge("Significant after correction", icon=":material/check:", color="green")
        else:
            st.badge("Not significant after correction", icon=":material/close:", color="gray")

        pair = means[[f"{metric}_base", f"{metric}_music"]].dropna()
        long = (pair.rename(columns={f"{metric}_base": "Baseline", f"{metric}_music": "Music"}).rename_axis("subject")
                .reset_index().melt("subject", var_name="condition", value_name="value"))
        long["highlighted"] = long["subject"] == highlight
        order = alt.X("condition:N", sort=["Baseline", "Music"], title=None, axis=alt.Axis(labelAngle=0))
        lines = alt.Chart(long).mark_line(point=True).encode(
            x=order, y=alt.Y("value:Q", title=METRIC_LABELS[metric], scale=alt.Scale(zero=False)), detail="subject:N",
            color=alt.condition("datum.highlighted", alt.value(ACCENT), alt.value(SUBJECT_LINE)),
            opacity=alt.condition("datum.highlighted", alt.value(1.0), alt.value(0.85)),
            size=alt.condition("datum.highlighted", alt.value(4), alt.value(1.8)),
            tooltip=["subject", "condition", alt.Tooltip("value:Q", format=".3f")])
        mean_line = alt.Chart(long.groupby("condition", as_index=False)["value"].mean()).mark_line(
            point=alt.OverlayMarkDef(color="#f1f3f5", size=60), color="#f1f3f5", strokeDash=[6, 3], size=3).encode(x=order, y="value:Q", tooltip=["condition", alt.Tooltip("value:Q", format=".3f")])
        layers = [lines, mean_line]
        if metric in ZERO_MEANINGFUL:  # for log ratios and asymmetry, 0 means "no difference between the two terms"
            layers.insert(0, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#868e96", strokeDash=[2, 3], opacity=0.8).encode(y="y:Q"))
        st.altair_chart(alt.layer(*layers), width="stretch")
        st.caption("Each line is one subject (the mean of their music trials vs their baseline); the dashed line is the group mean. "
                   "Ratios are log-transformed. Subjects missing a frontal channel are left out of the asymmetry metric.")
        if highlight != "None" and st.button(f"Open {highlight} in Explore", icon=":material/arrow_forward:"):
            st.session_state.focus_sub = highlight
            st.switch_page(PAGE_EXPLORE)

        st.subheader("All metrics")
        table = stats.reset_index().assign(significant=lambda d: d["wilcoxon_p_holm"] < 0.05)
        st.dataframe(table[["metric", "n", "base_mean", "music_mean", "n_music_higher", "wilcoxon_p", "wilcoxon_p_holm", "significant"]],
                     hide_index=True, width="stretch", column_config={
                         "metric": st.column_config.TextColumn("Metric"),
                         "n": st.column_config.NumberColumn("n", format="%d"),
                         "base_mean": st.column_config.NumberColumn("Baseline mean", format="%.3f"),
                         "music_mean": st.column_config.NumberColumn("Music mean", format="%.3f"),
                         "n_music_higher": st.column_config.ProgressColumn("Subjects higher in music", min_value=0, max_value=18, format="%d"),
                         "wilcoxon_p": st.column_config.NumberColumn("p", format="%.4f"),
                         "wilcoxon_p_holm": st.column_config.NumberColumn("p (Holm)", format="%.4f"),
                         "significant": st.column_config.CheckboxColumn("Significant")})
        st.download_button("Download stats (CSV)", stats.to_csv().encode(), "group_stats.csv", "text/csv", icon=":material/download:")

        controls = load_results("controls_stats.csv")
        if controls is not None:
            with st.expander("Controls: is it just drift or closed eyes?"):
                st.dataframe(controls, hide_index=True, width="stretch", column_config={
                    "test": st.column_config.TextColumn("Test", width="large"),
                    "wilcoxon_p": st.column_config.NumberColumn("p", format="%.4f"),
                    "a_mean": st.column_config.NumberColumn("A mean", format="%.3f"),
                    "b_mean": st.column_config.NumberColumn("B mean", format="%.3f")})
                st.caption("The alpha rise is already present in the first music trial, with no trend across trials and no drop in blink "
                           "activity, so simple drift or closed eyes don't explain it. Neither can be fully ruled out.")

    with st.expander("Recompute from the raw EEG"):
        st.write("Re-runs the group analysis and the controls on all 18 subjects (about 30 seconds).")
        if st.button("Recompute", icon=":material/refresh:"):
            import controls
            import group_analysis
            with st.status("Recomputing...", expanded=True) as status:
                st.write("Group analysis: baseline vs music...")
                group_analysis.main()
                st.write("Controls: drift and eyes-closed...")
                controls.main()
                status.update(label="Recomputed", state="complete", expanded=False)
            st.session_state.recomputed = True  # the toast is shown after the rerun, which refreshes the tables above
            st.rerun()


def listen_page():
    st.header("Listen")
    st.write("Each clean 2 s EEG window becomes one bar of four notes. **Higher alpha means higher pitch, a higher beta/alpha ratio means "
             "a faster tempo, and larger amplitude means louder.** Ranges are scaled across the subject's recordings, so differences "
             "between recordings are audible.")
    st.info("These tunes are generated from the EEG using a chosen mapping. They are **not** the music the participants heard, "
            "and the recordings (Baseline, E01, E02...) contain brain signals only.")
    subs = subjects()
    sub = st.selectbox("Subject", subs, index=subs.index(DEFAULT_SUBJECT), key="lis_sub",
                       help="S09 has the biggest baseline-vs-music contrast; S01 has almost none.")
    choices = recordings(sub)
    with st.form("listen_form"):
        picked = st.pills("Recordings to play", choices, selection_mode="multi",
                          default=[r for r in ("BS", "E01") if r in choices], key=f"lis_rec_{sub}")
        st.form_submit_button("Render", type="primary", icon=":material/music_note:")
    if not picked:
        st.info("Pick at least one recording.")
        st.stop()

    feats, ranges = sonify_features(sub)
    contour = []
    for rec in picked:
        if rec not in feats:
            st.warning(f"{rec}: too little clean data to render.")
            continue
        r = render_cached(sub, rec)
        with st.container(border=True):
            st.subheader("Baseline" if rec == "BS" else rec)
            a, b = st.columns([3, 2])
            a.audio(Path(r["wav"]).read_bytes(), format="audio/wav")
            d1, d2 = a.columns(2)
            d1.download_button("WAV", Path(r["wav"]).read_bytes(), Path(r["wav"]).name, "audio/wav", key=f"wav_{sub}_{rec}", icon=":material/download:")
            d2.download_button("MIDI", Path(r["mid"]).read_bytes(), Path(r["mid"]).name, "audio/midi", key=f"mid_{sub}_{rec}", icon=":material/download:")
            b.metric("Mean tempo", f"{r['mean_bpm']:.0f} BPM")
            b.caption(f"{r['bars']} bars, {r['seconds']:.0f} s, mean pitch (MIDI) {r['mean_pitch']:.0f}")
        notes = pd.DataFrame(make_notes(feats[rec], ranges), columns=["pitch", "velocity", "bpm", "bar", "beat"])
        contour.append(notes.groupby("bar", as_index=False).agg(pitch=("pitch", "mean"), bpm=("bpm", "mean")).assign(recording=rec))
    if contour:
        long = pd.concat(contour)
        st.subheader("How the music moves")
        c1, c2 = st.columns(2)
        for col, field, title in ((c1, "bpm", "Tempo (BPM)"), (c2, "pitch", "Mean pitch (MIDI note)")):
            col.altair_chart(alt.Chart(long).mark_line(interpolate="step-after").encode(
                x=alt.X("bar:Q", title="Bar (2 s of EEG each)"), y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(zero=False)),
                color=alt.Color("recording:N", legend=alt.Legend(orient="top", title=None)),
                tooltip=["recording", "bar", alt.Tooltip(f"{field}:Q", format=".0f")]), width="stretch")
    st.caption("The mapping from brain wave to note is a creative choice (sonification), not a research finding.")


def about_page():
    st.header("About")
    n_rec = sum(len(recordings(s)) for s in subjects())  # baselines + trials, without duplicate part-files
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Subjects", len(subjects()), border=True)
    m2.metric("Recordings", n_rec, border=True)
    m3.metric("Sampling rate", "300 Hz", border=True)
    m4.metric("EEG channels", 24, border=True, help="Some are dropped per recording when noisy.")
    st.markdown("**Question.** Does listening to music change brain-wave (EEG) power compared with a silent baseline?")
    st.markdown("**Data.** Kaggle's *Music Tempo Emotional Impact EEG Dataset*. Each subject has one baseline recording and 6-9 music trials. "
                "Music trials are cropped to the start/end markers stored in the file.")
    st.markdown("**Method.** Noisy channels are dropped and the rest are average-referenced. 2 s windows with extreme amplitude are rejected. "
                "Welch power spectra are integrated over five bands, then compared between baseline and music with paired Wilcoxon tests, "
                "Holm-corrected across 8 metrics.")
    with st.popover("Show the formulas", icon=":material/function:"):
        st.latex(r"\text{relative power}_{b} = \frac{P_b}{P_{1\text{-}45\,\mathrm{Hz}}}")
        st.latex(r"\text{beta/alpha} = \ln\frac{P_\beta}{P_\alpha}")
        st.latex(r"\text{FAA} = \ln P_\alpha(\mathrm{F4}) - \ln P_\alpha(\mathrm{F3})")
    st.markdown("**Main result.** Alpha power is higher, and the beta/alpha ratio and gamma power are lower, during music "
                "(16 of 18 subjects show higher alpha).")
    st.info("**How this relates to the original proposal.** The proposal planned to compare music genres and to report attention and "
            "relaxation indices. The public dataset used here varies music *tempo*, and which trial used which tempo or piece is not "
            "recorded in the files, so neither genre nor tempo conditions can be compared. What this analysis shows is music versus a "
            "silent baseline, with all music trials pooled. Only EEG markers were measured (no reaction-time, memory or stress-rating "
            "data), so relaxation is inferred from alpha and beta/alpha, and the attention marker (theta/beta) showed no change. "
            "Comparing genres or tempos, ideally with a behavioural measure alongside the EEG, is the next step and needs a dataset "
            "or experiment with labelled conditions.")
    st.warning("**Limits**\n"
               "- The baseline protocol (eyes open or closed, instructions) isn't documented, so a baseline-vs-music difference can't be fully "
               "separated from other differences between the recordings.\n"
               "- Trial tempo labels aren't in the files, so tempo conditions can't be compared and all music trials are pooled.\n"
               "- Several channels (O1 and O2 in particular) are often noisy, so posterior results rest on few channels.\n"
               "- The sonification mapping is a creative choice.")
