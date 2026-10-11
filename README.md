# Effect of Music on EEG Brain Waves

A band-power analysis of EEG recorded while people listen to music, with a Streamlit app to explore the results and a small demo that turns EEG into sound.

**Question.** Does listening to music change brain-wave power compared with a silent baseline?

**Short answer.** In 16 of 18 subjects, alpha power was higher during music and the beta/alpha ratio was lower, a pattern usually read as more relaxed wakefulness. The data cannot show that music *caused* it, and it cannot compare music types. See [Limits](#limits).

## Contents

- [Results](#results)
- [The app](#the-app)
- [How it works](#how-it-works)
- [Data](#data)
- [Limits](#limits)
- [Run it](#run-it)
- [Reproduce the analysis](#reproduce-the-analysis)
- [Project layout](#project-layout)
- [Deploying](#deploying)

## Results

Each subject's baseline is compared with the mean of their own music trials (18 baselines, 139 music trials). Tests are paired Wilcoxon signed-rank, Holm-corrected across the 8 metrics.

| Metric | Baseline | Music | Subjects higher in music | p (Holm) |
|---|---|---|---|---|
| **Alpha** (relative power) | 0.086 | 0.171 | 16 of 18 | **0.0009** |
| **Beta/alpha** (log ratio) | 0.40 | -0.19 | 3 of 18 | **0.003** |
| **Gamma** (relative power) | 0.066 | 0.049 | 3 of 18 | **0.040** |
| Delta, theta, beta, theta/beta, frontal alpha asymmetry | | | | not significant |

- **Relaxation markers rose:** more alpha and less beta relative to alpha.
- **The attention marker did not change:** theta/beta, p of about 0.9.
- **People differ a lot.** S09's alpha rose from 0.085 to 0.56, while S01 barely changed.

**Controls.**
- *Drift:* the alpha rise is already present in the first music trial and does not keep growing across trials, so slow drift over the session does not explain it.
- *Closed eyes:* blink activity (Fp1/Fp2) did not drop during music, but alpha rose more at the back of the head than at the front (posterior/frontal alpha ratio 1.25 to 1.49, 15 of 18 subjects, uncorrected p of about 0.006), which is also what closing the eyes does. This explanation is **not** ruled out.

## The app

Run it and use the sidebar to move between four pages:

- **Explore a recording:** pick a subject and a music trial, and see band power next to that subject's baseline, plus a time course over the recording. A *Preprocessing* menu lets you change the artifact thresholds and see how sensitive a result is.
- **Baseline vs music:** the group result for any metric, with a paired plot, optional subject highlight, the full statistics table, the controls, and a button to recompute everything from the raw EEG.
- **Listen:** the sonification demo (below).
- **About:** method, data, how this relates to the original proposal, and the limits.

Views are linkable, for example `?subject=S09` on the Explore page.

### Sonification

The Listen page turns EEG into a short tune. Each clean 2 s window becomes one bar of four notes:

| EEG measure | Sound |
|---|---|
| Relative alpha power | Pitch (higher alpha, higher notes; C major pentatonic) |
| Beta/alpha ratio | Tempo (higher ratio, faster; 60-140 BPM) |
| Signal amplitude | Loudness |

Ranges are scaled across each subject's recordings, so BPM values cannot be compared between subjects. **The audio is generated from the EEG with a mapping I chose. It is not the music the participants heard** (the dataset contains brain signals only), and the mapping is a creative choice, not a finding. Tunes can be downloaded as WAV or MIDI.

## How it works

1. **Load** each recording (24-channel EEG, 300 Hz, already band-pass filtered 1-50 Hz).
2. **Crop** music trials to the span between the file's start and end markers.
3. **Clean:** keep scalp EEG, drop channels whose standard deviation exceeds 3 times the median (detected on an average-referenced signal, repeated up to 3 times), average-reference the rest.
4. **Window and reject:** split into 2 s windows (2 s trimmed from each end) and drop windows with a peak-to-peak swing above 200 uV on any channel.
5. **Band power:** Welch power spectra integrated over delta (1-4 Hz), theta (4-8), alpha (8-13), beta (13-30) and gamma (30-45). Relative power is each band's share of the 1-45 Hz total.
6. **Metrics:** relative power per band, ln(beta/alpha), ln(theta/beta) and frontal alpha asymmetry, ln alpha(F4) minus ln alpha(F3).
7. **Statistics:** each subject's baseline against the mean of their music trials, paired Wilcoxon, Holm-corrected.

Recordings with fewer than 8 clean windows, and music segments shorter than 40 s, are skipped (5 trials in total).

## Data

[Music Tempo Emotional Impact EEG Dataset](https://www.kaggle.com/datasets/aphros/music-tempo-emotional-impact-eeg-dataset) on Kaggle, by `aphros` (Apache-2.0). Participants were recorded in a quiet setting with a DSI-24 headset. 18 subjects, each with one baseline recording (`BS`) and 6-9 music trials (`E01`, `E02`, ...).

This repository includes only the `music1` filtered EDF files the app uses (about 190 MB). The `music2` folder, the raw/unfiltered files and the CSV versions are not included.

## Limits

- **The baseline protocol is not documented** (eyes open or closed, instructions given), so a baseline-versus-music difference cannot be fully separated from other differences between the recordings.
- **Which trial used which tempo or piece is not in the files.** The dataset description mentions two pieces at different tempos, but the labels are not released, so all music trials are pooled and no genre or tempo comparison is possible.
- **Only EEG was measured.** There is no reaction-time, memory or stress-rating data, so relaxation is inferred from EEG markers.
- **Electrode quality varies.** Noisy channels are dropped per recording. Frontal channels (F7, F8, Fp1, Fp2) are dropped in about 20% of recordings and occipital O1/O2 in about 10%, so regional results rest on different channels in different recordings.
- **The sample is small** (18 subjects) and effect sizes vary widely between people.

**Relation to the original proposal.** The proposal planned to compare music genres. This dataset varies tempo and lacks trial labels, so genre comparison is left as the next step: it needs a dataset or experiment with labelled conditions, ideally with a behavioural measure alongside the EEG.

## Run it

Python 3.12 or newer is recommended (developed on 3.13).

```
pip install -r requirements.txt
python -m streamlit run src/app.py
```

The app opens at `http://localhost:8501`. On Windows PowerShell, run the two commands on separate lines (`&&` is not supported in older versions).

Tests (six page-level checks, run with Streamlit's `AppTest`):

```
pip install pytest
python -m pytest tests -q
```

`AppTest` cannot re-run after changing a single-select pills widget, so interactions like switching subject are checked in the browser rather than in the tests.

## Reproduce the analysis

Run from the project folder. Each takes under a minute.

```
python src/load_edf.py                 # inspect one recording
python src/band_power.py S09           # band power for one subject, all recordings
python src/group_analysis.py           # baseline vs music, writes results/group_*.csv and a plot
python src/controls.py                 # drift and eyes-closed controls
python src/sonify.py S09 BS E01        # writes results/sonify_S09_*.wav and .mid
```

The app's *Recompute* button on the group page runs the group analysis and the controls for you.

## Project layout

```
src/app.py             entry point: page config and navigation
src/views.py           the four pages and cached data access
src/band_power.py      channel and window cleaning, band power
src/group_analysis.py  baseline-vs-music statistics and plot
src/controls.py        drift and eyes-closed controls
src/sonify.py          EEG to MIDI and WAV
src/load_edf.py        recording loader
tests/test_app.py      page-level tests
results/               saved statistics (CSV) used by the app
data/tempo/music1/     EEG recordings, one folder per subject
.streamlit/config.toml dark theme and app settings
```

## Deploying

The app runs on [Streamlit Community Cloud](https://share.streamlit.io): choose this repository, branch `main` and main file `src/app.py`. Generated audio is written to `results/` at run time and is not stored in the repository.
