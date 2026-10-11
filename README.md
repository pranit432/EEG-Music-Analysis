# Effect of Music on EEG Brain Waves

Band-power analysis of EEG recorded while people listen to music at different tempos, with a Streamlit app to explore it.

**Question.** Does listening to music change brain-wave power compared with a silent baseline?
**Result.** Alpha power is higher, and the beta/alpha ratio and gamma power are lower, during music (16 of 18 subjects show higher alpha). See the *About* page of the app for the method and its limits.

## Run it

```
pip install -r requirements.txt
python -m streamlit run src/app.py
```

Tests: `pip install pytest` then `python -m pytest tests -q`.

## Layout

- `src/app.py`, `src/views.py`: the Streamlit app (entry point and pages)
- `src/band_power.py`: channel and window cleaning, band power
- `src/group_analysis.py`, `src/controls.py`: baseline-vs-music statistics and the drift / eyes-closed controls
- `src/sonify.py`: turns band power into music (MIDI and WAV)
- `results/`: saved statistics (CSV) used by the app
- `data/tempo/music1/`: the EEG recordings (filtered EDF)

## Data

Kaggle: *Music Tempo Emotional Impact EEG Dataset* by `aphros` (Apache-2.0). 18 subjects, 24-channel EEG at 300 Hz, one baseline recording and several music trials each.
