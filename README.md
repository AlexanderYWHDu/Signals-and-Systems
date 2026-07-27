# Signals and Systems — Python Practice

Scientific Python worked through as coding practice for a Signals and Systems course. Five
notebooks, each covering one library, all of them using signals-and-systems problems as the
worked examples rather than generic tutorial data.

The framing throughout: **MATLAB is the classroom tool, but `numpy` + `scipy.signal` do the
same job and are what you would reach for outside a classroom.** `signal.butter`, `freqz`,
`lfilter`, and `TransferFunction` map almost one-to-one onto the MATLAB functions of the same
name.

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 01 | [NumPy for signals](notebooks/01_numpy_for_signals.ipynb) | Time vectors, sampling and Nyquist, standard signals, convolution, correlation, the FFT, aliasing, the DFT matrix, state-space eigenvalues |
| 02 | [Matplotlib for signals](notebooks/02_matplotlib_for_signals.ipynb) | Figure/Axes, `stem` for discrete signals, multi-panel layouts, dB and log axes, **Bode plots**, **pole-zero maps**, spectrograms, annotation, animated convolution |
| 03 | [SciPy signal toolbox](notebooks/03_scipy_signal_toolbox.ipynb) | LTI objects, impulse/step/`lsim`, `freqz`/`bode`, IIR and FIR design, `lfilter` vs `filtfilt`, second-order sections, windows, Welch and STFT, resampling, Hilbert, peak finding, a full ECG cleaning case study |
| 04 | [Pandas for signal data](notebooks/04_pandas_for_signal_data.ipynb) | Loading messy instrument logs, sentinels and dropouts, `DatetimeIndex` and `resample`, rolling windows, bridging to SciPy, `groupby` over experiment sweeps, wide/long reshaping |
| 05 | [Seaborn for signal analysis](notebooks/05_seaborn_for_signal_analysis.ipynb) | Where seaborn helps (distributions, repeated trials, categorical comparisons, facets) and — explicitly — where it does not |

Work them in order. 01 and 02 are prerequisites for the rest; 03 is the one that maps most
directly onto course content; 04 and 05 are about handling real measurements and comparing
experiments.

Every notebook ends with exercises that have **no published solutions**. Verifying your own
answer numerically is the skill being practised.

## Data

`data/` holds synthetic-but-realistic recordings. They are deliberately imperfect — noise,
drift, dropouts, sentinel values, comment banners mid-file — because clean data teaches you
nothing about cleaning data.

| File | Contents |
|------|----------|
| `ecg_like.csv` | 20 s single-channel ECG at 360 Hz, with baseline wander, 50 Hz mains hum, and broadband noise |
| `accelerometer.csv` | 120 s three-axis accelerometer at 100 Hz, timestamped, with random dropouts, a 2 s logger stall, and a 24 Hz vibration burst |
| `bench_capture.txt` | 4000 samples from a fictional DAQ: `;` delimiters, comment headers, keepalive banners mid-file, `NaN` and `-999.0` sentinels, a status column |
| `filter_sweep.csv` | 750 rows of filter benchmark results — 5 families × 5 orders × 5 cutoffs × 6 trials, tidy long format |

Regenerate them all (deterministic, fixed seed) with:

```bash
python data/generate_data.py
```

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Then:

```bash
jupyter lab
```

If you already have an Anaconda environment, everything needed is almost certainly installed
already — check with:

```bash
python -c "import numpy, scipy, matplotlib, pandas, seaborn; print('all present')"
```

## Notes

- Notebooks are committed **without outputs** so diffs stay readable. Run them yourself.
- Notebooks write scratch files to `scratch/`, which is gitignored.
- Verified end-to-end against numpy 1.26, scipy 1.16, matplotlib 3.10, pandas 2.3, seaborn 0.13.
