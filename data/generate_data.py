"""
Regenerate every raw data file in this folder.

All files are synthetic but deliberately imperfect: they carry noise, drift,
dropouts, duplicated timestamps and inconsistent formatting, because clean data
teaches you nothing about cleaning data.

Run from the repository root:

    python data/generate_data.py
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260727)


def path(name):
    return os.path.join(HERE, name)


# ---------------------------------------------------------------------------
# 1. ECG-like biosignal: single channel, 360 Hz, baseline wander + mains hum
# ---------------------------------------------------------------------------
def make_ecg():
    fs = 360.0
    duration = 20.0
    t = np.arange(0, duration, 1 / fs)

    # Beat train at a wandering heart rate (~72 bpm with respiratory variation).
    bpm = 72 + 4 * np.sin(2 * np.pi * 0.25 * t)
    phase = np.cumsum(bpm / 60.0) / fs
    beat_phase = (phase % 1.0) - 0.5

    def gauss(center, width, amp):
        return amp * np.exp(-0.5 * ((beat_phase - center) / width) ** 2)

    clean = (
        gauss(-0.12, 0.025, 0.10)   # P wave
        + gauss(-0.02, 0.008, -0.15)  # Q
        + gauss(0.00, 0.009, 1.00)    # R
        + gauss(0.02, 0.010, -0.25)   # S
        + gauss(0.16, 0.040, 0.30)    # T wave
    )

    baseline = 0.35 * np.sin(2 * np.pi * 0.22 * t) + 0.12 * np.sin(2 * np.pi * 0.05 * t)
    mains = 0.08 * np.sin(2 * np.pi * 50.0 * t + 0.6)
    noise = RNG.normal(0, 0.03, t.size)

    signal = clean + baseline + mains + noise

    df = pd.DataFrame({"time_s": np.round(t, 6), "mV": np.round(signal, 6)})
    df.to_csv(path("ecg_like.csv"), index=False)
    return len(df)


# ---------------------------------------------------------------------------
# 2. Three-axis accelerometer, 100 Hz, timestamped, with dropouts
# ---------------------------------------------------------------------------
def make_accelerometer():
    fs = 100.0
    duration = 120.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    start = pd.Timestamp("2026-03-14 09:15:00")
    stamps = start + pd.to_timedelta(t, unit="s")

    walk = 1.4 * np.sin(2 * np.pi * 1.9 * t)          # stride cadence
    bounce = 0.6 * np.sin(2 * np.pi * 3.8 * t + 0.4)  # first harmonic
    tremor = 0.25 * np.sin(2 * np.pi * 11.0 * t)

    x = walk + 0.3 * bounce + RNG.normal(0, 0.15, n)
    y = 0.5 * walk + bounce + RNG.normal(0, 0.15, n)
    z = 9.81 + 0.4 * bounce + tremor + RNG.normal(0, 0.20, n)

    # A 3 s burst of vibration halfway through (the "event" to be detected).
    burst = (t > 61.0) & (t < 64.0)
    z[burst] += 2.5 * np.sin(2 * np.pi * 24.0 * t[burst])
    x[burst] += 1.2 * np.sin(2 * np.pi * 24.0 * t[burst] + 1.1)

    df = pd.DataFrame(
        {
            "timestamp": stamps,
            "acc_x": np.round(x, 4),
            "acc_y": np.round(y, 4),
            "acc_z": np.round(z, 4),
        }
    )

    # Sensor dropouts: ~1.5% of samples lose one or more channels.
    for col in ["acc_x", "acc_y", "acc_z"]:
        idx = RNG.choice(n, size=int(0.015 * n), replace=False)
        df.loc[idx, col] = np.nan

    # A gap where the logger stalled for 2 seconds.
    gap = (t >= 88.0) & (t < 90.0)
    df = df.loc[~gap].reset_index(drop=True)

    df.to_csv(path("accelerometer.csv"), index=False)
    return len(df)


# ---------------------------------------------------------------------------
# 3. Messy instrument log: comment headers, mixed delimiters, junk rows
# ---------------------------------------------------------------------------
def make_messy_log():
    fs = 500.0
    n = 4000
    t = np.arange(n) / fs
    v = (
        2.0 * np.sin(2 * np.pi * 7.0 * t)
        + 0.8 * np.sin(2 * np.pi * 43.0 * t)
        + RNG.normal(0, 0.25, n)
    )
    temp = 21.5 + 0.004 * np.arange(n) + RNG.normal(0, 0.05, n)

    lines = [
        "# Bench capture -- DAQ-2200, firmware 4.11b",
        "# operator: A. Du",
        "# date: 2026-03-14T09:15:00+08:00",
        "# sample_rate_hz = 500",
        "# columns: index ; time_s ; channel_a_volts ; probe_temp_c ; status",
        "#",
        "# NOTE: status=OK means the sample is trustworthy. Anything else is not.",
        "",
    ]

    for i in range(n):
        status = "OK"
        va = f"{v[i]:.5f}"
        # Occasional saturation and sensor faults, flagged in the status column.
        if RNG.random() < 0.004:
            status = "SAT"
            va = f"{np.sign(v[i]) * 5.0:.5f}"
        elif RNG.random() < 0.003:
            status = "ERR"
            va = "NaN"
        elif RNG.random() < 0.002:
            status = "ERR"
            va = "-999.0"  # legacy sentinel value for "no reading"
        lines.append(f"{i} ; {t[i]:.6f} ; {va} ; {temp[i]:.3f} ; {status}")
        # Instrument re-prints its banner every 1500 rows.
        if i and i % 1500 == 0:
            lines.append("# --- keepalive, buffer flushed ---")

    with open(path("bench_capture.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n


# ---------------------------------------------------------------------------
# 4. Filter comparison sweep: tidy long-format results for pandas/seaborn
# ---------------------------------------------------------------------------
def make_filter_sweep():
    rows = []
    families = {
        "butter": (1.00, 0.00),
        "cheby1": (1.35, 0.55),
        "cheby2": (1.30, 0.30),
        "ellip": (1.70, 0.85),
        "bessel": (0.65, 0.02),
    }
    for family, (steep, ripple_scale) in families.items():
        for order in [2, 4, 6, 8, 10]:
            for cutoff in [20, 40, 60, 80, 100]:
                for trial in range(6):
                    rolloff = 6.0 * order * steep + RNG.normal(0, 1.5)
                    ripple = ripple_scale * (0.05 * order) + abs(RNG.normal(0, 0.02))
                    # Group delay grows with order and falls with cutoff.
                    delay = 1000 * order / (2 * np.pi * cutoff) + RNG.normal(0, 0.4)
                    snr = (
                        18.0
                        + 2.4 * np.log2(order)
                        - 0.035 * cutoff
                        - 9.0 * ripple
                        + RNG.normal(0, 0.8)
                    )
                    rows.append(
                        {
                            "family": family,
                            "order": order,
                            "cutoff_hz": cutoff,
                            "trial": trial,
                            "rolloff_db_per_octave": round(rolloff, 3),
                            "passband_ripple_db": round(ripple, 4),
                            "group_delay_ms": round(delay, 3),
                            "output_snr_db": round(snr, 3),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(path("filter_sweep.csv"), index=False)
    return len(df)


if __name__ == "__main__":
    print(f"ecg_like.csv        {make_ecg():>6} rows")
    print(f"accelerometer.csv   {make_accelerometer():>6} rows")
    print(f"bench_capture.txt   {make_messy_log():>6} samples")
    print(f"filter_sweep.csv    {make_filter_sweep():>6} rows")
