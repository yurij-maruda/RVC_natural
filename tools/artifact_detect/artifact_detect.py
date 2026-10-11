"""
Offline "robotic artifact" detector for RVC output (analysis only).

Idea: the robotic artifact is periodic (voiced) energy in the RVC output at
moments where the original input voice is noise-like (sibilants S / SH / CH
and their edges).

Methods:
  long_s   (default) long, sustained S-type fricatives in the INPUT (hiss
           centred above ~6 kHz lasting >= ~80 ms). Listening tests showed the
           robotic artifact lives inside these; short S and SH are fine.
           Each detected segment is one event.
  voicing  frame-level pYIN voicing probability mismatch:
               artifact(t) = clip(vp_out(t) - vp_in(t), 0, 1)
           gated to frames where the input is unvoiced/weakly voiced and the
           output is not silent. Events come from this curve only. A per-band
           HPSS harmonic-ratio difference (half-octave bands, 2-12 kHz) is used
           to localize the artifact in frequency (heatmap + dominant band).
  flatness / hpss  (legacy, for comparison) per-band tonality mismatch:
               A(t, band) = clip(T_out - T_in, 0, 1) * gate
           These are pitch-dependent: at +12 semitones the output's harmonics
           are sparser per band and look "more tonal" in vowels.

Usage:
    python tools/artifact_detect/artifact_detect.py INPUT.wav OUTPUT.wav [options]

The band / tonality / voicing functions operate on plain arrays (STFT
magnitudes of shape (n_bins, n_frames), or audio) so they can be reused later
in a real-time post-filter.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal

# Repo root on sys.path so the shared detector can be imported when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infer.lib.sibilant_transplant import auto_segments  # noqa: E402

SR = 48000  # common analysis sample rate
N_FFT = 2048  # ~43 ms
HOP = 240  # 5 ms
EPS = 1e-12

VOICING_SR = 16000  # pYIN runs at 16 kHz
VOICING_HOP = 80  # 5 ms at 16 kHz, same frame grid as the 48 kHz STFT
VOICING_FRAME = 1024

SILENCE_DB = -55.0  # frame energy relative to the file's loudest frame
NEAR_FRICATIVE_SEC = 0.060


# ----------------------------------------------------------------------------
# Loading and alignment
# ----------------------------------------------------------------------------


def load_mono(path: Path, sr: int = SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 resampled to `sr`. Returns (audio, original_sr)."""
    import librosa
    import soundfile as sf

    audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32), orig_sr


def rms_envelope(x: np.ndarray, sr: int, win_ms: float = 10.0, hop_ms: float = 1.0) -> np.ndarray:
    """RMS envelope with a `win_ms` window and `hop_ms` hop."""
    win = int(round(sr * win_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[np.minimum(idx, len(x) - 1)]
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def find_lag(x_in: np.ndarray, x_out: np.ndarray, sr: int, max_lag_ms: float) -> int:
    """
    Estimate how many samples `x_out` lags behind `x_in` by cross-correlating
    their RMS envelopes (1 ms resolution). Positive => output is late.
    """
    hop_ms = 1.0
    e_in = rms_envelope(x_in, sr, hop_ms=hop_ms)
    e_out = rms_envelope(x_out, sr, hop_ms=hop_ms)
    e_in = e_in - e_in.mean()
    e_out = e_out - e_out.mean()
    corr = scipy.signal.correlate(e_out, e_in, mode="full", method="fft")
    lags = scipy.signal.correlation_lags(len(e_out), len(e_in), mode="full")
    max_lag = int(round(max_lag_ms / hop_ms))
    valid = np.abs(lags) <= max_lag
    best = lags[valid][np.argmax(corr[valid])]
    return int(round(best * hop_ms * sr / 1000))


def align(x_in: np.ndarray, x_out: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift output onto the input timeline by `lag` samples and trim to common length."""
    if lag > 0:
        x_out = x_out[lag:]
    elif lag < 0:
        x_out = np.concatenate([np.zeros(-lag, dtype=x_out.dtype), x_out])
    n = min(len(x_in), len(x_out))
    return x_in[:n], x_out[:n]


def stft_mag(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Magnitude STFT, Hann window, centered frames. Shape (n_bins, n_frames)."""
    import librosa

    return np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann", center=True)).astype(np.float32)


# ----------------------------------------------------------------------------
# Frame-level helpers (voicing, energy, fricatives)
# ----------------------------------------------------------------------------


def voicing_prob(
    x: np.ndarray,
    sr: int,
    n_frames: int,
    fmin: float = 60.0,
    fmax: float = 1000.0,
) -> np.ndarray:
    """
    pYIN voicing probability per 5 ms frame. `x` is resampled to 16 kHz;
    centered frames (hop 80 @ 16 kHz) share the grid of the 48 kHz STFT
    (hop 240), so frame i is at i * 5 ms. Output is padded/trimmed to `n_frames`.
    """
    import librosa

    y = librosa.resample(x, orig_sr=sr, target_sr=VOICING_SR) if sr != VOICING_SR else x
    _, _, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=VOICING_SR, frame_length=VOICING_FRAME,
                            hop_length=VOICING_HOP, center=True)
    vp = np.nan_to_num(vp, nan=0.0)
    if len(vp) < n_frames:
        vp = np.pad(vp, (0, n_frames - len(vp)))
    return vp[:n_frames]


def frame_energy_db(mag: np.ndarray) -> np.ndarray:
    """Frame energy in dB relative to the loudest frame."""
    e = 10 * np.log10((mag.astype(np.float64) ** 2).sum(axis=0) + EPS)
    return e - e.max()


def fricative_frames(mag: np.ndarray, freqs: np.ndarray, floor_db: float = SILENCE_DB) -> np.ndarray:
    """Frames that look fricative: energy 3.5-12 kHz > energy 80-1200 Hz, and not silent."""
    p = mag.astype(np.float64) ** 2
    hf = p[(freqs >= 3500) & (freqs <= 12000)].sum(axis=0)
    lf = p[(freqs >= 80) & (freqs <= 1200)].sum(axis=0)
    return (hf > lf) & (frame_energy_db(mag) > floor_db)


def smooth_time(x: np.ndarray, frames: int = 3) -> np.ndarray:
    """Moving average along the time axis (last axis)."""
    if frames <= 1:
        return x
    return scipy.ndimage.uniform_filter1d(x, size=frames, axis=-1, mode="nearest")


def voicing_artifact(
    vp_in: np.ndarray,
    vp_out: np.ndarray,
    out_active: np.ndarray,
    input_voicing_max: float = 0.35,
    smooth_frames: int = 3,
) -> np.ndarray:
    """
    artifact(t) = clip(vp_out - vp_in, 0, 1), only where the input is unvoiced /
    weakly voiced (vp_in < input_voicing_max) and the output is not silent;
    then smoothed over `smooth_frames`.
    """
    gate = (vp_in < input_voicing_max) & out_active
    return smooth_time(np.clip(vp_out - vp_in, 0.0, 1.0) * gate, smooth_frames)


# ----------------------------------------------------------------------------
# Long S-type fricatives (default method; detector lives in infer/lib/sibilant_transplant.py)
# ----------------------------------------------------------------------------


def centroid_track(x: np.ndarray, hop: int = 240) -> np.ndarray:
    """Per-frame spectral centroid (Hz), computed the same way as in auto_segments."""
    P = np.abs(librosa.stft(x, n_fft=2048, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    return (P * f[:, None]).sum(0) / (P.sum(0) + 1e-12)


def segment_centroid(cen: np.ndarray, start: float, end: float, hop: int = 240) -> float:
    """Median centroid (Hz) over a segment (start_sec, end_sec) returned by auto_segments."""
    a = int(round(start * SR / hop))
    b = int(round(end * SR / hop)) - 1
    return float(np.median(cen[a:b + 1]))


# ----------------------------------------------------------------------------
# Bands and tonality (reusable: work on STFT magnitude arrays)
# ----------------------------------------------------------------------------


@dataclass
class Band:
    lo_bin: int  # inclusive
    hi_bin: int  # exclusive
    lo_hz: float
    hi_hz: float

    @property
    def sl(self) -> slice:
        return slice(self.lo_bin, self.hi_bin)


def make_bands(
    sr: int = SR,
    n_fft: int = N_FFT,
    fmin: float = 1000.0,
    fmax: float = 12000.0,
    n_bands: int = 18,
    min_bins: int = 8,
) -> list[Band]:
    """
    Log-spaced bands between fmin and fmax (18 bands over 1-12 kHz ~ 1/5 octave).
    Bands with fewer than `min_bins` FFT bins are merged with their upper
    neighbour; a too-narrow last band is merged into the previous one.
    """
    bin_hz = sr / n_fft
    edges_hz = np.geomspace(fmin, fmax, n_bands + 1)
    edges_bin = np.unique(np.round(edges_hz / bin_hz).astype(int))

    bands: list[Band] = []
    lo = edges_bin[0]
    for hi in edges_bin[1:]:
        if hi - lo >= min_bins:
            bands.append(Band(lo, hi, lo * bin_hz, hi * bin_hz))
            lo = hi
    if lo < edges_bin[-1]:  # leftover narrow top band
        if bands:
            last = bands[-1]
            bands[-1] = Band(last.lo_bin, edges_bin[-1], last.lo_hz, edges_bin[-1] * bin_hz)
        else:
            bands.append(Band(lo, edges_bin[-1], lo * bin_hz, edges_bin[-1] * bin_hz))
    return bands


def band_energy(mag: np.ndarray, bands: list[Band], mean: bool = False) -> np.ndarray:
    """Band power per frame (sum, or mean per bin if `mean`). Shape (n_bands, n_frames)."""
    power = mag.astype(np.float64) ** 2
    out = np.stack([power[b.sl].sum(axis=0) for b in bands])
    if mean:
        out /= np.array([b.hi_bin - b.lo_bin for b in bands], dtype=np.float64)[:, None]
    return out


def tonality_flatness(mag: np.ndarray, bands: list[Band], range_db: float = 30.0) -> np.ndarray:
    """
    Tonality from spectral flatness of power within each band:
        flatness = geometric_mean(P) / arithmetic_mean(P)
        T = clip(-10*log10(flatness) / range_db, 0, 1)
    White noise gives flatness ~ -2.5 dB (T ~ 0.08); clear harmonics give much lower.
    Returns (n_bands, n_frames) in [0, 1].
    """
    power = mag.astype(np.float64) ** 2 + EPS
    log_p = np.log(power)
    out = np.empty((len(bands), mag.shape[1]))
    for i, b in enumerate(bands):
        geo = np.exp(log_p[b.sl].mean(axis=0))
        arith = power[b.sl].mean(axis=0)
        flat_db = 10.0 * np.log10(geo / arith + EPS)
        out[i] = np.clip(-flat_db / range_db, 0.0, 1.0)
    return out


def tonality_hpss(mag: np.ndarray, bands: list[Band], kernel_size: int = 31, pad_bins: int = 32) -> np.ndarray:
    """
    Tonality from harmonic/percussive separation (librosa soft masks, margin 1):
        T = E_harmonic / (E_harmonic + E_percussive) within each band.
    "Harmonic" here means horizontally continuous (steady partials) energy.
    HPSS runs only on the bins covered by `bands` (+ padding) for speed.
    Returns (n_bands, n_frames) in [0, 1].
    """
    import librosa

    lo = max(0, min(b.lo_bin for b in bands) - pad_bins)
    hi = min(mag.shape[0], max(b.hi_bin for b in bands) + pad_bins)
    sub = mag[lo:hi]
    mask_h, mask_p = librosa.decompose.hpss(sub, kernel_size=kernel_size, power=2.0, mask=True, margin=1.0)
    p_h = (mask_h * sub).astype(np.float64) ** 2
    p_p = (mask_p * sub).astype(np.float64) ** 2
    out = np.empty((len(bands), mag.shape[1]))
    for i, b in enumerate(bands):
        sl = slice(b.lo_bin - lo, b.hi_bin - lo)
        eh = p_h[sl].sum(axis=0)
        ep = p_p[sl].sum(axis=0)
        out[i] = eh / (eh + ep + EPS)
    return out


def energy_gate(
    mag_in: np.ndarray,
    mag_out: np.ndarray,
    bands: list[Band],
    band_floor_db: float = -50.0,
    input_floor_db: float = SILENCE_DB,
) -> np.ndarray:
    """
    1 where the cell is worth scoring, 0 otherwise:
      - output band energy (per-bin mean) within |band_floor_db| of the file's max band energy, and
      - input frame energy above `input_floor_db` relative to the loudest input frame.
    Shape (n_bands, n_frames).
    """
    e_out = band_energy(mag_out, bands, mean=True)
    e_out_db = 10 * np.log10(e_out + EPS)
    band_ok = e_out_db > e_out_db.max() + band_floor_db
    frame_ok = frame_energy_db(mag_in) > input_floor_db
    return (band_ok & frame_ok[None, :]).astype(np.float64)


def artifact_map(t_in: np.ndarray, t_out: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """A = clip(T_out - T_in, 0, 1) * gate. Shape (n_bands, n_frames)."""
    return np.clip(t_out - t_in, 0.0, 1.0) * gate


def band_localization(mag_in: np.ndarray, mag_out: np.ndarray, bands: list[Band], active: np.ndarray) -> np.ndarray:
    """
    Where in frequency is the artifact: per-band HPSS harmonic ratio,
    output minus input, clipped to [0, 1], only on `active` frames.
    Used for display / dominant band only, never to create events.
    """
    t_in = smooth_time(tonality_hpss(mag_in, bands))
    t_out = smooth_time(tonality_hpss(mag_out, bands))
    return np.clip(t_out - t_in, 0.0, 1.0) * active[None, :]


def frame_scores(a_map: np.ndarray, e_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame score: output-band-energy-weighted mean of A across bands,
    plus the max across bands. `e_out` is band energy (n_bands, n_frames).
    """
    w = e_out / (e_out.sum(axis=0, keepdims=True) + EPS)
    return (w * a_map).sum(axis=0), a_map.max(axis=0)


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------


@dataclass
class Event:
    start: float
    end: float
    peak: float
    mean: float
    band_lo_hz: float  # nan if no band shows a harmonic surplus
    band_hi_hz: float
    fricative_input: bool
    vp_in: float
    vp_out: float
    near_fricative: bool
    centroid_hz: float = math.nan  # long_s only: median input spectral centroid

    @classmethod
    def long_s(cls, start: float, end: float, centroid_hz: float) -> "Event":
        nan = math.nan
        return cls(start, end, nan, nan, nan, nan, True, nan, nan, True, centroid_hz)

    @property
    def is_long_s(self) -> bool:
        return not math.isnan(self.centroid_hz)

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_events(
    score: np.ndarray,
    threshold: float,
    frame_sec: float,
    merge_gap_sec: float = 0.020,
    min_dur_sec: float = 0.015,
) -> list[tuple[int, int]]:
    """Frame runs above threshold, gaps < merge_gap merged, runs < min_dur dropped. Returns [(i0, i1_exclusive)]."""
    above = score > threshold
    if not above.any():
        return []
    d = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

    merged: list[list[int]] = [list(runs[0])]
    max_gap = int(round(merge_gap_sec / frame_sec))
    for s, e in runs[1:]:
        if s - merged[-1][1] < max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    min_frames = int(np.ceil(min_dur_sec / frame_sec - 1e-9))
    return [(s, e) for s, e in merged if e - s >= min_frames]


def describe_event(
    i0: int,
    i1: int,
    score: np.ndarray,
    a_map: np.ndarray,
    bands: list[Band],
    mag_in: np.ndarray,
    freqs: np.ndarray,
    vp_in: np.ndarray,
    vp_out: np.ndarray,
    fric: np.ndarray,
    frame_sec: float,
    duration_sec: float,
) -> Event:
    """Collect stats for frames [i0, i1)."""
    seg = score[i0:i1]
    # Dominant band: highest mean A, extended to contiguous neighbours with >= 50% of that.
    band_mean = a_map[:, i0:i1].mean(axis=1)
    top = int(np.argmax(band_mean))
    if band_mean[top] > 0:
        lo = hi = top
        while lo > 0 and band_mean[lo - 1] >= 0.5 * band_mean[top]:
            lo -= 1
        while hi < len(bands) - 1 and band_mean[hi + 1] >= 0.5 * band_mean[top]:
            hi += 1
        band_lo, band_hi = bands[lo].lo_hz, bands[hi].hi_hz
    else:
        band_lo = band_hi = math.nan

    # "Input looks fricative" over the event: input energy 3.5-12 kHz > energy 80-1200 Hz.
    p_in = mag_in[:, i0:i1].astype(np.float64) ** 2
    hf = p_in[(freqs >= 3500) & (freqs <= 12000)].sum()
    lf = p_in[(freqs >= 80) & (freqs <= 1200)].sum()

    # Near fricative: any fricative input frame within +-60 ms of the event.
    pad = int(round(NEAR_FRICATIVE_SEC / frame_sec))
    near = bool(fric[max(0, i0 - pad):i1 + pad].any())

    # Frames are centered at i*frame_sec; event covers half a hop either side.
    start = max(0.0, (i0 - 0.5) * frame_sec)
    end = min(duration_sec, (i1 - 0.5) * frame_sec)
    return Event(start, end, float(seg.max()), float(seg.mean()), band_lo, band_hi, bool(hf > lf),
                 float(vp_in[i0:i1].mean()), float(vp_out[i0:i1].mean()), near)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def fmt_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:06.3f}"


def fmt_band(ev: Event, sep: str = "–") -> str:
    if math.isnan(ev.band_lo_hz):
        return "n/a"
    return f"{ev.band_lo_hz / 1000:.1f}{sep}{ev.band_hi_hz / 1000:.1f} kHz"


def event_line(n: int, ev: Event) -> str:
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    if ev.is_long_s:
        return (f"#{n:<3d} {fmt_time(ev.start)} – {fmt_time(ev.end)}  ({ev.duration * 1000:4.0f} ms)  "
                f"centroid {ev.centroid_hz / 1000:.1f} kHz")
    return (
        f"#{n:<3d} {fmt_time(ev.start)} – {fmt_time(ev.end)}  ({ev.duration * 1000:4.0f} ms)  "
        f"peak {ev.peak:.2f}  mean {ev.mean:.2f}  band {fmt_band(ev):<12s}  "
        f"vp_in {ev.vp_in:.2f}  vp_out {ev.vp_out:.2f}  "
        f"fricative-input: {yn(ev.fricative_input):<3s}  near-fricative: {yn(ev.near_fricative)}"
    )


def write_reports(out_dir: Path, suffix: str, header: list[str], events: list[Event],
                  long_s: bool = False) -> dict[str, Path]:
    paths = {
        "txt": out_dir / f"artifacts{suffix}.txt",
        "csv": out_dir / f"artifacts{suffix}.csv",
        "labels": out_dir / f"artifacts{suffix}_audacity_labels.txt",
    }
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n\n")
        for n, ev in enumerate(events, 1):
            f.write(event_line(n, ev) + "\n")

    with open(paths["csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if long_s:
            w.writerow(["n", "start_s", "end_s", "duration_ms", "centroid_hz"])
            for n, ev in enumerate(events, 1):
                w.writerow([n, f"{ev.start:.4f}", f"{ev.end:.4f}", f"{ev.duration * 1000:.1f}", f"{ev.centroid_hz:.0f}"])
        else:
            w.writerow(["n", "start_s", "end_s", "duration_ms", "peak", "mean", "band_lo_hz", "band_hi_hz",
                        "vp_in", "vp_out", "fricative_input", "near_fricative"])
            for n, ev in enumerate(events, 1):
                w.writerow([n, f"{ev.start:.4f}", f"{ev.end:.4f}", f"{ev.duration * 1000:.1f}", f"{ev.peak:.4f}",
                            f"{ev.mean:.4f}",
                            "" if math.isnan(ev.band_lo_hz) else f"{ev.band_lo_hz:.0f}",
                            "" if math.isnan(ev.band_hi_hz) else f"{ev.band_hi_hz:.0f}",
                            f"{ev.vp_in:.3f}", f"{ev.vp_out:.3f}", int(ev.fricative_input), int(ev.near_fricative)])

    with open(paths["labels"], "w", encoding="utf-8") as f:
        for n, ev in enumerate(events, 1):
            if long_s:
                label = f"#{n} long S {ev.centroid_hz / 1000:.1f}k"
            else:
                label = f"#{n} p{ev.peak:.2f} {fmt_band(ev, '-')}{' fric' if ev.near_fricative else ''}"
            f.write(f"{ev.start:.6f}\t{ev.end:.6f}\t{label}\n")
    return paths


def plot(
    path: Path,
    mag_in: np.ndarray,
    mag_out: np.ndarray,
    freqs: np.ndarray,
    a_map: np.ndarray | None,
    bands: list[Band],
    map_title: str,
    score: np.ndarray | None,
    score_label: str,
    vp_in: np.ndarray | None,
    vp_out: np.ndarray | None,
    threshold: float,
    events: list[Event],
    frame_sec: float,
    titles: tuple[str, str],
    zoom: tuple[float | None, float | None],
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = mag_in.shape[1]
    t_edges = (np.arange(n_frames + 1) - 0.5) * frame_sec
    t_centers = np.arange(n_frames) * frame_sec
    spec_mask = freqs <= 12000
    f_top = freqs[spec_mask][-1]

    # Without a heatmap (long_s) only the two spectrogram rows are drawn.
    full = a_map is not None
    fig, axes = plt.subplots(4 if full else 2, 1, figsize=(20, 12 if full else 8), dpi=100, sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 2.2, 1.8] if full else [1, 1]})

    for ax, mag, title in ((axes[0], mag_in, titles[0]), (axes[1], mag_out, titles[1])):
        db = 20 * np.log10(mag[spec_mask] + 1e-9)
        vmax = db.max()
        ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 80, vmax=vmax,
                  extent=(t_edges[0], t_edges[-1], 0, f_top), interpolation="nearest")
        ax.set_ylabel("Hz")
        ax.set_title(title, loc="left", fontsize=10)
        for ev in events:
            ax.axvspan(ev.start, ev.end, color="cyan", alpha=0.15, lw=0)
            ax.axvline(ev.start, color="cyan", alpha=0.5, lw=0.6)

    lo, hi = zoom
    if not full:
        axes[1].set_xlabel("time (s, input timeline)")
        axes[1].set_xlim(lo if lo is not None else t_edges[0], hi if hi is not None else t_edges[-1])
        fig.tight_layout()
        fig.savefig(path)
        if show:
            plt.show()
        plt.close(fig)
        return

    band_edges = np.array([b.lo_hz for b in bands] + [bands[-1].hi_hz])
    mesh = axes[2].pcolormesh(t_edges, band_edges, a_map, cmap="magma", vmin=0, vmax=1, shading="flat")
    axes[2].set_yscale("log")
    ticks = [t for t in (1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000)
             if band_edges[0] <= t <= band_edges[-1]]
    axes[2].set_yticks(ticks)
    axes[2].set_yticklabels([f"{t / 1000:g}k" for t in ticks])
    axes[2].minorticks_off()
    axes[2].set_ylabel("Hz")
    axes[2].set_title(map_title, loc="left", fontsize=10)
    fig.colorbar(mesh, ax=axes[2], pad=0.005, fraction=0.02)

    ax = axes[3]
    ax.plot(t_centers, vp_in, color="C0", lw=0.8, alpha=0.8, label="voicing in (pYIN)")
    ax.plot(t_centers, vp_out, color="C1", lw=0.8, alpha=0.8, label="voicing out (pYIN)")
    ax.plot(t_centers, score, color="C3", lw=1.2, label=score_label)
    ax.axhline(threshold, color="k", ls="--", lw=0.8, label=f"threshold {threshold:g}")
    for ev in events:
        ax.axvspan(ev.start, ev.end, color="C3", alpha=0.2, lw=0)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("prob / score")
    ax.set_xlabel("time (s, input timeline)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    # Keep the non-colorbar rows the same width as the heatmap row.
    for a in (axes[0], axes[1], axes[3]):
        fig.colorbar(mesh, ax=a, pad=0.005, fraction=0.02).ax.set_visible(False)

    axes[3].set_xlim(lo if lo is not None else t_edges[0], hi if hi is not None else t_edges[-1])
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect voiced-where-input-is-noisy (robotic) artifacts in RVC output.")
    p.add_argument("input", type=Path, help="original (source) recording")
    p.add_argument("output", type=Path, help="RVC-converted recording")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write reports (default: <output_stem>_artifacts/ next to OUTPUT)")
    p.add_argument("--method", choices=("long_s", "voicing", "flatness", "hpss"), default="long_s",
                   help="long_s = long S-type fricatives in the input (default); voicing = pYIN voicing mismatch; "
                        "flatness/hpss = legacy per-band tonality")
    p.add_argument("--min-dur", type=float, default=0.08, help="long_s: minimum segment duration (s)")
    p.add_argument("--min-centroid", type=float, default=6000.0, help="long_s: minimum median spectral centroid (Hz)")
    p.add_argument("--threshold", type=float, default=None,
                   help="event threshold for voicing/flatness/hpss (default 0.15 for voicing, 0.25 for flatness/hpss)")
    p.add_argument("--input-voicing-max", type=float, default=0.35,
                   help="voicing: only score frames where input voicing prob is below this")
    p.add_argument("--fmin", type=float, default=None, help="band range low edge (default 2000 voicing / 1000 legacy)")
    p.add_argument("--fmax", type=float, default=12000.0)
    p.add_argument("--bands", type=int, default=None,
                   help="number of log-spaced bands (default half-octave for voicing, 18 for legacy)")
    p.add_argument("--start", type=float, default=None, help="plot zoom start (s)")
    p.add_argument("--end", type=float, default=None, help="plot zoom end (s)")
    p.add_argument("--show", action="store_true", help="also open the matplotlib window")
    p.add_argument("--max-lag-ms", type=float, default=100.0, help="alignment search range")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    long_s = args.method == "long_s"
    voicing = args.method == "voicing"
    threshold = args.threshold if args.threshold is not None else (0.15 if voicing else 0.25)
    fmin = args.fmin if args.fmin is not None else (2000.0 if voicing else 1000.0)
    n_bands = args.bands or (max(1, round(2 * math.log2(args.fmax / fmin))) if voicing else 18)

    in_path, out_path = args.input.resolve(), args.output.resolve()
    for pth in (in_path, out_path):
        if not pth.is_file():
            print(f"error: file not found: {pth}", file=sys.stderr)
            return 2
    out_dir = (args.out_dir or out_path.parent / f"{out_path.stem}_artifacts").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if long_s else f"_{args.method}"

    # Load, resample, align
    x_in, sr_in = load_mono(in_path)
    x_out, sr_out = load_mono(out_path)
    lag = find_lag(x_in, x_out, SR, args.max_lag_ms)
    x_in, x_out = align(x_in, x_out, lag)
    duration = len(x_in) / SR

    # STFT (shared 5 ms grid)
    mag_in, mag_out = stft_mag(x_in), stft_mag(x_out)
    n_frames = mag_in.shape[1]
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    frame_sec = HOP / SR
    titles = (f"INPUT  {in_path.name}", f"OUTPUT  {out_path.name}  (aligned, lag {lag / SR * 1000:+.1f} ms)")

    if long_s:
        # Each long S-type fricative in the input is one event.
        cen = centroid_track(x_in)
        events = [Event.long_s(s, e, segment_centroid(cen, s, e))
                  for s, e in auto_segments(x_in, args.min_dur, args.min_centroid)]
        flagged = sum(ev.duration for ev in events)
        flagged_pct = 100 * flagged / duration if duration else 0.0
        header = [
            f"input:   {in_path}  ({sr_in} Hz)",
            f"output:  {out_path}  ({sr_out} Hz)",
            f"analysis: {SR} Hz, n_fft {N_FFT}, hop {HOP} ({frame_sec * 1000:.0f} ms)",
            f"alignment lag: {lag / SR * 1000:+.1f} ms (positive = output late; times below are on the input timeline)",
            f"method: long_s   min-dur: {args.min_dur:g} s   min-centroid: {args.min_centroid:g} Hz",
            f"events: {len(events)}   flagged: {flagged:.2f} s of {duration:.2f} s ({flagged_pct:.1f}%)",
        ]
        paths = write_reports(out_dir, suffix, header, events, long_s=True)
        png = out_dir / f"artifacts{suffix}.png"
        plot(png, mag_in, mag_out, freqs, None, [], "", None, "", None, None, 0.0, events, frame_sec, titles,
             (args.start, args.end), args.show)
        print("\n".join(header))
        for n, ev in enumerate(events, 1):
            print("  " + event_line(n, ev))
        print(f"wrote: {paths['txt'].name}, {paths['csv'].name}, {paths['labels'].name}, {png.name}  ->  {out_dir}")
        return 0

    # Frame-level features for the score-based methods
    vp_in = voicing_prob(x_in, SR, n_frames)
    vp_out = voicing_prob(x_out, SR, n_frames)
    fric = fricative_frames(mag_in, freqs)
    bands = make_bands(SR, N_FFT, fmin, args.fmax, n_bands)

    if voicing:
        out_active = frame_energy_db(mag_out) > SILENCE_DB
        score = voicing_artifact(vp_in, vp_out, out_active, args.input_voicing_max)
        a_map = band_localization(mag_in, mag_out, bands, active=score > 0)
        map_title = "Band localization: HPSS harmonic ratio, out − in (only where voicing artifact > 0)"
        score_label = "artifact = clip(vp_out − vp_in) · gate"
        min_dur = 0.010
        method_desc = f"voicing (pYIN 16 kHz, input-voicing-max {args.input_voicing_max:g})"
    else:
        tonality = tonality_flatness if args.method == "flatness" else tonality_hpss
        t_in = smooth_time(tonality(mag_in, bands))
        t_out = smooth_time(tonality(mag_out, bands))
        a_map = artifact_map(t_in, t_out, energy_gate(mag_in, mag_out, bands))
        score, _ = frame_scores(a_map, band_energy(mag_out, bands))
        map_title = "Artifact map A = clip(T_out − T_in, 0, 1) · gate"
        score_label = "energy-weighted mean A (event score)"
        min_dur = 0.015
        method_desc = args.method

    runs = find_events(score, threshold, frame_sec, min_dur_sec=min_dur)
    events = [describe_event(i0, i1, score, a_map, bands, mag_in, freqs, vp_in, vp_out, fric, frame_sec, duration)
              for i0, i1 in runs]
    flagged = sum(ev.duration for ev in events)
    flagged_pct = 100 * flagged / duration if duration else 0.0
    n_near = sum(ev.near_fricative for ev in events)

    header = [
        f"input:   {in_path}  ({sr_in} Hz)",
        f"output:  {out_path}  ({sr_out} Hz)",
        f"analysis: {SR} Hz, n_fft {N_FFT}, hop {HOP} ({frame_sec * 1000:.0f} ms), "
        f"{len(bands)} bands {bands[0].lo_hz:.0f}–{bands[-1].hi_hz:.0f} Hz",
        f"alignment lag: {lag / SR * 1000:+.1f} ms (positive = output late; times below are on the input timeline)",
        f"method: {method_desc}   threshold: {threshold:g}",
        f"events: {len(events)}   flagged: {flagged:.2f} s of {duration:.2f} s ({flagged_pct:.1f}%)",
        f"near input fricative (±{NEAR_FRICATIVE_SEC * 1000:.0f} ms): {n_near}   elsewhere: {len(events) - n_near}",
    ]
    paths = write_reports(out_dir, suffix, header, events)

    png = out_dir / f"artifacts{suffix}.png"
    plot(png, mag_in, mag_out, freqs, a_map, bands, map_title, score, score_label, vp_in, vp_out, threshold,
         events, frame_sec, titles, (args.start, args.end), args.show)

    # Console summary
    print("\n".join(header))
    if events:
        print("top events by peak:")
        ranked = sorted(enumerate(events, 1), key=lambda ne: ne[1].peak, reverse=True)[:5]
        for n, ev in ranked:
            print("  " + event_line(n, ev))
    print(f"wrote: {paths['txt'].name}, {paths['csv'].name}, {paths['labels'].name}, {png.name}  ->  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
