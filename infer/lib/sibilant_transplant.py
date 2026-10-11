"""
Sibilant-transplant post-filter for RVC output (offline).

RVC output (e.g. male -> female, +12) can carry a robotic texture inside long
S sounds. This replaces those segments with the speaker's own input hiss:
  - `auto_segments` detects long S-type fricatives in the INPUT (hiss centred
    above ~6 kHz lasting >= ~80 ms); detection runs fresh on every call,
  - `transplant` swaps everything above the crossover (~1.5 kHz) inside those
    segments (linear fades) for the input hiss, with a mild colour EQ towards
    the RVC hiss spectrum and loudness following the converted voice frame by
    frame.

Both functions were validated by listening; keep the processing unchanged.
The transplant adapts to the output model's bandwidth (`output_bandwidth`):
the input hiss is low-passed to what the converted voice actually contains, so
band-limited models (e.g. ~10 kHz) don't get over-bright S/SH patches.
`apply_sibilant_transplant` is the entry point used by inference.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

SR = 48000  # analysis / processing sample rate
N_FFT, HOP = 1024, 128  # transplant STFT (2.7 ms hop)


def auto_segments(x, min_dur=0.08, min_centroid=6000):
    """Long S-type fricatives detected from the INPUT signal (mono, 48 kHz float).
    Returns a list of (start_sec, end_sec)."""
    hop = 240
    P = np.abs(librosa.stft(x, n_fft=2048, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    hi = P[(f >= 3500) & (f < 14000)].sum(0) + 1e-12
    lo = P[(f >= 80) & (f < 1200)].sum(0) + 1e-12
    en = P.sum(0) + 1e-12
    fr = (10 * np.log10(hi / lo) > 3) & (10 * np.log10(en / en.max()) > -50)
    cen = (P * f[:, None]).sum(0) / en
    idx = np.where(fr)[0]
    if len(idx) == 0:
        return []
    segs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i > p + 4:          # merge gaps up to 4 frames (20 ms)
            segs.append((s, p)); s = i
        p = i
    segs.append((s, p))
    out = []
    for a, b in segs:
        dur = (b - a) * hop / SR
        c = np.median(cen[a:b + 1])
        if dur >= min_dur and c >= min_centroid:
            out.append((a * hop / SR, (b + 1) * hop / SR))
    return out


def output_bandwidth(Y, f, thr_db=-60.0):
    """Highest frequency where the output's long-term spectrum is within thr_db of its peak."""
    P = (np.abs(Y) ** 2).mean(1)
    Pdb = 10 * np.log10(P / P.max() + 1e-15)
    idx = np.where(Pdb > thr_db)[0]
    return float(f[idx.max()]) if len(idx) else float(f[-1])


def transplant(x, y, segs, fade_ms=15, crossover=1500, eq_max_db=6,
               bandwidth_thr_db=-60.0, return_info=False, match_bandwidth=True):
    """x = input (48 kHz mono), y = RVC output (48 kHz mono, same length),
    segs = [(start_s, end_s), ...] from auto_segments(x). Returns processed output
    (or (output, {"bandwidth_hz": ...}) with return_info=True)."""
    # free-form settings: only guard against values that would break the maths
    fade_ms = max(float(fade_ms), 0.1)
    crossover = max(float(crossover), 1.0)
    eq_max_db = max(float(eq_max_db), 0.0)
    X = librosa.stft(x, n_fft=N_FFT, hop_length=HOP)
    Y = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    f = librosa.fft_frequencies(sr=SR, n_fft=N_FFT); T = X.shape[1]
    t = np.arange(T) * HOP / SR

    # bandwidth of the converted voice (model-dependent, measured fresh every call)
    cutoff = output_bandwidth(Y, f, bandwidth_thr_db)
    if not match_bandwidth:            # previous behaviour: full-band input hiss
        cutoff = float(f[-1]) + 1.0
    below = f < cutoff

    # time mask with linear fades
    fade = fade_ms / 1000
    m = np.zeros(T)
    for s, e in segs:
        r = np.clip(np.minimum((t - (s - fade)) / fade, ((e + fade) - t) / fade), 0, 1)
        m = np.maximum(m, r)

    # mild colour EQ: input hiss spectrum -> RVC output hiss spectrum over the segments
    core = m > 0.99
    if core.sum() > 5 and eq_max_db > 0:
        px = (np.abs(X[:, core]) ** 2).mean(1) + 1e-12
        py = (np.abs(Y[:, core]) ** 2).mean(1) + 1e-12
        g = 10 * np.log10(py / px)
        # bins above the output bandwidth must not influence the EQ: hold the last in-band value
        if below.any() and not below.all():
            g[~below] = g[below][-1]
        g = gaussian_filter1d(g, 8)
        norm = (f > 3000) & (f < min(12000.0, cutoff))
        g -= np.average(g[norm] if norm.any() else g[(f > 3000) & (f < 12000)])
        g = np.clip(g, -eq_max_db, eq_max_db)
        G = 10 ** (g / 20)
    else:
        G = np.ones(len(f))
    Xs = X * G[:, None]

    # low-pass the input hiss to the output's bandwidth (soft ~1 kHz roll-off below the cutoff)
    if match_bandwidth:
        roll = max(500.0, 0.1 * cutoff)
        Xs = Xs * np.clip((cutoff - f) / roll, 0, 1)[:, None]

    # loudness from the converted voice, frame by frame, in the band above the crossover
    # (only up to the output bandwidth)
    band = (f >= crossover) & below
    if band.sum() < 4:
        band = f >= crossover
    lx = np.sqrt((np.abs(Xs[band]) ** 2).sum(0) + 1e-12)
    ly = np.sqrt((np.abs(Y[band]) ** 2).sum(0) + 1e-12)
    gain = uniform_filter1d(np.clip(ly / lx, 0.1, 10), 9)
    Xs *= gain[None, :]

    # soft frequency crossover (~1-2 kHz): below it, keep the RVC output
    xo = np.clip((f - crossover * 0.67) / (crossover * 0.66), 0, 1)[:, None]
    M = m[None, :] * xo
    Z = Y * (1 - M) + Xs * M
    z = librosa.istft(Z, hop_length=HOP, length=len(y))
    pk = np.abs(z).max()
    if pk > 0.99:
        z *= 0.99 / pk
    if return_info:
        return z, {"bandwidth_hz": output_bandwidth(Y, f, bandwidth_thr_db),
                   "matched": bool(match_bandwidth)}
    return z


def apply_sibilant_transplant(
    input_path,
    output_audio,
    output_sr,
    min_dur=0.08,
    min_centroid=6000,
    fade_ms=15,
    crossover=1500,
    eq_max_db=6,
    bandwidth_thr_db=-60.0,
    match_bandwidth=True,
):
    """
    Run the transplant on a finished conversion.

    input_path:   ORIGINAL input file (loaded at full quality, mono, 48 kHz;
                  the pipeline's 16 kHz audio has nothing above 8 kHz).
    output_audio: 1-D converted audio (int16 as returned by the pipeline, or float).
    output_sr:    sample rate of `output_audio`.

    Returns (audio, n_segments, segments, info): audio has the same length, dtype and
    sample rate as `output_audio`; segments are (start_s, end_s) on the input timeline;
    info = {"bandwidth_hz": detected output bandwidth, "matched": match_bandwidth}.
    """
    out = np.asarray(output_audio)
    if out.ndim != 1:
        raise ValueError("expected mono 1-D output audio, got shape %s" % (out.shape,))
    is_int = np.issubdtype(out.dtype, np.integer)
    scale = float(np.iinfo(out.dtype).max + 1) if is_int else 1.0
    y = out.astype(np.float32) / scale

    x, _ = librosa.load(str(input_path), sr=SR, mono=True)
    y48 = librosa.resample(y, orig_sr=output_sr, target_sr=SR) if output_sr != SR else y
    n = min(len(x), len(y48))
    x, y48 = x[:n], y48[:n]

    segs = auto_segments(x, min_dur, min_centroid)
    if not segs:
        return output_audio, 0, [], {"bandwidth_hz": None, "matched": bool(match_bandwidth)}
    z48, info = transplant(x, y48, segs, fade_ms, crossover, eq_max_db,
                           bandwidth_thr_db=bandwidth_thr_db, return_info=True,
                           match_bandwidth=match_bandwidth)
    print("Sibilant transplant: output bandwidth %.1f kHz, %d segments"
          % (info["bandwidth_hz"] / 1000, len(segs)), flush=True)

    # Resample only the change (resampling is linear), so audio outside the
    # segments is not put through a 48 kHz round trip.
    delta = z48 - y48
    if output_sr != SR:
        delta = librosa.resample(delta, orig_sr=SR, target_sr=output_sr)
    if len(delta) < len(y):
        delta = np.pad(delta, (0, len(y) - len(delta)))
    res = np.clip(y + delta[: len(y)], -1.0, 1.0)

    if is_int:
        lim = np.iinfo(out.dtype)
        res = np.clip(res * scale, lim.min, lim.max).astype(out.dtype)
    else:
        res = res.astype(out.dtype)
    return res, len(segs), segs, info
