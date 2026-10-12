"""
Offline test of the REAL-TIME sibilant transplant (no microphone, no GPU).

Feeds an input recording and its RVC conversion through
`RealtimeSibilantTransplant` block by block, exactly like realtime_gui.py does,
and compares the result with the offline transplant on the same pair.

    python tools/artifact_detect/rt_simulate.py INPUT.wav RVC_OUTPUT.wav \
        [--block 0.13] [--lookahead 0.12] [--sr 48000] [--min-dur 0.08] \
        [--min-centroid 6000] [--out-dir TEMP/rt_sim]

Writes  <out-dir>/rt.wav (streaming result, delay removed), offline.wav,
and diff.wav (rt - offline), and prints timing and the difference level.
"""

import argparse
import os
import sys
import time

import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from infer.lib.sibilant_transplant import auto_segments, transplant  # noqa: E402
from infer.lib.sibilant_transplant_rt import RealtimeSibilantTransplant  # noqa: E402


def db(v):
    return 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--block", type=float, default=0.13)
    ap.add_argument("--lookahead", type=float, default=0.12)
    ap.add_argument("--min-dur", type=float, default=0.08)
    ap.add_argument("--min-centroid", type=float, default=6000)
    ap.add_argument("--crossover", type=float, default=1500)
    ap.add_argument("--fade-ms", type=float, default=15)
    ap.add_argument("--eq-max-db", type=float, default=6)
    ap.add_argument("--out-dir", default=os.path.join("TEMP", "rt_sim"))
    a = ap.parse_args()

    sr = a.sr
    x, _ = librosa.load(a.input, sr=sr, mono=True)
    y, _ = librosa.load(a.output, sr=sr, mono=True)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # offline reference (the offline module works at 48 kHz)
    t0 = time.perf_counter()
    if sr == 48000:
        segs = auto_segments(x, a.min_dur, a.min_centroid)
        off = transplant(x, y, segs, a.fade_ms, a.crossover, a.eq_max_db) if segs else y.copy()
    else:
        x48 = librosa.resample(x, orig_sr=sr, target_sr=48000)
        y48 = librosa.resample(y, orig_sr=sr, target_sr=48000)
        segs = auto_segments(x48, a.min_dur, a.min_centroid)
        o48 = transplant(x48, y48, segs, a.fade_ms, a.crossover, a.eq_max_db) if segs else y48
        off = librosa.resample(o48, orig_sr=48000, target_sr=sr)[:n]
    t_off = time.perf_counter() - t0
    print("offline: %d segments, %.2f s" % (len(segs), t_off))

    # streaming, block by block
    zc = sr // 100
    block = int(round(a.block * sr / zc)) * zc
    st = RealtimeSibilantTransplant(
        sr, block, lookahead_s=a.lookahead, min_dur=a.min_dur,
        min_centroid=a.min_centroid, fade_ms=a.fade_ms,
        crossover=a.crossover, eq_max_db=a.eq_max_db)
    pad = st.L + block
    xp = np.pad(x, (0, pad))
    yp = np.pad(y, (0, pad))
    outs, times, nseg = [], [], 0
    for i in range(0, len(xp) - block + 1, block):
        t0 = time.perf_counter()
        outs.append(st.process(xp[i:i + block], yp[i:i + block]))
        times.append(time.perf_counter() - t0)
        nseg += st.last_segments > 0
    rt = np.concatenate(outs)
    delay = st.W - st.ctx - block
    rt = rt[delay:delay + n]
    if len(rt) < n:
        rt = np.pad(rt, (0, n - len(rt)))

    times = np.array(times) * 1000
    print("streaming: block %.0f ms, look-ahead %.0f ms, added delay %.0f ms"
          % (block / sr * 1000, st.L / sr * 1000, delay / sr * 1000))
    print("  per-block time: mean %.1f ms, max %.1f ms (budget %.0f ms); blocks with hiss: %d/%d"
          % (times.mean(), times.max(), block / sr * 1000, nseg, len(times)))

    chg_off, chg_rt = off - y, rt - y
    print("  change made by offline:   %.1f dBFS" % db(chg_off))
    print("  change made by streaming: %.1f dBFS" % db(chg_rt))
    print("  streaming - offline:      %.1f dBFS  (%.1f dB below the offline change)"
          % (db(rt - off), db(chg_off) - db(rt - off)))

    os.makedirs(a.out_dir, exist_ok=True)
    sf.write(os.path.join(a.out_dir, "rt.wav"), rt, sr)
    sf.write(os.path.join(a.out_dir, "offline.wav"), off, sr)
    sf.write(os.path.join(a.out_dir, "diff.wav"), rt - off, sr)
    print("written to", a.out_dir)


if __name__ == "__main__":
    main()
