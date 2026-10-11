"""
Static (offline) sibilant-transplant post-filter for RVC output.

Long S-type fricatives in RVC output (male -> female, +12) carry a robotic
texture. This replaces them with the speaker's own input hiss:
  - segments are detected fresh on every call from the INPUT audio with
    `auto_segments` (the `long_s` rule in artifact_detect.py),
  - inside a segment (with linear fades), everything above the crossover
    (~1.5 kHz) comes from the input; below it the RVC output is kept,
  - the input hiss gets a mild colour EQ towards the RVC hiss spectrum and its
    loudness follows the converted voice frame by frame.

Usage:
    python tools/artifact_detect/transplant.py INPUT.wav RVC_OUT.wav [OUT.wav]
        [--min-dur 0.08] [--min-centroid 6000] [--fade-ms 15] [--crossover 1500]
        [--eq-max-db 6] [--bandwidth-thr-db -60] [--write-segments]

The processing lives in infer/lib/sibilant_transplant.py (shared with the WebUI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import soundfile as sf

# Repo root on sys.path so the shared module can be imported when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infer.lib.sibilant_transplant import SR, auto_segments, transplant  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replace long S sounds in RVC output with the input's own hiss.")
    p.add_argument("input", type=Path, help="original (source) recording")
    p.add_argument("rvc_out", type=Path, help="RVC-converted recording")
    p.add_argument("out", type=Path, nargs="?", default=None,
                   help="result (default: <RVC_OUT stem>_transplant.wav next to RVC_OUT)")
    p.add_argument("--min-dur", type=float, default=0.08, help="minimum S duration (s); 0 = all S-type hiss")
    p.add_argument("--min-centroid", type=float, default=6000.0, help="minimum median spectral centroid (Hz)")
    p.add_argument("--fade-ms", type=float, default=15.0, help="linear fade in/out around each segment")
    p.add_argument("--crossover", type=float, default=1500.0, help="below this the RVC output is kept (Hz)")
    p.add_argument("--eq-max-db", type=float, default=6.0, help="colour EQ limit in dB (0 disables the EQ)")
    p.add_argument("--bandwidth-thr-db", type=float, default=-60.0,
                   help="output bandwidth = highest frequency within this many dB of the spectrum peak")
    p.add_argument("--no-match-bandwidth", action="store_true",
                   help="paste full-band input hiss (old behaviour) instead of matching the output's bandwidth")
    p.add_argument("--write-segments", action="store_true",
                   help="also write <OUT stem>_segments.txt (Audacity labels) of the replaced segments")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_path, rvc_path = args.input.resolve(), args.rvc_out.resolve()
    for pth in (in_path, rvc_path):
        if not pth.is_file():
            print(f"error: file not found: {pth}", file=sys.stderr)
            return 2
    out_path = (args.out or rvc_path.with_name(f"{rvc_path.stem}_transplant.wav")).resolve()

    x, _ = librosa.load(str(in_path), sr=SR, mono=True)
    y, _ = librosa.load(str(rvc_path), sr=SR, mono=True)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # Detection runs fresh on the input every time.
    segs = auto_segments(x, args.min_dur, args.min_centroid)
    z, info = transplant(x, y, segs, args.fade_ms, args.crossover, args.eq_max_db,
                         bandwidth_thr_db=args.bandwidth_thr_db, return_info=True,
                         match_bandwidth=not args.no_match_bandwidth)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), z, SR, subtype="PCM_16")

    print(f"output bandwidth: {info['bandwidth_hz'] / 1000:.1f} kHz")
    print(f"segments: {len(segs)}  (min-dur {args.min_dur:g} s, min-centroid {args.min_centroid:g} Hz)")
    for i, (s, e) in enumerate(segs, 1):
        print(f"  #{i:<3d} {s:8.3f} – {e:8.3f} s  ({(e - s) * 1000:4.0f} ms)")
    print(f"wrote: {out_path}")

    if args.write_segments:
        seg_path = out_path.with_name(f"{out_path.stem}_segments.txt")
        with open(seg_path, "w", encoding="utf-8") as fh:
            for i, (s, e) in enumerate(segs, 1):
                fh.write(f"{s:.6f}\t{e:.6f}\t#{i} long S\n")
        print(f"wrote: {seg_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
