"""Plot cumulative audio duration vs. sample index (files ordered by name).

Usage:
    python tools/plot_cumulative_duration.py C:/dev/stream/VC/train_data/kateryna
    python tools/plot_cumulative_duration.py <dir> --out plot.png --show
"""

import argparse
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")


def natural_key(name):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def get_duration(path):
    try:
        return sf.info(path).duration
    except sf.LibsndfileError:
        # libsndfile rejects some Opus streams that ffprobe reads fine.
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--out", default=None, help="PNG path (default: <folder>_cumulative.png)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    folder = os.path.normpath(args.folder)
    files = sorted(
        (f for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXTS)), key=natural_key
    )
    if not files:
        raise SystemExit(f"No audio files in {folder}")

    durations = np.array([get_duration(os.path.join(folder, f)) for f in files])
    cumulative_min = np.cumsum(durations) / 60.0
    total_min = cumulative_min[-1]
    index = np.arange(1, len(files) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(index, cumulative_min, color="#2a78d6", linewidth=2)
    ax.annotate(
        f"{total_min:.1f} min ({total_min / 60:.2f} h)",
        xy=(index[-1], total_min),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color="#0b0b0b",
    )

    name = os.path.basename(folder)
    ax.set_title(
        f"{name}: cumulative duration, {len(files)} files "
        f"({files[0]} … {files[-1]}), mean {durations.mean():.2f} s",
        loc="left",
        color="#0b0b0b",
    )
    ax.set_xlabel("Sample index (ordered by file name)", color="#52514e")
    ax.set_ylabel("Total time (min)", color="#52514e")
    ax.set_xlim(0, index[-1])
    ax.set_ylim(0, total_min * 1.08)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    fig.tight_layout()

    out = args.out or f"{folder}_cumulative.png"
    fig.savefig(out, dpi=150)
    print(f"{len(files)} files, total {total_min:.2f} min -> {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
