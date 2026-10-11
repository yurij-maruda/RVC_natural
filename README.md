# RVC_natural

**A fork of [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) that removes the robotic texture from long S / SH sounds**, the most noticeable tell when converting a voice across a large pitch gap (e.g. male → female, transpose +12). 
Tested on model v1 only.

Everything else is upstream RVC: training, inference, the real-time GUI and all models work as before. The fix is an optional post-filter, off by default.

- Original project README: [upstream on GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) · [English](./docs/en/README.en.md) · [local copy of the upstream README](./docs/README.upstream.md)

---

## Why this fork exists

With a well-trained, full-band RVC model, male → female conversion can sound natural, except for one stubborn artifact: **long, sustained S and SH sounds come out robotic**. Short sibilants are masked by the sounds around them; long ones expose a synthetic texture inside the hiss.

The usual knobs don't fix it:

- **Protect, index rate, pitch method, checkpoints, retraining:** no audible improvement on this artifact.
- **Pitch tracking:** sibilants have no pitch at all, so pitch tweaks don't reach the problem.
- **De-essing / EQ:** masks harshness, but the robotic texture stays.
- **Other engines** (e.g. Seed-VC): no sibilant problem, but many new ones (flattened intonation, accent, 22 kHz output).

The reason is phonetic. Voiced sounds come from the vocal folds and carry the speaker's identity, which is what RVC must rebuild. **S, SH, CH and F are turbulent air at the teeth/tongue: no pitch and very little speaker identity.** RVC's generator still re-synthesizes them, and in long hiss its synthetic texture becomes audible. Your own recorded hiss, on the other hand, is already almost the right sound.

## The method: sibilant transplant

For every conversion, the original input recording is analysed fresh, and in exactly the moments where it contains a long S/SH, the converted output above ~1.5 kHz is replaced by **the speaker's own real hiss**, shaped to fit the converted voice.

```
frequency
 high ┤  ████ your real hiss ████     ← replaced, inside the detected segment only
      │
1.5kHz┤ - - - soft crossover - - -
      │
  low ┤  ░░░░ converted voice ░░░░    ← always kept
      └──── time: long S, with 15 ms fades ────
```

**1. Detection (on the input, every call; no stored analysis)**
- STFT at 48 kHz, 5 ms hop.
- A frame is a fricative when energy 3.5–14 kHz exceeds energy 80–1200 Hz by > 3 dB, and the frame is not silence (> −50 dB re file max).
- Frames are merged into segments (gaps ≤ 20 ms).
- A segment is kept when it lasts ≥ *Min S duration* (80 ms) and its median spectral centroid is ≥ *Min hiss centroid* (6000 Hz = S only; 3000 Hz = S and SH).

**2. Replacement (in the time–frequency domain)**
- **Time:** the segment, with linear fades at both ends.
- **Frequency:** above the crossover (default 1500 Hz, soft blend around it). Below it, the converted voice is always kept, so no trace of the source voice leaks in at segment edges.
- **Loudness:** taken from the converted voice **frame by frame** (2.7 ms), so the hiss follows the converted voice's dynamics.
- **Colour:** an optional mild EQ (≤ ±6 dB), learned from the replaced segments, bends the input hiss towards the converted S colour.
- **Bandwidth:** the output's bandwidth is measured, and the input hiss is low-passed to it, so a band-limited model (e.g. ~10 kHz) never gets over-bright patches.

Nothing is pitch-shifted: hiss has no pitch, and the transplant must happen **after** RVC, because RVC re-synthesizes everything it is given.

## Usage

### WebUI (single and batch inference)
Enable **"Sibilant transplant (fix robotic long S)"** in the inference tab. The settings are always visible below it, and a collapsible panel explains each one. After converting, the info text shows how many segments were replaced and the detected output bandwidth.

| Setting | Default | Meaning |
|---|---|---|
| Min S duration (ms) | 80 | Only hiss at least this long is replaced. 0 = every detected hiss. |
| Min hiss centroid (Hz) | 6000 | How bright the hiss must be. S ≈ 6–9 kHz, SH ≈ 3–4.5 kHz. Use 3000 to include SH. |
| Fade (ms) | 15 | Cross-fade at both ends of each segment. |
| Crossover (Hz) | 1500 | Below: converted voice kept. Above: input hiss. |
| Colour EQ max (dB) | 6 | How strongly the hiss tone is bent towards the converted S. 0 = off. |
| Match output bandwidth | on | Low-pass the hiss to the converted voice's bandwidth. |

The input must be a file on disk; the original file is reloaded at full quality, because the pipeline's internal 16 kHz copy has nothing above 8 kHz.

### Command line
```bash
# transplant: input recording + its RVC conversion -> fixed file
python tools/artifact_detect/transplant.py INPUT.wav RVC_OUTPUT.wav [OUT.wav] \
    [--min-dur 0.08] [--min-centroid 6000] [--fade-ms 15] [--crossover 1500] \
    [--eq-max-db 6] [--bandwidth-thr-db -60] [--no-match-bandwidth] [--write-segments]

# analysis: show which segments would be replaced (log, CSV, Audacity labels, plot)
python tools/artifact_detect/artifact_detect.py INPUT.wav RVC_OUTPUT.wav
```
`--write-segments` and the analysis tool both write Audacity label files, so you can check every replaced spot by ear.

## Preserving finished models

Trained models are not part of the repository (`assets/` and `logs/` are git-ignored, and model files are too large for GitHub). Use `tools/model_backup.py` to pack a finished model into one zip you can keep in private cloud storage.

A backup contains only what is needed to **use** the model, with repo-relative paths preserved:

| File in the zip | Purpose |
|---|---|
| `assets/weights/<model>.pth` | the model used for conversion |
| `assets/indices/<model>_added_….index` | feature index (used by *search feature ratio*) |
| `logs/<model>/config.json` | training config (sample rate, architecture) |
| `MODEL_BACKUP_<model>.json` | manifest: file list, sizes, SHA-256 checksums, your notes |

Output directory: `assets/model_backups/<model_name>.zip`.

Training checkpoints (`G_*.pth` / `D_*.pth`) are not included, so a backup can't be used to resume training.

```bash
# see which models exist and whether their files are complete
python tools/model_backup.py --list

# back up one model -> assets/model_backups/<model>.zip
python tools/model_backup.py v6_MyVoice --notes "transpose +12, protect 0.33, transplant centroid 3000"

# an existing <model>.zip is kept; the new one gets a timestamp, unless you pass --overwrite
python tools/model_backup.py v6_MyVoice --overwrite

# check a backup's integrity (e.g. after downloading it again)
python tools/model_backup.py --verify assets/model_backups/v2_MyVoice.zip
```

Use `--notes` to record the conversion settings that work for the model; they are stored in the manifest and shown by `--verify`.

**Restore:** unpack the zip at the repository root, keeping the folder structure, so the files land in `assets/weights/`, `assets/indices/` and `logs/<model>/`. Then restart the WebUI and pick the model and its index as usual. Run `--verify` on the zip first if it has been through cloud storage.

## Files

| Path | What |
|---|---|
| `infer/lib/sibilant_transplant.py` | Detection, transplant and the inference entry point (`apply_sibilant_transplant`) |
| `infer/vc/modules.py` | Hook after single / batch conversion; never fails the conversion (falls back to the unprocessed output) |
| `webui.py` | Checkbox, settings and help panel |
| `tools/artifact_detect/` | CLI transplant and analysis tool |
| `tools/model_backup.py` | Back up / verify finished models (inference files only) |

## Limitations

- **Offline only for now** (WebUI inference). Real-time integration is the next step; the duration rule needs ~80 ms of look-ahead, or can be relaxed to "all S-type hiss".
- **Use a full-band model** (output reaching ~16–20 kHz). On band-limited models the transplant blends in but can't fix the overall dull sound. Check your training data's spectrum before training.
- **Thresholds were tuned on one male → female voice pair.** Other pairs (female → male, lisping speakers) may need different settings.
- Voiced fricatives (Z, ZH) and SH at the very start of an utterance may be missed.
- The transplanted hiss is literally the speaker's own; this is intended, since sibilants carry very little speaker identity.

## Responsible use

Convert only voices you have permission to use, and don't use this to impersonate real people. Publish code, not other people's voice models or datasets.

## Credits & license

Based on [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by the RVC-Project team, MIT license (see [LICENSE](./LICENSE)). The sibilant-transplant post-filter and analysis tools in this fork are released under the same license.
