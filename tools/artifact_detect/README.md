# artifact_detect — offline "robotic artifact" detector

Standalone analysis tool (it does not touch the RVC pipeline). It compares an input recording with its
RVC conversion and flags places where the **output is voiced or tonal while the input is noise-like**, for
example a pitched buzz inside or at the edges of S / SH / CH.

Run it with the project venv: `.\.venv\Scripts\python.exe` (or activate `.venv` first).

## Usage

```
python tools/artifact_detect/artifact_detect.py INPUT.wav OUTPUT.wav [options]
```

| option | default | meaning |
|---|---|---|
| `--out-dir DIR` | `<output_stem>_artifacts/` next to OUTPUT | where reports go |
| `--method {voicing,flatness,hpss}` | `voicing` | detector (see below) |
| `--threshold F` | `0.15` (voicing), `0.25` (legacy) | event threshold |
| `--input-voicing-max F` | `0.35` | voicing: only score frames where the input's voicing probability is below this |
| `--fmin / --fmax` | `2000 / 12000` (voicing), `1000 / 12000` (legacy) | band range (Hz) |
| `--bands N` | half-octave (voicing), `18` (legacy) | number of log-spaced bands; each band needs ≥ 8 FFT bins |
| `--start / --end SEC` | – | zoom the plot (the analysis still covers the whole file) |
| `--show` | off | also open the matplotlib window |
| `--max-lag-ms` | `100` | alignment search range |

## Methods

**`voicing` (default)**
- Computes the pYIN voicing probability (16 kHz, fmin 60, fmax 1000, frame 1024, hop 80 = 5 ms) for both signals.
- Computes `artifact(t) = clip(vp_out − vp_in, 0, 1)`, but only where `vp_in < --input-voicing-max` and the
  output frame is above −55 dB. The result is smoothed over 3 frames.
- Events come from this curve only: threshold, then merge gaps under 20 ms, then drop events shorter than 10 ms.
- Band localization, used only for the heatmap and the "dominant band" column: the per-band HPSS harmonic
  ratio, output minus input, in half-octave bands from 2 to 12 kHz, on the frames where the artifact is above 0.

**`flatness` / `hpss` (legacy, for comparison)** compute the per-band tonality mismatch
`A = clip(T_out − T_in, 0, 1) · energy gate` and its energy-weighted mean across bands. Both depend on pitch:
with a +12 semitone shift, the output's harmonics are sparser in each band, so vowels look "more tonal".

## Outputs

The `voicing` run writes `artifacts.*`. Legacy methods add a suffix (`artifacts_flatness.*`, `artifacts_hpss.*`).

- `.txt`: header plus one line per event: time, duration, peak and mean, dominant band, mean `vp_in` and
  `vp_out`, `fricative-input` (the event's own input energy at 3.5–12 kHz is greater than at 80–1200 Hz),
  and `near-fricative` (an input fricative frame lies within ±60 ms)
- `.csv`: the same events in machine-readable form
- `_audacity_labels.txt`: in Audacity, use *File → Import → Labels…*
- `.png`: input and output spectrograms, the band heatmap, and voicing in/out with the artifact curve and events

All times are on the **input** timeline. The output was shifted by the reported lag.

## Reusable pieces

`voicing_prob`, `voicing_artifact`, `fricative_frames`, `make_bands`, `band_energy`, `tonality_flatness`,
`tonality_hpss`, `band_localization`, `energy_gate`, `artifact_map` and `frame_scores` take plain arrays:
audio, or STFT magnitudes `(n_bins, n_frames)`.

## Known limitation (ch_s_sh_test vs output_trans_12_test_4)

pYIN often scores the male input as weakly voiced (below 0.35) in frames that are clearly voiced and
dominated by low frequencies: creaky phonation, vowel onsets and offsets, nasals. The output stays strongly
voiced there, so most `voicing` events fall in those frames rather than in sibilants. Inside the input's
fricatives, the output's pYIN voicing does not rise above the input's.
