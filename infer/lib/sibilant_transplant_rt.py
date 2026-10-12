"""
Streaming (real-time) version of the sibilant-transplant post-filter.

Same method as `infer/lib/sibilant_transplant.py` (offline, validated by ear):
long S-type hiss is detected in the INPUT, and inside those segments the
converted output above the crossover is replaced by the speaker's own hiss,
colour-EQ'd towards the converted hiss and following its loudness frame by frame.

What changes for streaming:
  * Look-ahead delay line. Output is delayed by `lookahead_s`, so every emitted
    block is processed with `lookahead_s` of future audio and `context_s` of past
    audio visible. With lookahead >= min_dur + fade the "long S" decision is the
    same as offline; with less it is relaxed (a hiss still running at the edge of
    the look-ahead is accepted once it has lasted `lookahead_s`).
  * Running statistics instead of whole-file ones: the -50 dB silence reference,
    the output bandwidth and the colour EQ are slow running averages kept on the
    object, so they don't jump between blocks.
  * Only the change (processed - original) is added to the output, and it is
    cross-faded at block boundaries; outside segments the output is untouched.
  * Bypass keeps the same delay, so the effect can be A/B'd live without a jump.

Usage (one call per audio block, fixed block size):
    st = RealtimeSibilantTransplant(sr=48000, block=6240)
    out_block = st.process(aligned_input_block, converted_block)
`aligned_input_block` must be the input samples that correspond to
`converted_block` sample for sample (same sample rate, full band).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


def _even(v):
    return max(2, int(round(v / 2.0)) * 2)


class RealtimeSibilantTransplant:
    def __init__(
        self,
        sr,
        block,
        lookahead_s=0.12,
        context_s=0.35,
        xfade_s=0.01,
        min_dur=0.08,
        min_centroid=6000,
        fade_ms=15,
        crossover=1500,
        eq_max_db=6,
        bandwidth_thr_db=-60.0,
        match_bandwidth=True,
        enabled=True,
    ):
        self.sr = int(sr)
        self.block = int(block)
        self.L = int(round(max(lookahead_s, 0.03) * self.sr))
        self.ctx = int(round(max(context_s, 0.15) * self.sr))
        self.xf = int(round(max(xfade_s, 0.002) * self.sr))
        self.xf = min(self.xf, self.L // 3)
        self.W = self.ctx + self.block + self.L
        self.in_hist = np.zeros(self.W, np.float32)
        self.out_hist = np.zeros(self.W, np.float32)

        # parameters (hot-updatable via set_params)
        self.enabled = bool(enabled)
        self.min_dur = float(min_dur)
        self.min_centroid = float(min_centroid)
        self.fade_ms = float(fade_ms)
        self.crossover = float(crossover)
        self.eq_max_db = float(eq_max_db)
        self.bandwidth_thr_db = float(bandwidth_thr_db)
        self.match_bandwidth = bool(match_bandwidth)

        # analysis sizes, scaled from the 48 kHz offline values
        k = self.sr / 48000.0
        self.det_nfft = _even(2048 * k)
        self.det_hop = max(1, int(round(0.005 * self.sr)))   # 5 ms (240 @ 48k)
        self.nfft = _even(1024 * k)
        self.hop = max(1, self.nfft // 8)                      # 2.7 ms (128 @ 48k)
        self.f_det = np.fft.rfftfreq(self.det_nfft, 1.0 / self.sr)
        self.f = np.fft.rfftfreq(self.nfft, 1.0 / self.sr)
        self.win = np.hanning(self.nfft + 1)[:-1].astype(np.float32)

        # running state
        self.level_ref = 0.0          # running loudest-frame energy (silence gate reference)
        self.bw_psd = None            # running long-term output power spectrum
        self.eq_px = None             # running input hiss spectrum (core frames)
        self.eq_py = None             # running output hiss spectrum (core frames)
        self.tail = None              # delta for [e1, e1+xf) from the previous block
        self.blocks = 0
        self.last_segments = 0        # segments touching the last emitted block
        self.total_segments_frames = 0

    # ------------------------------------------------------------------ params
    @property
    def delay_s(self):
        return self.L / self.sr

    def set_params(self, **kw):
        for k, v in kw.items():
            if v is None or not hasattr(self, k):
                continue
            setattr(self, k, type(getattr(self, k))(v))

    def reset(self):
        self.in_hist[:] = 0
        self.out_hist[:] = 0
        self.tail = None

    # ------------------------------------------------------------- helpers
    def _stft(self, x, n_fft, hop):
        return librosa.stft(x, n_fft=n_fft, hop_length=hop)

    def _detect(self, x):
        """Long S-type fricatives in the input window; same rules as offline
        `auto_segments`, with edge handling. Returns [(start_sample, end_sample)]."""
        hop, f = self.det_hop, self.f_det
        P = np.abs(self._stft(x, self.det_nfft, hop)) ** 2
        hi = P[(f >= 3500) & (f < 14000)].sum(0) + 1e-12
        lo = P[(f >= 80) & (f < 1200)].sum(0) + 1e-12
        en = P.sum(0) + 1e-12

        # running reference for the "-50 dB re loudest" silence gate:
        # peak-hold over the new frames, decaying 1 dB/s
        nF = P.shape[1]
        new = en[max(0, nF - self.block // hop - 1):]
        decay = 10 ** (-1.0 / 10 * self.block / self.sr)
        self.level_ref = max(self.level_ref * decay, float(new.max()))
        ref = max(self.level_ref, 1e-9)

        fr = (10 * np.log10(hi / lo) > 3) & (10 * np.log10(en / ref) > -50)
        cen = (P * f[:, None]).sum(0) / en
        idx = np.where(fr)[0]
        if len(idx) == 0:
            return []
        segs, s, p = [], idx[0], idx[0]
        for i in idx[1:]:
            if i > p + 4:  # merge gaps up to 20 ms
                segs.append((s, p)); s = i
            p = i
        segs.append((s, p))

        fade = self.fade_ms / 1000
        # how much hiss we can see past the emitted block before the window ends
        look = (self.L - self.xf) / self.sr - fade
        out = []
        for a, b in segs:
            dur = (b - a) * hop / self.sr
            c = np.median(cen[a:b + 1])
            open_right = b >= nF - 1 - 4
            need = min(self.min_dur, max(look, 0.0)) if open_right else self.min_dur
            if dur >= need and c >= self.min_centroid:
                out.append((a * hop, (b + 1) * hop))
        return out

    def _update_bandwidth(self, Pnew):
        if Pnew.size == 0:
            return
        p = Pnew.mean(1)
        if self.bw_psd is None:
            self.bw_psd = p
        else:
            a = min(1.0, self.block / (5.0 * self.sr))  # ~5 s time constant
            self.bw_psd = (1 - a) * self.bw_psd + a * p

    def _cutoff(self):
        if self.bw_psd is None or self.bw_psd.max() <= 0:
            return float(self.f[-1])
        Pdb = 10 * np.log10(self.bw_psd / self.bw_psd.max() + 1e-15)
        idx = np.where(Pdb > self.bandwidth_thr_db)[0]
        return float(self.f[idx.max()]) if len(idx) else float(self.f[-1])

    # --------------------------------------------------------------- process
    def process(self, in_block, out_block):
        n = self.block
        in_block = np.asarray(in_block, np.float32).reshape(-1)[:n]
        out_block = np.asarray(out_block, np.float32).reshape(-1)[:n]
        if len(in_block) < n:
            in_block = np.pad(in_block, (0, n - len(in_block)))
        if len(out_block) < n:
            out_block = np.pad(out_block, (0, n - len(out_block)))
        self.in_hist[:-n] = self.in_hist[n:]
        self.in_hist[-n:] = in_block
        self.out_hist[:-n] = self.out_hist[n:]
        self.out_hist[-n:] = out_block
        self.blocks += 1

        e0, e1 = self.ctx, self.ctx + n          # emitted region in window samples
        y_emit = self.out_hist[e0:e1].copy()
        if not self.enabled:
            self.tail = None
            self.last_segments = 0
            return y_emit

        try:
            delta = self._delta(e0, e1)
        except Exception:  # never break the audio stream
            import traceback
            traceback.print_exc()
            delta = None

        xf = self.xf
        cur = np.zeros(e1 - e0 + xf, np.float32) if delta is None else delta
        if self.tail is not None:
            w = np.sin(0.5 * np.pi * np.linspace(0, 1, xf, dtype=np.float32)) ** 2
            cur[:xf] = cur[:xf] * w + self.tail * (1 - w)
        self.tail = None if delta is None else delta[e1 - e0:].copy()
        res = y_emit + cur[: e1 - e0]
        return np.clip(res, -1.0, 1.0)

    def _delta(self, e0, e1):
        """Change (processed - converted) for window samples [e0, e1 + xf),
        or None when nothing is replaced there."""
        x, y = self.in_hist, self.out_hist
        sr, hop = self.sr, self.hop
        fade = max(self.fade_ms, 0.1) / 1000
        fade_n = fade * sr

        segs = self._detect(x)
        lo, hi = e0 - fade_n, e1 + self.xf + fade_n
        segs = [(s, e) for s, e in segs if e + fade_n > lo and s - fade_n < hi]
        self.last_segments = len(segs)

        Y = self._stft(y, self.nfft, hop)
        T = Y.shape[1]
        t = np.arange(T) * hop        # frame centres in samples
        # bandwidth: running long-term spectrum from the newest block's frames
        new_frames = (t >= self.W - self.block) & (t < self.W - self.nfft // 2)
        self._update_bandwidth(np.abs(Y[:, new_frames]) ** 2)
        if not segs:
            return None

        X = self._stft(x, self.nfft, hop)
        f = self.f
        cutoff = self._cutoff() if self.match_bandwidth else float(f[-1]) + 1.0
        below = f < cutoff

        m = np.zeros(T)
        for s, e in segs:
            r = np.clip(np.minimum((t - (s - fade_n)) / fade_n,
                                   ((e + fade_n) - t) / fade_n), 0, 1)
            m = np.maximum(m, r)

        # colour EQ from running averages over core frames of emitted blocks
        eq_max = max(self.eq_max_db, 0.0)
        core = m > 0.99
        core_new = core & (t >= e0) & (t < e1)
        if core_new.sum() > 0:
            px = (np.abs(X[:, core_new]) ** 2).mean(1) + 1e-12
            py = (np.abs(Y[:, core_new]) ** 2).mean(1) + 1e-12
            if self.eq_px is None:
                self.eq_px, self.eq_py = px, py
            else:
                a = min(0.3, core_new.sum() / 400.0)  # ~1 s of hiss to settle
                self.eq_px = (1 - a) * self.eq_px + a * px
                self.eq_py = (1 - a) * self.eq_py + a * py
        if eq_max > 0 and (self.eq_px is not None or core.sum() > 5):
            if self.eq_px is not None:
                px, py = self.eq_px, self.eq_py
            else:
                px = (np.abs(X[:, core]) ** 2).mean(1) + 1e-12
                py = (np.abs(Y[:, core]) ** 2).mean(1) + 1e-12
            g = 10 * np.log10(py / px)
            if below.any() and not below.all():
                g[~below] = g[below][-1]
            g = gaussian_filter1d(g, 8)
            norm = (f > 3000) & (f < min(12000.0, cutoff))
            g -= np.average(g[norm] if norm.any() else g[(f > 3000) & (f < 12000)])
            g = np.clip(g, -eq_max, eq_max)
            G = 10 ** (g / 20)
        else:
            G = np.ones(len(f))
        Xs = X * G[:, None]

        if self.match_bandwidth:
            roll = max(500.0, 0.1 * cutoff)
            Xs = Xs * np.clip((cutoff - f) / roll, 0, 1)[:, None]

        crossover = max(self.crossover, 1.0)
        band = (f >= crossover) & below
        if band.sum() < 4:
            band = f >= crossover
        lx = np.sqrt((np.abs(Xs[band]) ** 2).sum(0) + 1e-12)
        ly = np.sqrt((np.abs(Y[band]) ** 2).sum(0) + 1e-12)
        gain = uniform_filter1d(np.clip(ly / lx, 0.1, 10), 9)
        Xs *= gain[None, :]

        xo = np.clip((f - crossover * 0.67) / (crossover * 0.66), 0, 1)[:, None]
        M = m[None, :] * xo
        D = (Xs - Y) * M                      # change only
        d = librosa.istft(D, hop_length=hop, length=len(y)).astype(np.float32)
        return d[e0:e1 + self.xf].copy()
