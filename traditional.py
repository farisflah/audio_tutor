"""
traditional.py — Classical signal processing features (Tab 4)
=============================================================
Six traditional audio analysis techniques:
    zcr             — Zero Crossing Rate
    chroma          — Chromagram (12 pitch classes)
    autocorrelation — Periodicity / pitch estimation on a single frame
    fft_frame       — Full frequency spectrum at a chosen time point
    wavelet         — Continuous Wavelet Transform scalogram

Plus one comparison-only technique (Tab 5):
    dtw             — Dynamic Time Warping of MFCC sequences

Public API:
    compute(key, file_path, **kwargs) → (plotly_fig, stats_dict)

kwargs forwarded per-feature:
    frame_time      : float  — seconds into audio (autocorrelation, fft_frame)
    wavelet_family  : str    — wavelet name (wavelet); default "ricker"
"""

import numpy as np
import soundfile as sf
import scipy.signal
import librosa
import plotly.graph_objects as go

import config

SR   = config.SAMPLE_RATE
HOP  = config.HOP_LENGTH
DISP = config.DISPLAY_STEP


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════
def _load(file_path: str):
    data, orig_sr = sf.read(file_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if orig_sr != SR:
        n_target = int(len(data) * SR / orig_sr)
        data = scipy.signal.resample(data, n_target).astype(np.float32)
    return data, SR


def _time_axis(n_frames: int) -> np.ndarray:
    return np.array([i * HOP / SR for i in range(n_frames)], dtype=np.float32)


def _mm_ss_ticks(t_end: float):
    interval = 10 if t_end <= 120 else (30 if t_end <= 600 else 60)
    vals  = list(range(0, int(t_end) + interval + 1, interval))
    texts = [f"{v//60}:{v%60:02d}" for v in vals]
    return vals, texts


def _base_layout(title: str, height: int = 300, y_title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13, color=config.COLOR_TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=60, r=20, t=50, b=50),
        height=height,
        xaxis=dict(
            title="Time (mm:ss)",
            showgrid=True, gridcolor="#222",
            rangeslider=dict(visible=True, thickness=0.06, bgcolor="#1a1c23"),
            tickfont=dict(size=10, color="#ccc"),
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True, gridcolor="#222",
            tickfont=dict(size=10, color="#ccc"),
        ),
    )


def _set_time_ticks(fig: go.Figure, t_end: float):
    vals, texts = _mm_ss_ticks(t_end)
    fig.update_xaxes(
        tickmode="array", tickvals=vals, ticktext=texts,
        range=[0, min(60, t_end)],
    )


# ══════════════════════════════════════════════════════════════════
# 1. ZERO CROSSING RATE
# ══════════════════════════════════════════════════════════════════
def compute_zcr(file_path: str, **_):
    y, sr = _load(file_path)
    zcr   = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]

    n_frames = len(zcr)
    t        = _time_axis(n_frames)
    t_end    = float(t[-1]) if len(t) else 0.0

    t_d   = t[::DISP]
    zcr_d = zcr[::DISP]

    # Colour gradient: low ZCR = tonal (teal), high ZCR = noisy (purple)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_d, y=zcr_d,
        mode="lines",
        name="ZCR",
        line=dict(color=config.COLOR_PRIMARY, width=1.2),
        fill="tozeroy",
        fillcolor="rgba(0,200,160,0.10)",
        hovertemplate="Time: %{x:.2f}s<br>ZCR: %{y:.4f}<extra></extra>",
    ))

    # Annotate the mean as a reference line
    mean_zcr = float(zcr.mean())
    fig.add_hline(
        y=mean_zcr, line_dash="dot",
        line_color=config.COLOR_ACCENT,
        annotation_text=f"Mean ZCR: {mean_zcr:.4f}",
        annotation_font_color=config.COLOR_ACCENT,
    )

    layout = _base_layout(
        "Zero Crossing Rate  —  how often the waveform crosses zero",
        y_title="ZCR (crossings / sample)",
    )
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    stats = {
        "mean ZCR":  f"{mean_zcr:.4f}",
        "max ZCR":   f"{float(zcr.max()):.4f}",
        "std ZCR":   f"{float(zcr.std()):.4f}",
    }
    return fig, stats


# ══════════════════════════════════════════════════════════════════
# 2. CHROMAGRAM
# ══════════════════════════════════════════════════════════════════
CHROMA_LABELS = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]


def compute_chroma(file_path: str, **_):
    y, sr  = _load(file_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    # shape: (12, frames), values 0–1

    n_frames = chroma.shape[1]
    t        = _time_axis(n_frames)
    t_end    = float(t[-1]) if len(t) else 0.0

    fig = go.Figure(go.Heatmap(
        z=chroma,
        x=t,
        y=CHROMA_LABELS,
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        colorbar=dict(title="Energy", tickfont=dict(size=9, color="#ccc")),
        hovertemplate="Time: %{x:.2f}s<br>Pitch: %{y}<br>Energy: %{z:.3f}<extra></extra>",
    ))

    layout = _base_layout(
        "Chromagram  —  energy in each of the 12 pitch classes over time",
        height=340, y_title="Pitch Class",
    )
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    # Dominant pitch class over the whole file
    mean_per_class = chroma.mean(axis=1)
    dominant_idx   = int(np.argmax(mean_per_class))
    stats = {
        "dominant pitch": CHROMA_LABELS[dominant_idx],
        "mean energy":    f"{float(mean_per_class.mean()):.3f}",
        "max energy":     f"{float(mean_per_class.max()):.3f}",
    }
    return fig, stats


# ══════════════════════════════════════════════════════════════════
# 3. AUTOCORRELATION
# ══════════════════════════════════════════════════════════════════
def compute_autocorrelation(file_path: str, frame_time: float = 0.5, **_):
    """
    Compute autocorrelation of a single frame centred at frame_time seconds.
    Shows periodicity and enables pitch estimation.
    """
    y, sr = _load(file_path)

    # Extract frame — 50 ms centred at frame_time
    frame_len  = int(0.05 * sr)    # 50 ms
    frame_len  = max(frame_len, 256)
    centre     = int(frame_time * sr)
    start      = max(0, centre - frame_len // 2)
    frame      = y[start : start + frame_len]
    if len(frame) < frame_len:
        frame = np.pad(frame, (0, frame_len - len(frame)))

    # Normalised autocorrelation via FFT (faster than np.correlate for long frames)
    n   = len(frame)
    f   = np.fft.rfft(frame, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    # Normalise so lag-0 = 1.0
    acf = acf / (acf[0] + 1e-9)

    lags    = np.arange(n)
    lag_ms  = lags / sr * 1000   # convert samples → milliseconds

    # Only show the useful range: lag 1ms (1000 Hz max) to 20ms (50 Hz min)
    min_lag = max(1, int(sr / 1000))    # 1 Hz upper limit
    max_lag = min(n - 1, int(sr / 50))  # 50 Hz lower limit
    acf_view    = acf[min_lag : max_lag]
    lag_ms_view = lag_ms[min_lag : max_lag]

    # Estimate fundamental frequency from first peak after lag 0
    pitch_hz = None
    if len(acf_view) > 3:
        from scipy.signal import find_peaks
        peaks, props = find_peaks(acf_view, height=0.3, distance=max(1, int(sr/1000)))
        if len(peaks) > 0:
            best_peak  = peaks[np.argmax(props["peak_heights"])]
            lag_samp   = best_peak + min_lag
            pitch_hz   = sr / lag_samp if lag_samp > 0 else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lag_ms_view, y=acf_view,
        mode="lines", name="ACF",
        line=dict(color=config.COLOR_SECONDARY, width=1.5),
        hovertemplate="Lag: %{x:.2f} ms<br>ACF: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=0, line_dash="dash", line_color="#444",
        annotation_text="Zero", annotation_font_color="#666",
    )

    pitch_label = f" — est. pitch: {pitch_hz:.1f} Hz" if pitch_hz else " — no clear pitch"
    fig.update_layout(
        title=dict(
            text=f"Autocorrelation at t={frame_time:.2f}s{pitch_label}",
            font=dict(size=13, color=config.COLOR_TEXT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=60, r=20, t=50, b=50),
        height=300,
        xaxis=dict(
            title="Lag (ms)",
            showgrid=True, gridcolor="#222",
            tickfont=dict(size=10, color="#ccc"),
        ),
        yaxis=dict(
            title="Normalised ACF", range=[-1.05, 1.05],
            showgrid=True, gridcolor="#222",
            tickfont=dict(size=10, color="#ccc"),
        ),
    )

    stats = {
        "frame time":  f"{frame_time:.2f} s",
        "frame length": f"{len(frame)} samples ({len(frame)/sr*1000:.0f} ms)",
    }
    if pitch_hz:
        stats["est. pitch"] = f"{pitch_hz:.1f} Hz"

    return fig, stats


# ══════════════════════════════════════════════════════════════════
# 4. SINGLE-FRAME FFT SPECTRUM
# ══════════════════════════════════════════════════════════════════
def compute_fft_frame(file_path: str, frame_time: float = 1.0, **_):
    """
    Full magnitude spectrum (FFT) at a single chosen time point.
    Shows exactly which frequencies are present at that instant.
    """
    y, sr = _load(file_path)

    n_fft     = config.N_FFT
    centre    = int(frame_time * sr)
    start     = max(0, centre - n_fft // 2)
    frame     = y[start : start + n_fft]
    if len(frame) < n_fft:
        frame = np.pad(frame, (0, n_fft - len(frame)))

    # Apply Hann window to reduce spectral leakage
    window    = np.hanning(n_fft).astype(np.float32)
    windowed  = frame * window
    spectrum  = np.abs(np.fft.rfft(windowed))

    # Convert to dB
    spectrum_db = 20 * np.log10(spectrum + 1e-9)
    spectrum_db -= spectrum_db.max()   # normalise so peak = 0 dB

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Only display up to 8 kHz (most musical/speech content is here)
    show_mask   = freqs <= 8000
    freqs_show  = freqs[show_mask]
    spec_show   = spectrum_db[show_mask]

    # Find top-5 peaks for annotation
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spec_show, height=-40, distance=10)
    if len(peaks) > 0:
        top_idx   = peaks[np.argsort(spec_show[peaks])[-5:]]
    else:
        top_idx   = []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs_show / 1000,   # display as kHz
        y=spec_show,
        mode="lines",
        name="Magnitude",
        line=dict(color=config.COLOR_PRIMARY, width=1.2),
        fill="tozeroy",
        fillcolor="rgba(0,200,160,0.08)",
        hovertemplate="Freq: %{x:.3f} kHz<br>Level: %{y:.1f} dB<extra></extra>",
    ))

    # Mark top peaks
    for idx in top_idx:
        fig.add_annotation(
            x=freqs_show[idx] / 1000,
            y=spec_show[idx],
            text=f"{freqs_show[idx]:.0f} Hz",
            showarrow=True, arrowhead=2,
            arrowcolor=config.COLOR_ACCENT,
            font=dict(size=9, color=config.COLOR_ACCENT),
            ax=0, ay=-25,
        )

    fig.update_layout(
        title=dict(
            text=f"Frequency Spectrum (FFT) at t={frame_time:.2f}s  "
                 f"— Hann window, {n_fft}-pt FFT",
            font=dict(size=13, color=config.COLOR_TEXT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=60, r=20, t=50, b=50),
        height=320,
        xaxis=dict(
            title="Frequency (kHz)",
            showgrid=True, gridcolor="#222",
            tickfont=dict(size=10, color="#ccc"),
        ),
        yaxis=dict(
            title="Magnitude (dB, peak-normalised)",
            showgrid=True, gridcolor="#222",
            tickfont=dict(size=10, color="#ccc"),
            range=[-80, 5],
        ),
    )

    # Dominant frequency band label
    dominant_freq = float(freqs_show[np.argmax(spec_show)])
    if dominant_freq < 300:
        band = "Sub-bass / Bass"
    elif dominant_freq < 2000:
        band = "Mid-range"
    elif dominant_freq < 5000:
        band = "Presence"
    else:
        band = "Brilliance / Air"

    stats = {
        "frame time":  f"{frame_time:.2f} s",
        "FFT size":    str(n_fft),
        "freq res":    f"{sr/n_fft:.1f} Hz/bin",
        "dominant":    f"{dominant_freq:.0f} Hz ({band})",
    }
    return fig, stats


# ══════════════════════════════════════════════════════════════════
# 5. WAVELET TRANSFORM (Continuous Wavelet Transform scalogram)
# ══════════════════════════════════════════════════════════════════
# Supported families and their implementation paths:
#   "ricker"   — scipy.signal.ricker      (no extra dependency)
#   "morlet"   — scipy.signal.morlet      (no extra dependency)
#   "db4"      — PyWavelets pywt.cwt      (needs pywt)
#   "haar"     — PyWavelets pywt.cwt      (needs pywt)

_WAVELET_SCIPY = {"ricker", "morlet"}   # handled by scipy.signal.cwt
_WAVELET_PWT   = {"db4", "haar"}        # handled by PyWavelets


def _ricker(points: int, a: float) -> np.ndarray:
    """
    Ricker (Mexican hat) wavelet — manual implementation.
    scipy.signal.ricker was removed in scipy 1.12.
    """
    A    = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq  = a ** 2
    vec  = np.arange(0, points) - (points - 1.0) / 2
    tsq  = vec ** 2
    mod  = (1 - tsq / wsq)
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss


def _morlet2(points: int, a: float) -> np.ndarray:
    """
    Real Morlet wavelet (w=5) — replaces scipy.signal.morlet2 for cwt().
    scipy.signal.morlet2 signature changed; defining it manually is safer.
    """
    w    = 5.0
    x    = np.arange(0, points, dtype=np.float64) - (points - 1.0) / 2
    x    = x / a
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x ** 2)
    wavelet *= (np.pi ** (-0.25))
    return wavelet.real.astype(np.float32)


def _cwt_scipy(y: np.ndarray, wavelet: str, n_scales: int = 64) -> np.ndarray:
    """Return scalogram magnitude — manual CWT loop (avoids deprecated scipy API)."""
    max_scale = max(2, min(n_scales, len(y) // 4))
    scales    = np.geomspace(1, max_scale, num=n_scales).astype(np.float32)
    wavelet_fn = _ricker if wavelet == "ricker" else _morlet2

    n  = len(y)
    cwt = np.zeros((n_scales, n), dtype=np.float32)
    for i, s in enumerate(scales):
        # Wavelet length: 10 × scale, capped to signal length, always odd
        wlen   = min(n, max(3, int(10 * s) | 1))
        kernel = wavelet_fn(wlen, s).astype(np.float32)
        # Convolve and take centre
        conv   = np.convolve(y, kernel[::-1], mode="same")
        cwt[i] = conv / (s ** 0.5 + 1e-9)   # scale normalisation

    return np.abs(cwt), scales


def _haar(points: int, a: float) -> np.ndarray:
    """
    Haar wavelet — manual implementation, no PyWavelets needed.
    Square wave: +1 for first half, -1 for second half, scaled by 1/sqrt(a).
    """
    kernel = np.zeros(points, dtype=np.float32)
    mid    = points // 2
    if mid > 0:
        kernel[:mid]  =  1.0
        kernel[mid:]  = -1.0
    return kernel / (np.sqrt(a) + 1e-9)


def _db4(points: int, a: float) -> np.ndarray:
    """
    Daubechies 4 (db4) wavelet — manual implementation, no PyWavelets needed.
    Uses the known db4 scaling filter coefficients, builds the mother wavelet
    by convolution cascade, then scales to the requested length/scale.
    """
    # db4 scaling filter coefficients (h0..h7)
    h = np.array([
        0.23037781330885523,
        0.7148465705525415,
        0.6308807679295904,
        -0.02798376941698385,
        -0.18703481171888114,
        0.03084138183598667,
        0.032883011666982945,
        -0.010597401784997278,
    ], dtype=np.float64)
    # Mother wavelet from scaling filter (g = alternating-flip of h)
    g = np.array([(-1)**k * h[len(h) - 1 - k] for k in range(len(h))], dtype=np.float64)

    # Build at native scale via 3 iterations of upsampling + convolution
    psi = g.copy()
    for _ in range(3):
        up = np.zeros(len(psi) * 2 - 1, dtype=np.float64)
        up[::2] = psi
        psi = np.convolve(up, h, mode="full")

    # Resample to requested `points` using linear interpolation
    x_orig = np.linspace(0, 1, len(psi))
    x_new  = np.linspace(0, 1, points)
    kernel = np.interp(x_new, x_orig, psi).astype(np.float32)
    return kernel / (np.sqrt(a) + 1e-9)


def _cwt_pywt(y: np.ndarray, wavelet: str, n_scales: int = 64):
    """
    Manual CWT for db4 / haar — no PyWavelets dependency.
    Falls back to PyWavelets if installed, otherwise uses manual kernels.
    """
    try:
        import pywt
        max_scale = max(2, min(n_scales, len(y) // 4))
        scales    = np.geomspace(1, max_scale, num=n_scales).astype(np.float32)
        coeff, _  = pywt.cwt(y.astype(np.float64), scales, wavelet)
        return np.abs(coeff).astype(np.float32), scales
    except ImportError:
        pass

    # PyWavelets not installed — use manual kernels
    wavelet_fn = _haar if wavelet == "haar" else _db4
    max_scale  = max(2, min(n_scales, len(y) // 4))
    scales     = np.geomspace(1, max_scale, num=n_scales).astype(np.float32)
    n          = len(y)
    cwt        = np.zeros((n_scales, n), dtype=np.float32)
    for i, s in enumerate(scales):
        wlen   = min(n, max(3, int(10 * s) | 1))
        kernel = wavelet_fn(wlen, float(s))
        conv   = np.convolve(y, kernel[::-1], mode="same")
        cwt[i] = conv
    return np.abs(cwt), scales


def compute_wavelet(file_path: str, wavelet_family: str = "ricker", **_):
    """
    Continuous Wavelet Transform (CWT) scalogram.

    wavelet_family options:
        "ricker"  — Mexican hat (scipy, no extra install)
        "morlet"  — Complex Morlet (scipy, no extra install)
        "db4"     — Daubechies 4 (requires PyWavelets)
        "haar"    — Haar (requires PyWavelets)
    """
    y, sr = _load(file_path)

    # Downsample heavily for CWT performance — target ~4000 samples max
    max_samples = 4000
    if len(y) > max_samples:
        step = len(y) // max_samples
        y_cwt = y[::step]
    else:
        y_cwt = y
        step  = 1

    # Compute CWT
    n_scales = 64
    if wavelet_family in _WAVELET_SCIPY:
        scalogram, scales = _cwt_scipy(y_cwt, wavelet_family, n_scales)
    else:
        scalogram, scales = _cwt_pywt(y_cwt, wavelet_family, n_scales)

    # Time axis (seconds)
    n_samples   = scalogram.shape[1]
    t_total     = len(y) / sr
    t           = np.linspace(0, t_total, n_samples, endpoint=False)

    # Colour clip
    vmax = float(np.percentile(scalogram, 98)) or 1.0
    vmin = 0.0

    # Scale labels: approximate pseudo-frequency (SR / scale) in Hz
    pseudo_hz = [f"{sr / (s * step):.0f} Hz" for s in scales]

    fig = go.Figure(go.Heatmap(
        z=scalogram,
        x=t,
        y=pseudo_hz,
        colorscale=config.COLORSCALE_WAVELET,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="|CWT|", tickfont=dict(size=9, color="#ccc")),
        hovertemplate="Time: %{x:.2f}s<br>Scale: %{y}<br>|CWT|: %{z:.3f}<extra></extra>",
    ))

    family_names = {
        "ricker": "Ricker (Mexican hat)",
        "morlet": "Morlet",
        "db4":    "Daubechies 4",
        "haar":   "Haar",
    }
    fname = family_names.get(wavelet_family, wavelet_family)

    fig.update_layout(
        title=dict(
            text=f"Wavelet Scalogram — {fname}  ({n_scales} scales, "
                 f"{n_samples} time steps)",
            font=dict(size=13, color=config.COLOR_TEXT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=80, r=20, t=50, b=50),
        height=400,
        xaxis=dict(
            title="Time (s)",
            showgrid=True, gridcolor="#222",
            tickfont=dict(size=10, color="#ccc"),
        ),
        yaxis=dict(
            title="Pseudo-frequency (Hz)",
            showgrid=True, gridcolor="#222",
            tickfont=dict(size=10, color="#ccc"),
            # Show every 8th label to avoid crowding
            tickmode="array",
            tickvals=pseudo_hz[::8],
            ticktext=pseudo_hz[::8],
        ),
    )

    peak_scale_idx = int(np.argmax(scalogram.mean(axis=1)))
    stats = {
        "wavelet":    fname,
        "n_scales":   str(n_scales),
        "time steps": str(n_samples),
        "peak scale": pseudo_hz[peak_scale_idx],
    }
    return fig, stats


# ══════════════════════════════════════════════════════════════════
# 6. DTW — Dynamic Time Warping (comparison only, Tab 5)
# ══════════════════════════════════════════════════════════════════
def compute_dtw(path_a: str, path_b: str, name_a: str = "A", name_b: str = "B"):
    """
    Align two audio files using Dynamic Time Warping on their MFCC sequences.
    Returns (plotly_fig, stats_dict).

    The figure shows:
    - Top-left: cost matrix with optimal warping path overlaid in teal
    - Bottom row: MFCC heatmaps for both files at the aligned scale
    """
    import librosa.sequence

    # Load both files
    def _load_mfcc(fp):
        data, orig_sr = sf.read(fp, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if orig_sr != SR:
            n_target = int(len(data) * SR / orig_sr)
            data = scipy.signal.resample(data, n_target).astype(np.float32)
        M = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=config.N_MFCC,
                                 hop_length=HOP, n_fft=config.N_FFT)
        return M.T   # (frames, n_mfcc) — librosa.sequence.dtw expects this

    mfcc_a = _load_mfcc(path_a)   # (T_a, n_mfcc)
    mfcc_b = _load_mfcc(path_b)   # (T_b, n_mfcc)

    # ── Downsample MFCCs to max 500 frames each to avoid 200 MB message limit ──
    # (3-min audio at hop=512, sr=22050 → ~7700 frames → 7700×7700 matrix = ~450 MB)
    MAX_FRAMES = 500
    def _ds(M, max_f):
        """Temporally downsample MFCC matrix (frames × n_mfcc) if needed."""
        if M.shape[0] <= max_f:
            return M
        step = M.shape[0] // max_f
        return M[::step][:max_f]

    mfcc_a_ds = _ds(mfcc_a, MAX_FRAMES)
    mfcc_b_ds = _ds(mfcc_b, MAX_FRAMES)

    # librosa DTW — returns (cost_matrix, warping_path)
    # cost_matrix shape: (T_a_ds, T_b_ds)
    D, wp = librosa.sequence.dtw(
        X=mfcc_a_ds.T,   # (n_mfcc, T_a_ds)
        Y=mfcc_b_ds.T,   # (n_mfcc, T_b_ds)
        metric="euclidean",
    )
    # wp is (path_len, 2) — rows are [idx_a, idx_b]
    path_a_idx = wp[:, 0]
    path_b_idx = wp[:, 1]

    dtw_distance = float(D[wp[-1, 0], wp[-1, 1]])

    # Convert frame indices to seconds for labelling
    t_a = path_a_idx * HOP / SR
    t_b = path_b_idx * HOP / SR
    t_a_total = mfcc_a.shape[0] * HOP / SR
    t_b_total = mfcc_b.shape[0] * HOP / SR

    # ── Build figure: 3 rows, 1 column — avoids all colorbar/label collisions ──
    # Row 1: cost matrix + warping path (tall)
    # Row 2: MFCC A (short)
    # Row 3: MFCC B (short)
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "DTW Cost Matrix + Warping Path",
            f"MFCC — {name_a}",
            f"MFCC — {name_b}",
        ],
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.18,
    )

    # Row 1 — cost matrix
    fig.add_trace(go.Heatmap(
        z=D,
        colorscale="YlOrRd",
        colorbar=dict(
            title=dict(text="Cost", font=dict(size=10, color="#ccc")),
            len=0.45,
            thickness=14,
            x=1.01,
            y=0.76,
            tickfont=dict(size=8, color="#ccc"),
        ),
        hovertemplate=(f"{name_a} frame %{{y}}<br>"
                       f"{name_b} frame %{{x}}<br>"
                       f"Cost: %{{z:.2f}}<extra></extra>"),
        showscale=True,
    ), row=1, col=1)

    # Row 1 — warping path overlay
    fig.add_trace(go.Scatter(
        x=path_b_idx.tolist(),
        y=path_a_idx.tolist(),
        mode="lines",
        name="Warping path",
        line=dict(color=config.COLOR_PRIMARY, width=2.0),
        hovertemplate=f"{name_a}: %{{y}} → {name_b}: %{{x}}<extra></extra>",
    ), row=1, col=1)

    # Row 2 — MFCC A
    fig.add_trace(go.Heatmap(
        z=mfcc_a_ds.T,
        colorscale=config.COLORSCALE_MFCC,
        showscale=False,
        hovertemplate="Frame %{x}<br>MFCC %{y}: %{z:.1f}<extra></extra>",
    ), row=2, col=1)

    # Row 3 — MFCC B
    fig.add_trace(go.Heatmap(
        z=mfcc_b_ds.T,
        colorscale=config.COLORSCALE_MFCC,
        showscale=False,
        hovertemplate="Frame %{x}<br>MFCC %{y}: %{z:.1f}<extra></extra>",
    ), row=3, col=1)

    _ax = dict(showgrid=True, gridcolor="#222", tickfont=dict(size=9, color="#ccc"))
    fig.update_layout(
        title=dict(
            text=(f"DTW Alignment  —  distance: {dtw_distance:.1f}  "
                  f"| {name_a}: {t_a_total:.1f}s  |  {name_b}: {t_b_total:.1f}s"),
            font=dict(size=13, color=config.COLOR_TEXT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=70, r=80, t=70, b=50),
        height=780,
        showlegend=False,
    )
    fig.update_xaxes(**_ax)
    fig.update_yaxes(**_ax)
    fig.update_xaxes(title_text=f"{name_b} frames", row=1, col=1)
    fig.update_yaxes(title_text=f"{name_a} frames", row=1, col=1)
    fig.update_xaxes(title_text="Frame",       row=2, col=1)
    fig.update_yaxes(title_text="MFCC coeff",  row=2, col=1)
    fig.update_xaxes(title_text="Frame",       row=3, col=1)
    fig.update_yaxes(title_text="MFCC coeff",  row=3, col=1)

    # Similarity interpretation
    if dtw_distance < 100:
        similarity = "Very similar"
    elif dtw_distance < 300:
        similarity = "Moderately similar"
    elif dtw_distance < 600:
        similarity = "Different"
    else:
        similarity = "Very different"

    stats = {
        "DTW distance": f"{dtw_distance:.1f}",
        "similarity":   similarity,
        f"{name_a} dur": f"{t_a_total:.2f}s",
        f"{name_b} dur": f"{t_b_total:.2f}s",
        "path length":  str(len(path_a_idx)),
    }
    return fig, stats


# ══════════════════════════════════════════════════════════════════
# PUBLIC DISPATCH
# ══════════════════════════════════════════════════════════════════
_DISPATCH = {
    "zcr":             compute_zcr,
    "chroma":          compute_chroma,
    "autocorrelation": compute_autocorrelation,
    "fft_frame":       compute_fft_frame,
    "wavelet":         compute_wavelet,
}


def compute(key: str, file_path: str, **kwargs):
    """
    Single entry point called by gui.py.
    Returns (plotly_fig, stats_dict).
    kwargs: frame_time=float for autocorrelation and fft_frame;
            wavelet_family=str for wavelet.
    """
    fn = _DISPATCH.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown traditional feature key '{key}'. "
            f"Valid keys: {list(_DISPATCH)}"
        )
    return fn(file_path, **kwargs)