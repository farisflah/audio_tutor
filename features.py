"""
features.py — Acoustic Feature Extraction (no DB, no caching)
==============================================================
Each public function takes a file path and returns:
    (plotly_fig, data_dict)

data_dict always contains:
    feature_vector      : list[float]  — fixed-length ML input vector
    feature_metadata    : dict         — stats, event info, shape info
    feature_time_series : list[float]  — downsampled curve for plotting

compute(key, file_path) is the single public entry point used by gui.py.
"""

import numpy as np
import soundfile as sf
import scipy.signal
import librosa
import plotly.graph_objects as go

import config

SR         = config.SAMPLE_RATE
HOP        = config.HOP_LENGTH
N_MELS     = config.N_MELS
N_MFCC     = config.N_MFCC
N_FFT      = config.N_FFT
DISP_STEP  = config.DISPLAY_STEP
SILENCE_DB = config.SILENCE_DB


# ══════════════════════════════════════════════════════════════════
# AUDIO LOADER  (Numba-free — soundfile + scipy only)
# ══════════════════════════════════════════════════════════════════
def _load(file_path: str, target_sr: int = SR):
    """Load audio, mix to mono, resample to target_sr."""
    data, orig_sr = sf.read(file_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if orig_sr != target_sr:
        n_target = int(len(data) * target_sr / orig_sr)
        data = scipy.signal.resample(data, n_target)
    return data.astype(np.float32), target_sr


# ══════════════════════════════════════════════════════════════════
# SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════════════
def _time_axis(n_frames: int) -> np.ndarray:
    return np.array([i * HOP / SR for i in range(n_frames)], dtype=np.float32)


def _mm_ss_ticks(t_end: float):
    interval = 10 if t_end <= 120 else (30 if t_end <= 600 else 60)
    vals  = list(range(0, int(t_end) + interval + 1, interval))
    texts = [f"{v//60}:{v%60:02d}" for v in vals]
    return vals, texts


def _base_layout(title: str, height: int = 320, y_title: str = "") -> dict:
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
# 1. MEL-SPECTROGRAM
# ══════════════════════════════════════════════════════════════════
def mel_spectrogram(file_path: str):
    y, sr = _load(file_path)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT, fmax=8000
    )
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Clip colour scale so one loud frame doesn't wash everything out
    vmin = float(np.percentile(S_db, 5))
    vmax = float(np.percentile(S_db, 98))

    n_frames  = S_db.shape[1]
    t         = _time_axis(n_frames)
    t_end     = float(t[-1]) if len(t) else 0.0
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=8000)

    fig = go.Figure(go.Heatmap(
        z=S_db, x=t, y=mel_freqs,
        colorscale=config.COLORSCALE_MEL,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="dB", tickfont=dict(size=9, color="#ccc")),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Power: %{z:.1f} dB<extra></extra>",
    ))
    layout = _base_layout("Mel-Spectrogram", height=340, y_title="Frequency (Hz)")
    layout["yaxis"]["type"] = "log"
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    return fig, dict(
        feature_vector      = [float(v) for v in np.mean(S_db, axis=1)],
        feature_metadata    = {"n_mels": N_MELS, "fmax": 8000,
                               "vmin_db": vmin, "vmax_db": vmax},
        feature_time_series = [float(v) for v in t[::DISP_STEP]],
    )


# ══════════════════════════════════════════════════════════════════
# 2. SPECTROGRAM (linear Hz — no mel filter bank)
# ══════════════════════════════════════════════════════════════════
def spectrogram(file_path: str):
    """
    Regular STFT-based spectrogram with a linear frequency axis.
    No mel filter bank — every bin is equally spaced in Hz.
    Shows up to 8 kHz (where most speech/music energy lives).
    """
    y, sr = _load(file_path)

    # STFT magnitude → dB
    D      = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    D_db   = librosa.amplitude_to_db(D, ref=np.max)

    # Clip colour scale for readability
    vmin = float(np.percentile(D_db, 5))
    vmax = float(np.percentile(D_db, 98))

    n_frames = D_db.shape[1]
    t        = _time_axis(n_frames)
    t_end    = float(t[-1]) if len(t) else 0.0

    # Frequency axis (only up to 8 kHz)
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freq_mask = freqs <= 8000
    freqs_show = freqs[freq_mask]
    D_show     = D_db[freq_mask, :]

    fig = go.Figure(go.Heatmap(
        z=D_show, x=t, y=freqs_show,
        colorscale=config.COLORSCALE_SPEC,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="dB", tickfont=dict(size=9, color="#ccc")),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Power: %{z:.1f} dB<extra></extra>",
    ))
    layout = _base_layout(
        "Spectrogram (Linear Hz — STFT magnitude)",
        height=340, y_title="Frequency (Hz)",
    )
    # Linear axis (NOT log — that is what distinguishes this from mel spectrogram)
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    mean_per_bin = np.mean(D_show, axis=1)
    dominant_hz  = float(freqs_show[np.argmax(mean_per_bin)])

    return fig, dict(
        feature_vector      = [float(v) for v in mean_per_bin[::max(1, len(mean_per_bin)//20)]],
        feature_metadata    = {
            "n_fft":       N_FFT,
            "freq_res_hz": round(sr / N_FFT, 2),
            "fmax_hz":     8000,
            "dominant_hz": round(dominant_hz, 1),
            "vmin_db":     round(vmin, 1),
            "vmax_db":     round(vmax, 1),
        },
        feature_time_series = [float(v) for v in t[::DISP_STEP]],
    )


# ══════════════════════════════════════════════════════════════════
# 3. MFCCs + Δ + ΔΔ
# ══════════════════════════════════════════════════════════════════
def mfcc(file_path: str):
    y, sr = _load(file_path)
    M      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP, n_fft=N_FFT)
    delta  = librosa.feature.delta(M)
    delta2 = librosa.feature.delta(M, order=2)
    stack  = np.vstack([M, delta, delta2])   # (39, frames)

    # Per-row z-score, clamped to ±3 so colour is readable
    mean = stack.mean(axis=1, keepdims=True)
    std  = stack.std(axis=1, keepdims=True) + 1e-9
    z    = np.clip((stack - mean) / std, -3, 3)

    n_frames = z.shape[1]
    t        = _time_axis(n_frames)
    t_end    = float(t[-1]) if len(t) else 0.0

    row_labels = (
            [f"MFCC {i}"  for i in range(N_MFCC)] +
            [f"Δ  {i}"    for i in range(N_MFCC)] +
            [f"ΔΔ {i}"   for i in range(N_MFCC)]
    )

    fig = go.Figure(go.Heatmap(
        z=z, x=t, y=row_labels,
        colorscale=config.COLORSCALE_MFCC,
        zmid=0, zmin=-3, zmax=3,
        colorbar=dict(title="z-score", tickfont=dict(size=9, color="#ccc")),
        hovertemplate="Time: %{x:.2f}s<br>%{y}<br>z: %{z:.2f}<extra></extra>",
    ))
    layout = _base_layout("MFCCs + Δ + ΔΔ  (z-score per row, clamped ±3)",
                          height=400, y_title="Coefficient")
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    return fig, dict(
        feature_vector      = [float(v) for v in M.mean(axis=1)],
        feature_metadata    = {"n_mfcc": N_MFCC, "stack_shape": list(stack.shape)},
        feature_time_series = [float(v) for v in t[::DISP_STEP]],
    )


# ══════════════════════════════════════════════════════════════════
# 4. SPECTRAL FLUX
# ══════════════════════════════════════════════════════════════════
def spectral_flux(file_path: str):
    y, sr = _load(file_path)
    S     = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))

    # L2 norm of frame-to-frame magnitude difference
    diff = np.diff(S, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])   # pad so length == n_frames

    n_frames = len(flux)
    t        = _time_axis(n_frames)
    t_end    = float(t[-1]) if len(t) else 0.0
    thresh   = float(np.percentile(flux, 95))

    spike_mask = flux > thresh
    spike_t    = t[spike_mask]
    spike_v    = flux[spike_mask]

    t_d    = t[::DISP_STEP]
    flux_d = flux[::DISP_STEP]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_d, y=flux_d, mode="lines", name="Spectral Flux",
        line=dict(color=config.COLOR_PRIMARY, width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=spike_t, y=spike_v, mode="markers", name="Spike (>95th pct)",
        marker=dict(color=config.COLOR_DANGER, size=5, symbol="circle"),
    ))
    fig.add_hline(
        y=thresh, line_dash="dot", line_color=config.COLOR_ACCENT,
        annotation_text=f"95th pct: {thresh:.2f}",
        annotation_font_color=config.COLOR_ACCENT,
    )
    layout = _base_layout("Spectral Flux  (L2 frame-to-frame spectral change)",
                          y_title="Flux")
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    return fig, dict(
        feature_vector      = [float(flux.mean()), float(flux.std()),
                               float(flux.max()), float(thresh)],
        feature_metadata    = {"threshold_95p": float(thresh),
                               "spike_count":   int(spike_mask.sum())},
        feature_time_series = [float(v) for v in flux_d],
    )


# ══════════════════════════════════════════════════════════════════
# 5. ONSET STRENGTH
# ══════════════════════════════════════════════════════════════════
def onset_strength(file_path: str):
    y, sr = _load(file_path)
    oenv  = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=oenv, sr=sr, hop_length=HOP, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)

    n_frames = len(oenv)
    t        = _time_axis(n_frames)
    t_end    = float(t[-1]) if len(t) else 0.0
    duration = max(t_end, 0.001)
    onset_rate = len(onset_times) / duration

    t_d    = t[::DISP_STEP]
    oenv_d = oenv[::DISP_STEP]

    # Vertical lines at each onset — merged with None separators
    ox, oy = [], []
    for ot in onset_times:
        ox.extend([float(ot), float(ot), None])
        oy.extend([0.0, float(oenv.max()), None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_d, y=oenv_d, mode="lines", name="Onset Strength",
        line=dict(color=config.COLOR_SECONDARY, width=1.2),
        fill="tozeroy", fillcolor="rgba(124,106,247,0.12)",
    ))
    if ox:
        fig.add_trace(go.Scatter(
            x=ox, y=oy, mode="lines", name="Detected onset",
            line=dict(color=config.COLOR_ACCENT, width=1.0, dash="dot"),
            hoverinfo="skip",
        ))
    layout = _base_layout(
        f"Onset Strength  —  {len(onset_times)} onsets  "
        f"({onset_rate:.2f}/s)",
        y_title="Strength",
    )
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    return fig, dict(
        feature_vector      = [float(oenv.mean()), float(oenv.std()),
                               float(oenv.max()), float(onset_rate)],
        feature_metadata    = {"onset_times_s":     [float(v) for v in onset_times],
                               "onset_rate_per_sec": float(onset_rate)},
        feature_time_series = [float(v) for v in oenv_d],
    )


# ══════════════════════════════════════════════════════════════════
# 6. RMS ENERGY
# ══════════════════════════════════════════════════════════════════
def rms_energy(file_path: str):
    y, sr = _load(file_path)
    rms   = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-9)

    n_frames      = len(rms_db)
    t             = _time_axis(n_frames)
    t_end         = float(t[-1]) if len(t) else 0.0
    silence_mask  = rms_db < SILENCE_DB
    silence_ratio = float(silence_mask.mean())
    valid         = rms_db[~silence_mask]
    dynamic_range = float(rms_db.max() - valid.min()) if len(valid) else 0.0

    # Build silence region fills (merged with None separators)
    sil_x, sil_y = [], []
    y_lo = float(rms_db.min()) - 5
    y_hi = float(rms_db.max()) + 5
    in_sil, start_s = False, 0.0
    for i, (sil, tv) in enumerate(zip(silence_mask, t)):
        if sil and not in_sil:
            in_sil, start_s = True, float(tv)
        elif not sil and in_sil:
            in_sil = False
            sil_x.extend([start_s, start_s, float(t[i-1]), float(t[i-1]), None])
            sil_y.extend([y_lo, y_hi, y_hi, y_lo, None])
    if in_sil:
        sil_x.extend([start_s, start_s, t_end, t_end, None])
        sil_y.extend([y_lo, y_hi, y_hi, y_lo, None])

    t_d    = t[::DISP_STEP]
    rms_d  = rms_db[::DISP_STEP]

    fig = go.Figure()
    if sil_x:
        fig.add_trace(go.Scatter(
            x=sil_x, y=sil_y, fill="toself",
            fillcolor="rgba(255,75,75,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Silence region", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=t_d, y=rms_d, mode="lines", name="RMS (dB)",
        line=dict(color=config.COLOR_PRIMARY, width=1.3),
    ))
    fig.add_hline(
        y=SILENCE_DB, line_dash="dot", line_color=config.COLOR_DANGER,
        annotation_text=f"Silence threshold ({SILENCE_DB} dB)",
        annotation_font_color=config.COLOR_DANGER,
    )
    layout = _base_layout(
        f"RMS Energy  —  silence: {silence_ratio*100:.1f}%  |  "
        f"dyn. range: {dynamic_range:.1f} dB",
        y_title="dB",
    )
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    return fig, dict(
        feature_vector      = [float(rms_db.mean()), float(rms_db.std()),
                               float(rms_db.max()), float(silence_ratio)],
        feature_metadata    = {"silence_ratio":    float(silence_ratio),
                               "dynamic_range_db": float(dynamic_range)},
        feature_time_series = [float(v) for v in rms_d],
    )


# ══════════════════════════════════════════════════════════════════
# 7. SPECTRAL CONTRAST
# ══════════════════════════════════════════════════════════════════
def spectral_contrast(file_path: str):
    y, sr    = _load(file_path)
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=6, hop_length=HOP
    )   # shape: (7, frames)

    vmin = float(np.percentile(contrast, 2))
    vmax = float(np.percentile(contrast, 98))

    n_frames = contrast.shape[1]
    t        = _time_axis(n_frames)
    t_end    = float(t[-1]) if len(t) else 0.0

    band_labels = [
        "Sub-bass (< 200 Hz)",
        "Bass (200–500 Hz)",
        "Low-mid (500–1k Hz)",
        "Mid (1k–2k Hz)",
        "High-mid (2k–4k Hz)",
        "Presence (4k–6k Hz)",
        "Brilliance (> 6k Hz)",
    ]

    fig = go.Figure(go.Heatmap(
        z=contrast, x=t, y=band_labels,
        colorscale=config.COLORSCALE_CONTRAST,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title="Contrast (dB)", tickfont=dict(size=9, color="#ccc")),
        hovertemplate="Time: %{x:.2f}s<br>Band: %{y}<br>Contrast: %{z:.1f} dB<extra></extra>",
    ))
    layout = _base_layout("Spectral Contrast  (peak-to-valley dB per frequency band)",
                          height=340, y_title="Frequency Band")
    fig.update_layout(**layout)
    _set_time_ticks(fig, t_end)

    return fig, dict(
        feature_vector      = [float(v) for v in contrast.mean(axis=1)],
        feature_metadata    = {"n_bands": 7,
                               "mean_per_band": [float(v) for v in contrast.mean(axis=1)]},
        feature_time_series = [float(v) for v in t[::DISP_STEP]],
    )


# ══════════════════════════════════════════════════════════════════
# PUBLIC DISPATCH
# ══════════════════════════════════════════════════════════════════
_DISPATCH = {
    "mel_spectrogram":   mel_spectrogram,
    "spectrogram":       spectrogram,
    "mfcc":              mfcc,
    "spectral_flux":     spectral_flux,
    "onset_strength":    onset_strength,
    "rms_energy":        rms_energy,
    "spectral_contrast": spectral_contrast,
}


def compute(key: str, file_path: str):
    """
    Single entry point called by gui.py.
    Returns (plotly_fig, data_dict).
    """
    fn = _DISPATCH.get(key)
    if fn is None:
        raise ValueError(f"Unknown feature key '{key}'. Valid keys: {list(_DISPATCH)}")
    return fn(file_path)