"""
gui.py — All Plotly / Streamlit rendering components.
=======================================================
This module contains ONLY display logic.
All DSP lives in features.py, audio_utils.py, and traditional.py.

Lazy imports avoid loading heavy libs at Streamlit startup.
"""

import io
import wave
import numpy as np
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

SR   = config.SAMPLE_RATE
HOP  = config.HOP_LENGTH
WAVEFORM_MAX_POINTS = 4000   # downsample for browser performance


# ── Lazy module loaders ───────────────────────────────────────────────────────
def _features():
    import features as _f
    return _f

def _audio_utils():
    import audio_utils as _a
    return _a

def _traditional():
    import traditional as _t
    return _t


# ── Unique chart key counter (prevents Streamlit duplicate-ID error) ──────────
_chart_counter = [0]

def _next_key(prefix: str = "chart") -> str:
    """Return a session-unique key string for st.plotly_chart(key=...)."""
    _chart_counter[0] += 1
    return f"{prefix}_{_chart_counter[0]}"


# ── Internal helpers ──────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def _mm_ss_ticks(t_end: float):
    interval = 10 if t_end <= 120 else (30 if t_end <= 600 else 60)
    vals  = list(range(0, int(t_end) + interval + 1, interval))
    texts = [f"{v//60}:{v%60:02d}" for v in vals]
    return vals, texts


def _stat_pills(items: dict, _key: str = ""):
    """
    Render stat pills with a 3-line cap and expand/collapse button.
    Skips keys that hold raw arrays (feature_vector / feature_time_series).
    """
    # Filter out raw array values — only show human-readable scalars/strings
    _SKIP_KEYS = {"feature_vector", "feature_time_series", "feature_metadata"}
    display = {}
    for k, v in items.items():
        if k in _SKIP_KEYS:
            continue
        # If it's a dict itself (e.g. feature_metadata), flatten one level
        if isinstance(v, dict):
            for kk, vv in v.items():
                if not hasattr(vv, '__len__') or isinstance(vv, str):
                    display[kk] = vv
        elif hasattr(v, '__len__') and not isinstance(v, str) and len(v) > 10:
            continue   # skip long array-like values
        else:
            display[k] = v

    if not display:
        return

    pill_html = "".join(
        f'<div class="stat-pill">{k}: <span>{v}</span></div>'
        for k, v in display.items()
    )

    # Always show all pills — use a Streamlit expander for long stat sets
    n = len(display)
    THRESHOLD = 6   # show first 6 pills inline; rest inside expander

    if n <= THRESHOLD:
        st.markdown(f'<div class="stat-row">{pill_html}</div>',
                    unsafe_allow_html=True)
    else:
        # Split: first THRESHOLD pills always visible, rest in expander
        items_list  = list(display.items())
        visible     = items_list[:THRESHOLD]
        hidden      = items_list[THRESHOLD:]

        visible_html = "".join(
            f'<div class="stat-pill">{k}: <span>{v}</span></div>'
            for k, v in visible
        )
        hidden_html = "".join(
            f'<div class="stat-pill">{k}: <span>{v}</span></div>'
            for k, v in hidden
        )

        st.markdown(f'<div class="stat-row">{visible_html}</div>',
                    unsafe_allow_html=True)
        with st.expander(f"Show {len(hidden)} more stats"):
            st.markdown(f'<div class="stat-row">{hidden_html}</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WAVEFORM
# ══════════════════════════════════════════════════════════════════════════════
def render_waveform(file_path: str, title: str = "Waveform",
                    color: str = config.COLOR_PRIMARY):
    """Plot a downsampled waveform with MM:SS x-axis."""
    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    n = len(y)
    t = np.linspace(0, n / sr, n, endpoint=False)
    t_end = float(t[-1])

    # Downsample
    step = max(1, n // WAVEFORM_MAX_POINTS)
    t_d  = t[::step]
    y_d  = y[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_d, y=y_d,
        mode="lines", name="Waveform",
        line=dict(color=color, width=0.8),
        fill="tozeroy",
        fillcolor=_rgba(color, 0.08),
        hovertemplate="Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>",
    ))

    vals, texts = _mm_ss_ticks(t_end)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=config.COLOR_TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=60, r=20, t=50, b=50),
        height=220,
        xaxis=dict(
            title="Time (mm:ss)",
            showgrid=True, gridcolor=config.COLOR_GRID,
            tickmode="array", tickvals=vals, ticktext=texts,
            range=[0, min(60, t_end)],
            rangeslider=dict(visible=True, thickness=0.06, bgcolor="#1a1c23"),
            tickfont=dict(size=10, color="#ccc"),
        ),
        yaxis=dict(
            title="Amplitude",
            showgrid=True, gridcolor=config.COLOR_GRID,
            tickfont=dict(size=10, color="#ccc"),
            range=[-1.05, 1.05],
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=_next_key("waveform"))


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL STATS
# ══════════════════════════════════════════════════════════════════════════════
def render_signal_stats(file_path: str):
    """Render a row of stat pills for the loaded audio."""
    y, sr = sf.read(file_path, always_2d=False)
    channels = y.shape[1] if y.ndim > 1 else 1
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    n = len(y)

    rms_db  = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)
    peak_db = 20 * np.log10(np.max(np.abs(y)) + 1e-9)

    _stat_pills({
        "SR":       f"{sr:,} Hz",
        "Duration": f"{n/sr:.2f}s",
        "Samples":  f"{n:,}",
        "Channels": str(channels),
        "RMS":      f"{rms_db:.1f} dB",
        "Peak":     f"{peak_db:.1f} dB",
    })


# ══════════════════════════════════════════════════════════════════════════════
# SINE GENERATOR (Tab 1 playground)
# ══════════════════════════════════════════════════════════════════════════════
def render_sine_generator(freq: float, duration: float, amplitude: float):
    """Render a sine wave plot + playable audio widget."""
    sr     = 22050
    n      = int(sr * duration)
    t      = np.linspace(0, duration, n, endpoint=False)
    y      = amplitude * np.sin(2 * np.pi * freq * t)

    # Show only the first 3 periods (but at least 5 ms) for readability
    period_s   = 1.0 / freq
    show_dur   = max(0.005, min(3 * period_s, duration))
    mask       = t <= show_dur
    t_ms       = t[mask] * 1000   # display in ms

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_ms, y=y[mask],
        mode="lines", name="Sine",
        line=dict(color=config.COLOR_ACCENT, width=1.5),
        fill="tozeroy",
        fillcolor=_rgba(config.COLOR_ACCENT, 0.07),
        hovertemplate="t: %{x:.2f} ms<br>A: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Sine wave — {freq} Hz, A={amplitude:.2f}",
            font=dict(size=13, color=config.COLOR_TEXT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=60, r=20, t=50, b=50),
        height=200,
        xaxis=dict(title="Time (ms)", showgrid=True, gridcolor=config.COLOR_GRID,
                   tickfont=dict(size=10, color="#ccc")),
        yaxis=dict(title="Amplitude", showgrid=True, gridcolor=config.COLOR_GRID,
                   tickfont=dict(size=10, color="#ccc"), range=[-1.05, 1.05]),
    )
    st.plotly_chart(fig, use_container_width=True, key=_next_key("sine"))

    # Encode as WAV bytes for st.audio
    peak = np.max(np.abs(y)) + 1e-9
    pcm  = (np.clip(y / peak, -1, 1) * 32767).astype(np.int16)
    buf  = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    st.audio(buf.getvalue(), format="audio/wav")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GRID (Tab 2)
# ══════════════════════════════════════════════════════════════════════════════
def render_feature_grid(selected_keys: list, file_path: str):
    """Render each selected feature in a 2-column grid with explanation expander."""
    feat = _features()
    cols = st.columns(2)

    for i, key in enumerate(selected_keys):
        col = cols[i % 2]
        info = config.FEATURE_REGISTRY[key]
        with col:
            st.markdown(f"#### {info['label']}")
            try:
                fig, stats = feat.compute(key, file_path)
                st.plotly_chart(fig, use_container_width=True,
                                key=_next_key(f"feat_{key}"))
                _stat_pills(stats)
            except Exception as e:
                st.error(f"Error computing {info['label']}: {e}")

            explanation = config.FEATURE_EXPLANATIONS.get(key, "")
            if explanation:
                with st.expander("📖 What is this?"):
                    st.markdown(explanation)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING DEMOS (Tab 3)
# ══════════════════════════════════════════════════════════════════════════════
def render_resampling_demo(file_path: str):
    """Show the same audio at 4 different sample rates with playback."""
    au = _audio_utils()
    target_rates = [8000, 11025, 22050, 44100]
    cols = st.columns(len(target_rates))
    for col, tsr in zip(cols, target_rates):
        with col:
            st.markdown(f"**{tsr:,} Hz**")
            st.caption(f"Nyquist: {tsr//2:,} Hz")
            try:
                y, sr = au.resample_to(file_path, tsr)
                wav_bytes = au.to_wav_bytes(y, sr)
                st.audio(wav_bytes, format="audio/wav")
                duration = len(y) / sr
                _stat_pills({"Duration": f"{duration:.2f}s", "Samples": f"{len(y):,}"})
            except Exception as e:
                st.error(str(e))


def render_normalisation_demo(file_path: str):
    """Plot raw / peak / RMS variants side by side with audio."""
    au = _audio_utils()
    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    variants = au.normalisation_variants(y)
    labels   = {"raw": "Raw (unchanged)", "peak": "Peak normalised", "rms": "RMS −20 dBFS"}
    colors   = {"raw": config.COLOR_TEXT, "peak": config.COLOR_PRIMARY, "rms": config.COLOR_SECONDARY}
    cols     = st.columns(3)

    n = len(y)
    t = np.linspace(0, n / sr, n, endpoint=False)
    step = max(1, n // WAVEFORM_MAX_POINTS)
    t_d  = t[::step]

    for col, (vkey, label) in zip(cols, labels.items()):
        with col:
            st.markdown(f"**{label}**")
            y_v  = variants[vkey]
            y_d  = y_v[::step]
            rms  = float(np.sqrt(np.mean(y_v**2)))
            peak = float(np.max(np.abs(y_v)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_d, y=y_d,
                mode="lines", line=dict(color=colors[vkey], width=0.8),
                fill="tozeroy", fillcolor=_rgba(colors[vkey], 0.08),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=config.COLOR_BG,
                margin=dict(l=40, r=10, t=20, b=30),
                height=160,
                xaxis=dict(showgrid=True, gridcolor=config.COLOR_GRID,
                           tickfont=dict(size=9, color="#888")),
                yaxis=dict(showgrid=True, gridcolor=config.COLOR_GRID,
                           tickfont=dict(size=9, color="#888"), range=[-1.05, 1.05]),
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=_next_key(f"norm_{vkey}"))
            _stat_pills({"Peak": f"{peak:.3f}", "RMS": f"{rms:.4f}"})
            wav_bytes = au.to_wav_bytes(y_v, sr)
            st.audio(wav_bytes, format="audio/wav")


def render_noise_filter_demo(
        file_path: str, snr_db: float, filter_type: str, cutoff
):
    """Overlay clean + noisy + filtered waveforms and provide two audio players."""
    au = _audio_utils()
    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    y_noisy    = au.add_noise(y, snr_db)
    y_filtered = au.apply_filter(y_noisy, sr, filter_type, cutoff)

    n    = len(y)
    t    = np.linspace(0, n / sr, n, endpoint=False)
    step = max(1, n // WAVEFORM_MAX_POINTS)
    t_d  = t[::step]

    fig = go.Figure()
    for y_arr, name, color, width in [
        (y,          "Original",  config.COLOR_TEXT,      0.8),
        (y_noisy,    "Noisy",     config.COLOR_DANGER,    0.7),
        (y_filtered, "Filtered",  config.COLOR_PRIMARY,   1.2),
    ]:
        fig.add_trace(go.Scatter(
            x=t_d, y=y_arr[::step],
            mode="lines", name=name,
            line=dict(color=color, width=width),
            opacity=0.85,
        ))

    fig.update_layout(
        title=dict(
            text=f"SNR {snr_db} dB → {filter_type} @ {cutoff} Hz",
            font=dict(size=13, color=config.COLOR_TEXT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=60, r=20, t=50, b=50),
        height=260,
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor=config.COLOR_GRID,
                   tickfont=dict(size=10, color="#ccc")),
        yaxis=dict(title="Amplitude", showgrid=True, gridcolor=config.COLOR_GRID,
                   tickfont=dict(size=10, color="#ccc"), range=[-1.05, 1.05]),
    )
    st.plotly_chart(fig, use_container_width=True, key=_next_key("noise_filter"))

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Noisy audio")
        st.audio(au.to_wav_bytes(y_noisy, sr), format="audio/wav")
    with col2:
        st.caption("Filtered audio")
        st.audio(au.to_wav_bytes(y_filtered, sr), format="audio/wav")


def render_windowing_demo(file_path: str, window_type: str):
    """Show window shape + STFT magnitude spectrum side by side."""
    au = _audio_utils()
    shape_t, shape_y, freqs, mag = au.windowing_analysis(file_path, window_type)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{window_type.capitalize()} window shape",
                        "Magnitude spectrum (first energetic frame)"],
    )

    # Window shape
    fig.add_trace(go.Scatter(
        x=shape_t, y=shape_y,
        mode="lines", name="Window",
        line=dict(color=config.COLOR_ACCENT, width=1.5),
        fill="tozeroy", fillcolor=_rgba(config.COLOR_ACCENT, 0.08),
    ), row=1, col=1)

    # Magnitude (Hz up to 8 kHz for readability)
    mask      = freqs <= 8000
    fig.add_trace(go.Scatter(
        x=freqs[mask] / 1000, y=mag[mask],
        mode="lines", name="Magnitude",
        line=dict(color=config.COLOR_PRIMARY, width=1.2),
        fill="tozeroy", fillcolor=_rgba(config.COLOR_PRIMARY, 0.08),
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=config.COLOR_BG,
        font=dict(color=config.COLOR_TEXT, size=11),
        margin=dict(l=60, r=20, t=60, b=50),
        height=300,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=config.COLOR_GRID,
                     tickfont=dict(size=10, color="#ccc"))
    fig.update_yaxes(showgrid=True, gridcolor=config.COLOR_GRID,
                     tickfont=dict(size=10, color="#ccc"))
    fig.update_xaxes(title_text="Sample index",  row=1, col=1)
    fig.update_xaxes(title_text="Frequency (kHz)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude",       row=1, col=1)
    fig.update_yaxes(title_text="dB (peak-norm.)", row=1, col=2, range=[-80, 5])

    st.plotly_chart(fig, use_container_width=True,
                    key=_next_key(f"window_{window_type}"))

    # Sidelobe comparison hint
    sidelobe_info = {
        "rectangular": "No taper — lowest freq resolution, highest sidelobe leakage",
        "hann":        "Good all-rounder — cosine taper, −31.5 dB first sidelobe",
        "hamming":     "Similar to Hann, slightly better sidelobe (−42.7 dB)",
        "blackman":    "Best sidelobe suppression (−58 dB), widest main lobe",
    }
    st.markdown(
        f'<div class="info-card">💡 <b>{window_type.capitalize()}</b>: '
        f'{sidelobe_info.get(window_type, "")}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRADITIONAL FEATURE (Tab 4)
# ══════════════════════════════════════════════════════════════════════════════
def render_traditional_feature(file_path: str, key: str, **kwargs):
    """Render one traditional technique figure + stats + explanation."""
    trad = _traditional()
    info = config.TRADITIONAL_REGISTRY[key]

    try:
        fig, stats = trad.compute(key, file_path, **kwargs)
        st.plotly_chart(fig, use_container_width=True,
                        key=_next_key(f"trad_{key}"))
        _stat_pills(stats)
    except Exception as e:
        st.error(f"Error computing {info['label']}: {e}")
        return

    explanation = config.TRADITIONAL_EXPLANATIONS.get(key, "")
    if explanation:
        with st.expander("📖 What is this?"):
            st.markdown(explanation)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON (Tab 5)
# ══════════════════════════════════════════════════════════════════════════════
def render_comparison(
        path_a: str, name_a: str,
        path_b: str, name_b: str,
        feature_key: str,
):
    """Render the same feature for two files side by side."""
    is_traditional = feature_key in config.TRADITIONAL_REGISTRY
    col_a, col_b   = st.columns(2)

    for col, path, name, side in [
        (col_a, path_a, name_a, "a"),
        (col_b, path_b, name_b, "b"),
    ]:
        with col:
            st.markdown(f"**{name}**")
            try:
                if is_traditional:
                    trad = _traditional()
                    fig, stats = trad.compute(feature_key, path)
                else:
                    feat = _features()
                    fig, stats = feat.compute(feature_key, path)
                st.plotly_chart(fig, use_container_width=True,
                                key=_next_key(f"cmp_{feature_key}_{side}"))
                _stat_pills(stats)
            except Exception as e:
                st.error(str(e))


def render_dtw(
        path_a: str, name_a: str,
        path_b: str, name_b: str,
):
    """
    Render the DTW alignment between two files.
    Shows the cost matrix, warping path, and MFCC heatmaps for both files.
    """
    trad = _traditional()
    try:
        fig, stats = trad.compute_dtw(path_a, path_b, name_a, name_b)
        st.plotly_chart(fig, use_container_width=True, key=_next_key("dtw"))
        _stat_pills(stats)

        # Educational explanation
        with st.expander("📖 What is DTW?"):
            st.markdown("""
**Dynamic Time Warping (DTW)** measures the similarity between two sequences that
may vary in speed. Unlike a simple frame-by-frame comparison, DTW finds the
**optimal alignment** by warping the time axis.

**How it works**
1. Compute MFCC feature vectors for both audio files.
2. Build a **cost matrix** — each cell `[i, j]` holds the Euclidean distance between
   frame `i` of file A and frame `j` of file B.
3. Use dynamic programming to find the **lowest-cost path** through the matrix
   from the top-left to the bottom-right corner.
4. The total cost along this path is the **DTW distance** — lower = more similar.

**Reading the plot**
- The **cost matrix** (heatmap) shows the distance between every pair of frames —
  dark cells = similar frames, bright cells = different frames.
- The **teal diagonal line** is the optimal warping path. A perfectly diagonal path
  means the files are aligned at the same speed. Curves mean one file was faster
  or slower at that moment.

**Why it matters**
DTW is used in speech recognition, gesture recognition, and music alignment.
It handles the same spoken word at different speeds, or the same melody played
at slightly different tempos.
""")
    except Exception as e:
        st.error(f"DTW alignment failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SANDBOX MODES (Tab 6)
# ══════════════════════════════════════════════════════════════════════════════
def render_full_profile(file_path: str):
    """Render all acoustic features then all traditional features."""
    st.markdown("### Acoustic features")
    render_feature_grid(list(config.FEATURE_REGISTRY.keys()), file_path)

    st.markdown("---")
    st.markdown("### Traditional techniques")
    trad = _traditional()
    for key, info in config.TRADITIONAL_REGISTRY.items():
        st.markdown(f"#### {info['label']}")
        try:
            fig, stats = trad.compute(key, file_path)
            st.plotly_chart(fig, use_container_width=True,
                            key=_next_key(f"profile_trad_{key}"))
            _stat_pills(stats)
        except Exception as e:
            st.error(str(e))


def render_stacked_features(file_path: str, key_top: str, key_bot: str):
    """Show two acoustic features stacked vertically."""
    feat = _features()
    for key in (key_top, key_bot):
        info = config.FEATURE_REGISTRY[key]
        st.markdown(f"#### {info['label']}")
        try:
            fig, stats = feat.compute(key, file_path)
            st.plotly_chart(fig, use_container_width=True,
                            key=_next_key(f"stack_{key}"))
            _stat_pills(stats)
        except Exception as e:
            st.error(str(e))


def render_feature_download(file_path: str):
    """Compute all features and offer a CSV download."""
    import csv, io as _io

    feat = _features()
    rows = []

    with st.spinner("Computing all features…"):
        for key, info in config.FEATURE_REGISTRY.items():
            try:
                _, stats = feat.compute(key, file_path)
                for stat_k, stat_v in stats.items():
                    rows.append({"feature": info["label"], "metric": stat_k, "value": stat_v})
            except Exception as e:
                rows.append({"feature": info["label"], "metric": "ERROR", "value": str(e)})

    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["feature", "metric", "value"])
    writer.writeheader()
    writer.writerows(rows)

    st.download_button(
        "⬇️ Download feature statistics CSV",
        data=buf.getvalue().encode(),
        file_name="audio_features.csv",
        mime="text/csv",
    )

    # Preview table
    st.markdown("**Preview**")
    col_f, col_m, col_v = st.columns([2, 2, 2])
    col_f.markdown("**Feature**")
    col_m.markdown("**Metric**")
    col_v.markdown("**Value**")
    for row in rows:
        col_f.write(row["feature"])
        col_m.write(row["metric"])
        col_v.write(row["value"])