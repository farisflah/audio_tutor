"""
main.py — Audio Analysis Tutor
================================
Streamlit entry point. Run with:
    streamlit run main.py

Tabs:
    1  Introduction & Waveform
    2  Feature Extraction (7 features incl. linear Spectrogram)
    3  Preprocessing  (resample · normalise · noise · filter · window)
    4  Traditional Techniques (ZCR · Chroma · Autocorr · FFT · Wavelet)
    5  Comparison  (two files side-by-side + DTW alignment)
    6  Sandbox  (full profile + CSV download)
"""

import os
import tempfile

import streamlit as st

import config

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Audio Analysis Tutor",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* dark sidebar */
[data-testid="stSidebar"] { background-color: #13151e; }
/* tab bar */
.stTabs [data-baseweb="tab-list"]  { gap: 4px; background: #13151e; padding: 4px 4px 0; border-radius: 8px 8px 0 0; }
.stTabs [data-baseweb="tab"]       { background: #1e2030; color: #888; border-radius: 6px 6px 0 0; padding: 6px 16px; font-size: 0.82rem; }
.stTabs [aria-selected="true"]     { background: #0e1117 !important; color: #00c8a0 !important; border-bottom: 2px solid #00c8a0; }
/* info cards */
.info-card { background:#13151e; border-left:3px solid #00c8a0; padding:12px 16px; border-radius:4px; margin-bottom:12px; font-size:0.88rem; color:#ccc; }
/* stat pills */
.stat-row  { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }
.stat-pill { background:#1e2030; border:1px solid #2a2d3e; border-radius:20px; padding:4px 12px; font-size:0.78rem; color:#aaa; white-space:nowrap; }
.stat-pill span { color:#00c8a0; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        # Primary audio file
        "audio_bytes": None,
        "audio_name":  None,
        "audio_path":  None,
        # Comparison: slot A and B
        "cmp_bytes_a": None,  "cmp_name_a": None,  "cmp_path_a": None,
        "cmp_bytes_b": None,  "cmp_name_b": None,  "cmp_path_b": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _save_tmp(audio_bytes: bytes, name: str) -> str:
    """Write bytes to a temp file and return its path (needed by librosa)."""
    suffix = os.path.splitext(name)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="audiotool_")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def _render_sidebar():
    with st.sidebar:
        st.markdown("## 🎧 Audio Tutor")
        st.markdown("---")

        # ── File upload ──────────────────────────────────────────────────────
        st.markdown("### Upload audio")
        uploaded = st.file_uploader(
            "WAV / MP3 / FLAC / OGG",
            type=["wav", "mp3", "flac", "ogg"],
            key="file_uploader",
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state["audio_bytes"] = uploaded.read()
            st.session_state["audio_name"]  = uploaded.name
            st.session_state["audio_path"]  = _save_tmp(
                st.session_state["audio_bytes"], uploaded.name
            )

        # ── Example clips ────────────────────────────────────────────────────
        st.markdown("### Built-in examples")
        for key, (label, fname, desc) in config.EXAMPLE_CLIPS.items():
            if st.button(label, key=f"ex_{key}", use_container_width=True):
                path = os.path.join(config.EXAMPLES_DIR, fname)
                with open(path, "rb") as f:
                    data = f.read()
                st.session_state["audio_bytes"] = data
                st.session_state["audio_name"]  = fname
                st.session_state["audio_path"]  = path
            st.caption(desc)

        # ── Current file indicator ───────────────────────────────────────────
        st.markdown("---")
        name = st.session_state.get("audio_name")
        if name:
            st.success(f"✅ {name}")
        else:
            st.info("No audio loaded yet.")

        # ── Copyright ────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="
                margin-top: 2.5rem;
                padding-top: 0.6rem;
                border-top: 1px solid #1e1e2e;
                text-align: center;
                font-size: 0.68rem;
                color: #555;
                letter-spacing: 0.03em;
            ">
                © 2026 <span style="color:#00c8a0;">Faris Flah</span><br>
                Audio Tutor · All rights reserved
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INTRODUCTION & WAVEFORM
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_intro():
    st.markdown("## Week 1 — Introduction to Audio Analysis")

    col_info, col_wave = st.columns([1, 2])

    with col_info:
        st.markdown("""
<div class="info-card">
<b>What is audio?</b><br>
A continuous pressure wave sampled at discrete intervals. The
<b>sample rate</b> (e.g. 22 050 Hz) sets the highest frequency that can be
represented (<i>Nyquist = SR / 2</i>).
</div>
<div class="info-card">
<b>Key vocabulary</b><br>
• <b>Sample</b> — one amplitude measurement<br>
• <b>Frame</b> — short window of samples (e.g. 2 048)<br>
• <b>Hop</b> — step between frames (e.g. 512 = 75 % overlap)<br>
• <b>dB</b> — logarithmic amplitude scale
</div>
""", unsafe_allow_html=True)

        st.markdown("### Sine wave playground")
        freq  = st.slider("Frequency (Hz)", 50, 2000, 440, step=10)
        amp   = st.slider("Amplitude", 0.1, 1.0, 0.8, step=0.05)
        dur   = st.slider("Duration (s)", 0.1, 3.0, 1.0, step=0.1)

    with col_wave:
        import gui
        if st.session_state["audio_path"]:
            st.markdown("#### Loaded waveform")
            gui.render_waveform(
                st.session_state["audio_path"],
                title=st.session_state["audio_name"],
            )
            gui.render_signal_stats(st.session_state["audio_path"])
            st.audio(st.session_state["audio_bytes"], format="audio/wav")
        else:
            st.info("Load a file from the sidebar to see its waveform here.")

        st.markdown("#### Sine generator")
        gui.render_sine_generator(freq, dur, amp)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_features():
    st.markdown("## Week 2 — Acoustic Feature Extraction")

    if not st.session_state["audio_path"]:
        st.warning("Load an audio file from the sidebar first.")
        return

    all_keys   = list(config.FEATURE_REGISTRY.keys())
    all_labels = [config.FEATURE_REGISTRY[k]["label"] for k in all_keys]
    selected_labels = st.multiselect(
        "Select features to compute",
        options=all_labels,
        default=all_labels[:3],
    )
    selected_keys = [
        k for k, info in config.FEATURE_REGISTRY.items()
        if info["label"] in selected_labels
    ]

    if not selected_keys:
        st.info("Select at least one feature above.")
        return

    import gui
    gui.render_feature_grid(selected_keys, st.session_state["audio_path"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_preprocessing():
    st.markdown("## Week 3 — Preprocessing")

    if not st.session_state["audio_path"]:
        st.warning("Load an audio file from the sidebar first.")
        return

    import gui

    section = st.radio(
        "Choose a preprocessing demo",
        ["Resampling", "Normalisation", "Noise & Filtering", "Windowing"],
        horizontal=True,
    )

    st.markdown("---")

    if section == "Resampling":
        st.markdown("### Resampling — how sample rate affects quality")
        st.markdown("""
<div class="info-card">
Resampling converts audio between different sample rates.
Lower rates reduce file size but lose high-frequency content.
The <b>Nyquist theorem</b> states the highest reproducible frequency = SR / 2.
</div>""", unsafe_allow_html=True)
        gui.render_resampling_demo(st.session_state["audio_path"])

    elif section == "Normalisation":
        st.markdown("### Normalisation — scaling amplitude consistently")
        st.markdown("""
<div class="info-card">
<b>Peak normalisation</b> scales so the loudest sample = ±1.0 (0 dBFS).<br>
<b>RMS normalisation</b> scales so the average power = −20 dBFS — better for consistent loudness across clips.
</div>""", unsafe_allow_html=True)
        gui.render_normalisation_demo(st.session_state["audio_path"])

    elif section == "Noise & Filtering":
        st.markdown("### Noise & Filtering")
        col1, col2 = st.columns(2)
        with col1:
            snr = st.slider("SNR (dB) — lower = more noise", 5, 40, 20)
            ftype = st.selectbox("Filter type", ["lowpass", "highpass", "bandpass"])
        with col2:
            if ftype == "bandpass":
                low_hz  = st.slider("Low cutoff (Hz)",  20, 4000, 300)
                high_hz = st.slider("High cutoff (Hz)", 500, 10000, 3000)
                cutoff  = (low_hz, high_hz)
            else:
                cutoff = st.slider("Cutoff frequency (Hz)", 200, 8000, 1000)
        gui.render_noise_filter_demo(
            st.session_state["audio_path"], snr, ftype, cutoff
        )

    elif section == "Windowing":
        st.markdown("### Windowing — reducing spectral leakage")
        st.markdown("""
<div class="info-card">
When we cut a frame of audio, the sharp edges create <b>spectral leakage</b>.
A <b>window function</b> tapers the frame to zero at both ends, reducing leakage at the cost of slightly broadening peaks.
</div>""", unsafe_allow_html=True)
        wtype = st.selectbox(
            "Window type", ["hann", "hamming", "blackman", "rectangular"]
        )
        gui.render_windowing_demo(st.session_state["audio_path"], wtype)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADITIONAL TECHNIQUES
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_traditional():
    st.markdown("## Week 4 — Classical Signal Processing")

    if not st.session_state["audio_path"]:
        st.warning("Load an audio file from the sidebar first.")
        return

    import gui

    keys   = list(config.TRADITIONAL_REGISTRY.keys())
    labels = [config.TRADITIONAL_REGISTRY[k]["label"] for k in keys]
    sel    = st.selectbox("Choose a technique", labels)
    key    = keys[labels.index(sel)]

    kwargs = {}
    if key in ("autocorrelation", "fft_frame"):
        import soundfile as _sf
        y_tmp, sr_tmp = _sf.read(st.session_state["audio_path"], always_2d=False)
        dur_tmp = len(y_tmp) / sr_tmp
        t_default = min(0.5, dur_tmp * 0.3)
        kwargs["frame_time"] = st.slider(
            "Frame time (s)",
            min_value=0.0,
            max_value=float(dur_tmp),
            value=float(t_default),
            step=0.05,
        )
    elif key == "wavelet":
        kwargs["wavelet_family"] = st.selectbox(
            "Wavelet family",
            ["ricker", "morlet", "db4", "haar"],
            index=0,
            help="ricker / morlet use scipy (no extra install). db4 / haar require PyWavelets.",
        )

    gui.render_traditional_feature(st.session_state["audio_path"], key, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_comparison():
    st.markdown("## Week 5 — Compare Two Audio Files")

    import gui

    col_a, col_b = st.columns(2)

    # ── Slot A ───────────────────────────────────────────────────────────────
    with col_a:
        st.markdown("#### File A")
        up_a = st.file_uploader("Upload File A", type=["wav","mp3","flac","ogg"], key="cmp_up_a")
        if up_a:
            st.session_state["cmp_bytes_a"] = up_a.read()
            st.session_state["cmp_name_a"]  = up_a.name
            st.session_state["cmp_path_a"]  = _save_tmp(
                st.session_state["cmp_bytes_a"], up_a.name
            )
        # Quick-load from examples
        ex_a = st.selectbox(
            "…or pick an example",
            ["(none)"] + [v[0] for v in config.EXAMPLE_CLIPS.values()],
            key="cmp_ex_a",
            )
        if ex_a != "(none)":
            key_a = [k for k, v in config.EXAMPLE_CLIPS.items() if v[0] == ex_a][0]
            path_a = os.path.join(config.EXAMPLES_DIR, config.EXAMPLE_CLIPS[key_a][1])
            with open(path_a, "rb") as f:
                st.session_state["cmp_bytes_a"] = f.read()
            st.session_state["cmp_name_a"] = config.EXAMPLE_CLIPS[key_a][1]
            st.session_state["cmp_path_a"] = path_a

        if st.session_state["cmp_path_a"]:
            st.success(f"A: {st.session_state['cmp_name_a']}")
            st.audio(st.session_state["cmp_bytes_a"], format="audio/wav")

    # ── Slot B ───────────────────────────────────────────────────────────────
    with col_b:
        st.markdown("#### File B")
        up_b = st.file_uploader("Upload File B", type=["wav","mp3","flac","ogg"], key="cmp_up_b")
        if up_b:
            st.session_state["cmp_bytes_b"] = up_b.read()
            st.session_state["cmp_name_b"]  = up_b.name
            st.session_state["cmp_path_b"]  = _save_tmp(
                st.session_state["cmp_bytes_b"], up_b.name
            )
        ex_b = st.selectbox(
            "…or pick an example",
            ["(none)"] + [v[0] for v in config.EXAMPLE_CLIPS.values()],
            key="cmp_ex_b",
            )
        if ex_b != "(none)":
            key_b = [k for k, v in config.EXAMPLE_CLIPS.items() if v[0] == ex_b][0]
            path_b = os.path.join(config.EXAMPLES_DIR, config.EXAMPLE_CLIPS[key_b][1])
            with open(path_b, "rb") as f:
                st.session_state["cmp_bytes_b"] = f.read()
            st.session_state["cmp_name_b"] = config.EXAMPLE_CLIPS[key_b][1]
            st.session_state["cmp_path_b"] = path_b

        if st.session_state["cmp_path_b"]:
            st.success(f"B: {st.session_state['cmp_name_b']}")
            st.audio(st.session_state["cmp_bytes_b"], format="audio/wav")

    # ── Comparison ───────────────────────────────────────────────────────────
    st.markdown("---")
    if not (st.session_state["cmp_path_a"] and st.session_state["cmp_path_b"]):
        st.info("Load both File A and File B to enable comparison.")
        return

    # Build feature list: all acoustic + traditional features + DTW option
    all_features = {
        **{k: v["label"] for k, v in config.FEATURE_REGISTRY.items()},
        **{k: v["label"] for k, v in config.TRADITIONAL_REGISTRY.items()},
        "dtw": config.DTW_LABEL,   # ← new DTW option
    }

    chosen_label = st.selectbox("Feature / method to compare", list(all_features.values()))
    chosen_key   = [k for k, v in all_features.items() if v == chosen_label][0]

    name_a = st.session_state["cmp_name_a"]
    name_b = st.session_state["cmp_name_b"]

    if chosen_key == "dtw":
        # DTW compares both files together — special renderer
        st.markdown("""
<div class="info-card">
<b>DTW Alignment</b> — measures how similar two audio clips are by finding the optimal
time-warped alignment of their MFCC feature sequences. Lower DTW distance = more similar.
</div>""", unsafe_allow_html=True)
        gui.render_dtw(
            st.session_state["cmp_path_a"], name_a,
            st.session_state["cmp_path_b"], name_b,
        )
    else:
        gui.render_comparison(
            st.session_state["cmp_path_a"], name_a,
            st.session_state["cmp_path_b"], name_b,
            chosen_key,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SANDBOX (full profile + CSV)
# ══════════════════════════════════════════════════════════════════════════════
def render_tab_sandbox():
    st.markdown("## Sandbox — Full Profile & Export")

    if not st.session_state["audio_path"]:
        st.warning("Load an audio file from the sidebar first.")
        return

    import gui

    mode = st.radio(
        "Mode",
        ["Full profile (all features)", "Stacked pair", "Download feature CSV"],
        horizontal=True,
    )

    st.markdown("---")

    if mode == "Full profile (all features)":
        gui.render_full_profile(st.session_state["audio_path"])

    elif mode == "Stacked pair":
        all_keys   = list(config.FEATURE_REGISTRY.keys())
        all_labels = [config.FEATURE_REGISTRY[k]["label"] for k in all_keys]
        col1, col2 = st.columns(2)
        with col1:
            top_label = st.selectbox("Top feature",    all_labels, index=0)
        with col2:
            bot_label = st.selectbox("Bottom feature", all_labels, index=1)
        top_key = all_keys[all_labels.index(top_label)]
        bot_key = all_keys[all_labels.index(bot_label)]
        gui.render_stacked_features(st.session_state["audio_path"], top_key, bot_key)

    elif mode == "Download feature CSV":
        gui.render_feature_download(st.session_state["audio_path"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _render_sidebar()

    tabs = st.tabs([
        "🎵 Intro & Waveform",
        "📊 Feature Extraction",
        "🔧 Preprocessing",
        "📐 Traditional",
        "⚖️ Comparison",
        "🧪 Sandbox",
    ])

    with tabs[0]: render_tab_intro()
    with tabs[1]: render_tab_features()
    with tabs[2]: render_tab_preprocessing()
    with tabs[3]: render_tab_traditional()
    with tabs[4]: render_tab_comparison()
    with tabs[5]: render_tab_sandbox()

    # ── Footer copyright ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #1e1e2e;
            text-align: center;
            font-size: 0.72rem;
            color: #444;
            letter-spacing: 0.04em;
        ">
            © 2026 &nbsp;<span style="color:#00c8a0; font-weight:600;">Faris Flah</span>
            &nbsp;·&nbsp; Audio Tutor &nbsp;·&nbsp; Built for Saxion University of Applied Sciences
            &nbsp;·&nbsp; All rights reserved
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()