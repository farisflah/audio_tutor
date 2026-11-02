"""
audio_utils.py — Audio loading, resampling, normalisation, noise, filtering, windowing
=======================================================================================
All DSP helpers for the Preprocessing tab (Tab 3).
No Plotly figures live here — this module returns numpy arrays only.
gui.py builds the figures from the arrays.

Public API:
    load_raw(file_path)                      → (y_float32, sr)
    resample_to(file_path, target_sr)        → (y, sr)
    to_wav_bytes(y, sr)                      → bytes  (in-memory WAV for st.audio)
    normalisation_variants(y)                → dict {key: y_array}
    add_noise(y, snr_db)                     → y_noisy
    apply_filter(y, sr, ftype, cutoff)       → y_filtered
    windowing_analysis(file_path, win_type)  → (shape_t, shape_y, freqs, magnitude)
"""

import io
import wave

import numpy as np
import soundfile as sf
import scipy.signal
import librosa

import config


# ══════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════
def load_raw(file_path: str):
    """
    Load audio at its native sample rate, mixed to mono.
    Returns (y float32, sr).
    """
    data, sr = sf.read(file_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def resample_to(file_path: str, target_sr: int):
    """
    Load audio and resample to target_sr using scipy (Numba-free).
    Returns (y float32, target_sr).
    """
    data, orig_sr = sf.read(file_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if orig_sr != target_sr:
        n_target = int(len(data) * target_sr / orig_sr)
        data = scipy.signal.resample(data, n_target).astype(np.float32)
    return data, target_sr


# ══════════════════════════════════════════════════════════════════
# WAV BYTES ENCODER  (for st.audio inline playback)
# ══════════════════════════════════════════════════════════════════
def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """
    Encode a float32 numpy array as a WAV file in memory.
    Returns raw bytes that Streamlit's st.audio() can play directly.
    """
    # Peak-normalise to avoid clipping artefacts
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════
def normalisation_variants(y: np.ndarray) -> dict:
    """
    Return three normalised versions of a waveform as a dict:
        "raw"   → unchanged
        "peak"  → peak normalised to ±1.0
        "rms"   → RMS normalised to −20 dB target
    """
    y = y.astype(np.float32)

    # Peak normalisation
    peak = np.max(np.abs(y))
    y_peak = y / (peak + 1e-9)

    # RMS normalisation — target −20 dBFS
    target_rms_linear = 10 ** (-20 / 20)
    current_rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    y_rms = np.clip(y * (target_rms_linear / current_rms), -1.0, 1.0)

    return {
        "raw":  y.copy(),
        "peak": y_peak.astype(np.float32),
        "rms":  y_rms.astype(np.float32),
    }


# ══════════════════════════════════════════════════════════════════
# NOISE
# ══════════════════════════════════════════════════════════════════
def add_noise(y: np.ndarray, snr_db: float, rng_seed: int = 42) -> np.ndarray:
    """
    Add white Gaussian noise at the specified SNR (dB).
    SNR = 10 * log10(P_signal / P_noise)
    → noise_power = signal_power / 10^(SNR/10)
    """
    y = y.astype(np.float32)
    signal_power = float(np.mean(y ** 2)) + 1e-12
    noise_power  = signal_power / (10 ** (snr_db / 10))
    rng   = np.random.default_rng(rng_seed)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    # Scale noise to the desired power
    noise_actual_power = float(np.mean(noise ** 2)) + 1e-12
    noise = noise * np.sqrt(noise_power / noise_actual_power)
    return np.clip(y + noise, -1.0, 1.0)


# ══════════════════════════════════════════════════════════════════
# FILTERING
# ══════════════════════════════════════════════════════════════════
def apply_filter(
        y: np.ndarray,
        sr: int,
        filter_type: str,
        cutoff,
        order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth filter to the signal.

    filter_type : "lowpass" | "highpass" | "bandpass"
    cutoff      : single Hz for low/highpass, (low_hz, high_hz) for bandpass
    order       : filter order (4 is standard)
    """
    nyq = sr / 2.0

    if filter_type == "bandpass":
        low_hz, high_hz = cutoff
        low  = max(20, min(low_hz,  nyq * 0.95)) / nyq
        high = max(low + 0.01, min(high_hz, nyq * 0.99)) / nyq
        b, a = scipy.signal.butter(order, [low, high], btype="band")
    elif filter_type == "lowpass":
        Wn   = max(20, min(float(cutoff), nyq * 0.99)) / nyq
        b, a = scipy.signal.butter(order, Wn, btype="low")
    elif filter_type == "highpass":
        Wn   = max(20, min(float(cutoff), nyq * 0.99)) / nyq
        b, a = scipy.signal.butter(order, Wn, btype="high")
    else:
        raise ValueError(f"Unknown filter type: '{filter_type}'")

    return scipy.signal.filtfilt(b, a, y.astype(np.float32)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# WINDOWING
# ══════════════════════════════════════════════════════════════════
def windowing_analysis(
        file_path: str,
        window_type: str,
        n_fft: int = config.N_FFT,
):
    """
    Returns:
        shape_t   : x values for window shape plot (sample indices 0..n_fft-1)
        shape_y   : window amplitude values
        stft_freqs: frequency axis in Hz for the spectrum plot
        stft_mag  : magnitude spectrum of the first non-silent frame using this window
    """
    # Load
    data, orig_sr = sf.read(file_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # Resample to standard SR if needed
    sr = config.SAMPLE_RATE
    if orig_sr != sr:
        n_target = int(len(data) * sr / orig_sr)
        data = scipy.signal.resample(data, n_target).astype(np.float32)

    # Build window
    win_map = {
        "hann":        np.hanning(n_fft),
        "hamming":     np.hamming(n_fft),
        "blackman":    np.blackman(n_fft),
        "rectangular": np.ones(n_fft),
    }
    window = win_map.get(window_type, np.hanning(n_fft)).astype(np.float32)

    # Window shape
    shape_t = np.arange(n_fft)
    shape_y = window

    # Find first frame with enough energy so plot is interesting
    hop    = config.HOP_LENGTH
    frame  = None
    for start in range(0, max(1, len(data) - n_fft), hop):
        chunk = data[start : start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        if np.sqrt(np.mean(chunk ** 2)) > 0.001:
            frame = chunk
            break
    if frame is None:
        frame = data[:n_fft] if len(data) >= n_fft else np.pad(data, (0, n_fft - len(data)))

    # Apply window and compute magnitude spectrum
    windowed  = frame * window
    spectrum  = np.abs(np.fft.rfft(windowed))
    # Convert to dB, referenced to max
    spectrum_db = 20 * np.log10(spectrum + 1e-9)
    spectrum_db -= spectrum_db.max()   # range 0 dB .. -inf

    # Frequency axis
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    return (
        shape_t.astype(np.float32),
        shape_y.astype(np.float32),
        freqs.astype(np.float32),
        spectrum_db.astype(np.float32),
    )