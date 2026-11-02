"""
config.py — Single source of truth for all constants, colours, and registries.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(BASE_DIR, "assets", "examples")

# ── DSP constants ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050
HOP_LENGTH   = 512
N_MELS       = 128
N_MFCC       = 13
N_FFT        = 2048
SILENCE_DB   = -40      # threshold below which RMS frames are considered silent
DISPLAY_STEP = 2        # downsample factor for time-series plots (speed)

# ── Dark oscilloscope theme ───────────────────────────────────────────────────
COLOR_BG       = "#0e1117"
COLOR_TEXT     = "#e0e0e0"
COLOR_PRIMARY  = "#00c8a0"   # teal phosphor
COLOR_SECONDARY= "#7c6af7"   # soft purple
COLOR_ACCENT   = "#f7c948"   # amber
COLOR_DANGER   = "#ff4b4b"   # red
COLOR_GRID     = "#222222"

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLOR_BG,
    font=dict(color=COLOR_TEXT, size=11),
    margin=dict(l=60, r=20, t=50, b=50),
)

# ── Colorscales ───────────────────────────────────────────────────────────────
COLORSCALE_MEL      = "inferno"
COLORSCALE_MFCC     = "RdBu_r"
COLORSCALE_CONTRAST = "Viridis"
COLORSCALE_SPEC     = "Magma"     # for regular (linear) spectrogram
COLORSCALE_WAVELET  = "Plasma"    # for wavelet scalogram

# ── Example clips ─────────────────────────────────────────────────────────────
# key → (display label, filename, short description)
EXAMPLE_CLIPS = {
    "sine_440hz":   ("Sine 440 Hz",     "sine_440hz.wav",   "Pure tone – single frequency"),
    "white_noise":  ("White Noise",      "white_noise.wav",  "All frequencies equally – flat spectrum"),
    "clap_impact":  ("Clap / Impact",    "clap_impact.wav",  "Transient burst + low-frequency body thump"),
    "speech_like":  ("Speech-like",      "speech_like.wav",  "Formant resonances + glottal pulses + fricatives"),
    "music_chord":  ("Music Chord (C)",  "music_chord.wav",  "C-major triad with vibrato"),
    "silence":      ("Silence",          "silence.wav",      "Near-silence noise floor"),
    "chirp_sweep":  ("Chirp Sweep",      "chirp_sweep.wav",  "Log sweep 100 → 4 000 Hz"),
    "drumbeat":     ("Drumbeat",         "drumbeat.wav",     "Kick + snare pattern at 120 BPM"),
    "piano_note":   ("Piano Note (A4)",  "piano_note.wav",   "Struck string with inharmonic partials"),
}

# ── Feature registry (Tab 2 – Feature Extraction) ────────────────────────────
FEATURE_REGISTRY = {
    "mel_spectrogram": {
        "label":       "Mel Spectrogram",
        "description": "Time-frequency energy map on the perceptual mel scale.",
        "week":        "Week 2",
    },
    "spectrogram": {
        "label":       "Spectrogram (Linear)",
        "description": "STFT magnitude on a linear Hz frequency axis — no mel warping.",
        "week":        "Week 2",
    },
    "onset_strength": {
        "label":       "Onset Strength",
        "description": "How strongly new sounds begin at each moment.",
        "week":        "Week 2",
    },
    "spectral_flux": {
        "label":       "Spectral Flux",
        "description": "Rate of change in the frequency spectrum – highlights onsets.",
        "week":        "Week 2",
    },
    "rms_energy": {
        "label":       "RMS Energy",
        "description": "Loudness over time (root-mean-square amplitude in dB).",
        "week":        "Week 2",
    },
    "mfcc": {
        "label":       "MFCC (13 + Δ + ΔΔ)",
        "description": "Mel-frequency cepstral coefficients – compact timbral fingerprint.",
        "week":        "Week 2",
    },
    "spectral_contrast": {
        "label":       "Spectral Contrast",
        "description": "Peak-vs-valley energy ratio across 7 frequency sub-bands.",
        "week":        "Week 2",
    },
}

# ── Traditional-technique registry (Tab 4) ────────────────────────────────────
TRADITIONAL_REGISTRY = {
    "zcr": {
        "label":       "Zero Crossing Rate",
        "description": "How often the waveform crosses zero — separates tonal from noisy.",
        "week":        "Week 4",
    },
    "chroma": {
        "label":       "Chromagram",
        "description": "Energy in each of the 12 pitch classes (C, C#, D … B) over time.",
        "week":        "Week 4",
    },
    "autocorrelation": {
        "label":       "Autocorrelation",
        "description": "Periodicity and pitch estimation on a single audio frame.",
        "week":        "Week 4",
    },
    "fft_frame": {
        "label":       "FFT Spectrum",
        "description": "Full magnitude spectrum at a chosen time instant.",
        "week":        "Week 4",
    },
    "wavelet": {
        "label":       "Wavelet Transform",
        "description": "Time-frequency scalogram using the continuous wavelet transform (CWT).",
        "week":        "Week 4",
    },
}

# ── DTW is shown only in Tab 5 (comparison), not in Tab 4 selectbox ──────────
# It compares two files, so it lives outside TRADITIONAL_REGISTRY.
DTW_LABEL = "DTW Alignment (MFCC)"

# ── Long educational explanations (shown in expanders) ───────────────────────
FEATURE_EXPLANATIONS = {
    "mel_spectrogram": """
**What it shows**
A 2-D heat map where the horizontal axis is time, the vertical axis is mel-scaled frequency,
and colour represents energy (dB). The mel scale compresses high frequencies and expands low
ones, matching how the human ear perceives pitch.

**How to read it**
Bright horizontal bands = sustained pitched tones.
Scattered bright patches = transients (drums, consonants).
Dark regions = silence or absence of that frequency.

**Formula (key steps)**
1. Short-time Fourier transform (STFT) → power spectrum |STFT|²
2. Multiply each frame by a mel filter bank (triangular overlapping filters)
3. Convert to dB: `S_dB = 10 log₁₀(S + ε)`

**Why it matters**
It is the most common front-end for speech recognition and music classification neural networks.
""",

    "spectrogram": """
**What it shows**
A 2-D heat map identical in layout to the Mel Spectrogram, but with a **linear Hz** y-axis
instead of the mel-warped one. Every pixel in height represents an equal number of Hz.

**Difference from Mel Spectrogram**
The mel scale stretches the low-frequency region and squashes the high-frequency region
to match human hearing. The linear spectrogram shows each frequency bin at equal physical
spacing — more useful when you care about exact Hz values (e.g. engineering, acoustics).

**Formula**
1. Short-time Fourier transform (STFT) → magnitude |STFT|
2. Convert to dB: `S_dB = 20 log₁₀(|STFT| + ε)`
Display is clipped to 0–8 kHz (most speech/music energy lives here).

**How to read it**
A pure sine at 440 Hz shows as a bright horizontal line at exactly 440 Hz.
Harmonics appear as equally-spaced parallel lines above it.
""",

    "mfcc": """
**What it shows**
13 cepstral coefficients (MFCCs) stacked with their first (Δ) and second (ΔΔ) derivatives —
39 rows total — plotted as a heat map over time.

**How to read it**
Row 0 (C0) encodes overall energy. Rows 1-12 encode the spectral shape (timbre).
Δ rows capture how quickly coefficients change; ΔΔ rows capture acceleration.

**Formula**
1. Mel spectrogram (log energy in each mel band)
2. Discrete cosine transform (DCT) → first 13 coefficients
3. Δ = finite difference of each row across time

**Why it matters**
MFCCs were the standard feature for speech recognition for decades and are still widely used
as compact timbral descriptors.
""",

    "onset_strength": """
**What it shows**
A single curve showing how strongly a new sound begins at each moment in time, with detected
onset positions marked as vertical lines.

**How to read it**
Tall peaks = abrupt starts (drum hits, plucked strings, hard consonants).
Low baseline = sustained or gradual sounds.

**Formula**
`onset_env[t] = mean( max(0, S_dB[t] − S_dB[t-1]) )` over all mel bands.

**Why it matters**
Onset detection drives beat tracking, tempo estimation, and note segmentation.
""",

    "spectral_flux": """
**What it shows**
The L2 norm of the frame-to-frame difference of the STFT magnitude, plotted over time.
The dashed line marks the 95th-percentile spike threshold; circles highlight detected spikes.

**Formula**
`flux[t] = ‖ |STFT[t]| − |STFT[t-1]| ‖₂`

**Why it matters**
Spectral flux is a simple but effective novelty function used in onset detection and
source separation — high flux means the spectrum changed a lot.
""",

    "rms_energy": """
**What it shows**
Loudness (in dB) over time. Shaded regions below the silence threshold show where the audio
is essentially silent.

**Formula**
`RMS[t] = √( mean(x²) )` over each analysis frame, then `dB = 20 log₁₀(RMS + ε)`.

**Why it matters**
RMS energy is the simplest loudness descriptor. It is used for voice activity detection,
dynamic range compression, and normalisation.
""",

    "spectral_contrast": """
**What it shows**
For each of 7 frequency sub-bands, the difference (in dB) between the peaks and valleys of the
spectrum. Plotted as a heat map over time.

**Formula**
Split the spectrum into sub-bands, take the top α% as peaks and bottom α% as valleys,
then `contrast = mean(peaks) − mean(valleys)`.

**Why it matters**
High contrast = harmonic tones (music). Low contrast = noise. It captures the distinction
between harmonic and percussive content more robustly than a single spectral moment.
""",
}

TRADITIONAL_EXPLANATIONS = {
    "zcr": """
**What it shows**
How many times per sample the waveform crosses zero. The dotted line shows the mean ZCR.

**Formula**
`ZCR[t] = (1/N) Σ |sign(x[n]) − sign(x[n-1])| / 2`

**Interpretation**
High ZCR ≈ noise or unvoiced speech (lots of sign changes).
Low ZCR ≈ tonal / voiced content (slow, smooth oscillations).
""",

    "chroma": """
**What it shows**
The energy in each of the 12 pitch classes (C, C#, D, … B) over time, computed from the
constant-Q transform (CQT). Bright cells = that pitch class is active.

**Why it matters**
Chroma is the foundation of chord recognition and key detection. Because it folds all octaves
together, it is also octave-invariant.
""",

    "autocorrelation": """
**What it shows**
The normalised autocorrelation function (ACF) of a 50 ms audio frame centred at the chosen
time. The lag axis runs from ~1 ms (1 000 Hz) to ~20 ms (50 Hz).

**Formula**
`ACF[τ] = Σ x[n] · x[n+τ]`, computed efficiently via `IFFT(|FFT(x)|²)`.

**Pitch estimation**
The first large positive peak in the ACF at lag τ₀ corresponds to the fundamental period:
`f₀ = SR / τ₀`.

**Why it matters**
Autocorrelation-based pitch detection (YIN, AMDF) is one of the oldest and most reliable
methods for monophonic pitch tracking.
""",

    "fft_frame": """
**What it shows**
The magnitude spectrum (in dB, peak-normalised to 0 dB) of a single audio frame at the
chosen time instant. The top 5 spectral peaks are annotated with their frequency in Hz.
Display is limited to 0–8 kHz where most musical and speech energy lives.

**Formula**
1. Extract N_FFT samples centred at the target time.
2. Apply Hann window to reduce spectral leakage.
3. `spectrum = |FFT(windowed_frame)|`, convert to dB.

**Why it matters**
A single-frame FFT gives an instantaneous snapshot of frequency content — useful for
comparing individual notes, vowel formants, or transient spectra.
""",

    "wavelet": """
**What it shows**
A **scalogram** — the magnitude of the Continuous Wavelet Transform (CWT) plotted as a
heat map. The horizontal axis is time; the vertical axis is **scale** (inversely related to
frequency). Bright patches = high wavelet energy at that time and scale.

**Wavelet families available**
| Name | Best for |
|------|----------|
| Ricker (Mexican hat) | Sharp transients, peaks |
| Morlet | Oscillations, pitched tones (scipy only) |
| db4 (Daubechies 4) | General-purpose smooth signals |
| haar | Step changes, sharp edges |

**Compared to the STFT Spectrogram**
The STFT uses a fixed window length — good time resolution OR good frequency resolution,
not both. Wavelets use **short windows at high scales** (for fast transients) and
**long windows at low scales** (for slow oscillations), adapting automatically.
This is the **Heisenberg uncertainty principle** made practical.

**Formula (Ricker / Mexican hat)**
`ψ(t) = (1 − t²/σ²) · exp(−t²/2σ²)`
`CWT[scale, t] = ∫ x(τ) · ψ((τ−t)/scale) dτ`

**Why it matters**
Wavelets are used in audio compression (MPEG-4), ECG analysis, seismology, and anywhere
you need to detect both fast transients and slow rhythms in the same signal.
""",
}