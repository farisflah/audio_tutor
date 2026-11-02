# 🎧 Audio Tutor

An interactive teaching tool for audio signal analysis, built for Saxion University of Applied Sciences.  
Covers waveforms, feature extraction, preprocessing, wavelets, and DTW comparison — all in a visual, hands-on interface.

---

## How to run the app

**1. Download the project**

Go to the GitHub page → click the green **Code** button → click **Download ZIP** → extract the folder anywhere on your computer.

**2. Install Python** *(skip if you already have it)*

Download and install Python from [python.org/downloads](https://www.python.org/downloads/) — pick the latest version.  
⚠️ During installation, check the box **"Add Python to PATH"** before clicking Install.

**3. Open a terminal inside the project folder**

- On **Windows**: open the extracted folder → click the address bar at the top → type `powershell` → press Enter
- On **Mac**: right-click the folder → "New Terminal at Folder"

**4. Install dependencies** *(only needed once)*
```bash
pip install -r requirements.txt
```

**5. Start the app**
```bash
streamlit run main.py
```

The app opens automatically in your browser. Done.

---

## Usage

- Upload any WAV, MP3, FLAC, or OGG file using the sidebar — or pick one of the built-in example clips
- Navigate the tabs to explore different analysis techniques
- Use the **Comparison** tab to compare two audio files side by side with DTW alignment

---

*Built by Faris Flah · Ambient Intelligence Research Group · Saxion University of Applied Sciences*