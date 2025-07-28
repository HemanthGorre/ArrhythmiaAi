# src/config.py

import os

# Sampling rate (Hz)
FS = 360

# Streaming window
WIN_SIZE = 720   # 2 seconds
STEP = 72        # Slide by 0.2 seconds

# Bandpass filter (Hz)
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45

# Path to nested MIT-BIH data folder (relative to project root)
MITBIH_PATH = os.path.join("data", "mitbih", "mit-bih-arrhythmia-database-1.0.0")