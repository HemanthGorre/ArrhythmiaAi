import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from config import WIN_SIZE, STEP, FS, BANDPASS_LOW, BANDPASS_HIGH, MITBIH_PATH

def bandpass_filter(signal, fs, lowcut, highcut, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def get_label_for_window(start_idx, ann, win_size=WIN_SIZE):
    beats = [(i, s) for i, s in enumerate(ann.sample) if start_idx <= s < start_idx + win_size]
    if beats:
        beat_idx = beats[0][0]
        return ann.symbol[beat_idx]
    return 'N'

def mitbih_stream(record_name, mitbih_path=MITBIH_PATH, win_size=WIN_SIZE, step=STEP):
    rec_path = os.path.join(mitbih_path, record_name)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    ecg = record.p_signal[:, 0]
    ecg_filt = bandpass_filter(ecg, FS, BANDPASS_LOW, BANDPASS_HIGH)
    for start in range(0, len(ecg_filt) - win_size, step):
        window = ecg_filt[start : start + win_size]
        label = get_label_for_window(start, ann, win_size)
        timestamp = start / FS
        yield {
            'signal': window,
            'timestamp': timestamp,
            'label': label
        }
