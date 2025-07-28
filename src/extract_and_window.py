import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import WIN_SIZE, STEP, FS, BANDPASS_LOW, BANDPASS_HIGH, MITBIH_PATH

import wfdb
import numpy as np
import json
from tqdm import tqdm
from scipy.signal import butter, filtfilt

def bandpass_filter(ecg, low=BANDPASS_LOW, high=BANDPASS_HIGH, fs=FS, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, ecg)

DATA_PROCESSED = os.path.join("data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)
RECORDS = [f[:3] for f in os.listdir(MITBIH_PATH) if f.endswith('.dat')]

all_windows, all_labels = [], []
label_map = {}
label_cnt = 0

for rec in tqdm(RECORDS):
    record = wfdb.rdrecord(os.path.join(MITBIH_PATH, rec))
    annotation = wfdb.rdann(os.path.join(MITBIH_PATH, rec), 'atr')
    signal = record.p_signal[:,0]  # Lead II
    signal = bandpass_filter(signal)  # Bandpass using config
    ann_idx = annotation.sample
    ann_lbl = annotation.symbol

    for s in ann_lbl:
        if s not in label_map:
            label_map[s] = label_cnt
            label_cnt += 1

    for start in range(0, len(signal) - WIN_SIZE, STEP):
        end = start + WIN_SIZE
        labels_in_window = [s for i, s in zip(ann_idx, ann_lbl) if start <= i < end]
        if labels_in_window:
            main_lbl = max(set(labels_in_window), key=labels_in_window.count)
            all_windows.append(signal[start:end])
            all_labels.append(label_map[main_lbl])

np.save(os.path.join(DATA_PROCESSED, "all_signals.npy"), np.array(all_windows))
np.save(os.path.join(DATA_PROCESSED, "all_labels.npy"), np.array(all_labels))
with open(os.path.join(DATA_PROCESSED, "labels.json"), "w") as f:
    json.dump(label_map, f, indent=2)
