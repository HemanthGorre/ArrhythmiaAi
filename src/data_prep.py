import os, sys, glob, json
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_stream import mitbih_stream
from config import WIN_SIZE

RECORDS_DIR = 'data/mitbih/mit-bih-arrhythmia-database-1.0.0'
OUT_DIR = 'data/processed'

def can_stratify(labels):
    c = Counter(labels)
    return min(c.values()) >= 2

def auto_label_map(records):
    label_set = set()
    for rec in records:
        for packet in mitbih_stream(rec):
            label_set.add(packet['label'])
    label_list = sorted(list(label_set))
    label_map = {lbl: i for i, lbl in enumerate(label_list)}
    return label_map

def extract_all_windows(records, label_map):
    signals, labels = [], []
    for rec in records:
        for packet in mitbih_stream(rec):
            signals.append(packet['signal'].copy())
            labels.append(label_map[packet['label']])
    return np.stack(signals), np.array(labels)

def main(test_size=0.15, val_size=0.15, random_state=42):
    os.makedirs(OUT_DIR, exist_ok=True)
    rec_files = glob.glob(os.path.join(RECORDS_DIR, '*.dat'))
    records = sorted([os.path.splitext(os.path.basename(f))[0] for f in rec_files])
    print(f"Discovered {len(records)} records.")

    # --- UNFILTERED LABEL MAP
    label_map = auto_label_map(records)
    print(f"Label map: {label_map}")
    with open(os.path.join(OUT_DIR, "labels_unfiltered.json"), "w") as f:
        json.dump(label_map, f)

    signals, labels = extract_all_windows(records, label_map)
    print(f"Total windows: {signals.shape[0]}, window size: {signals.shape[1]}")
    # Save full unfiltered for label mapping
    np.save(os.path.join(OUT_DIR, "all_signals.npy"), signals)
    np.save(os.path.join(OUT_DIR, "all_labels.npy"), labels)

    strat = labels if can_stratify(labels) else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        signals, labels, test_size=val_size+test_size, stratify=strat, random_state=random_state)
    strat2 = y_tmp if can_stratify(y_tmp) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_size/(val_size+test_size), stratify=strat2, random_state=random_state)
    for split, X, y in zip(['train','val','test'], [X_train, X_val, X_test], [y_train, y_val, y_test]):
        np.save(os.path.join(OUT_DIR, f"{split}_signals_unfiltered.npy"), X)
        np.save(os.path.join(OUT_DIR, f"{split}_labels_unfiltered.npy"), y)
        print(f"{split}: {X.shape[0]} samples")
    print("Unfiltered split done.")

if __name__ == "__main__":
    main()
