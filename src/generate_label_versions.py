import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_PROCESSED = os.path.join("data", "processed")
main_classes = ['N', 'L', 'R', 'V', 'A']

with open(os.path.join(DATA_PROCESSED, "labels_unfiltered.json")) as f:
    orig_label_map = json.load(f)
main_label_ids = [orig_label_map[c] for c in main_classes if c in orig_label_map]

for split in ["train", "val", "test"]:
    # 6-class
    y = np.load(os.path.join(DATA_PROCESSED, f"{split}_labels_unfiltered.npy"))
    def map6(y):
        arr = []
        for lbl in y:
            if lbl in main_label_ids:
                arr.append(main_label_ids.index(lbl))
            else:
                arr.append(5)  # "other"
        return np.array(arr)
    y6 = map6(y)
    np.save(os.path.join(DATA_PROCESSED, f"{split}_labels_6class.npy"), y6)
    X = np.load(os.path.join(DATA_PROCESSED, f"{split}_signals_unfiltered.npy"))
    np.save(os.path.join(DATA_PROCESSED, f"{split}_signals_6class.npy"), X)

    # 5-class
    mask5 = np.isin(y, main_label_ids)
    y5 = np.array([main_label_ids.index(lbl) for lbl in y[mask5]])
    X5 = X[mask5]
    np.save(os.path.join(DATA_PROCESSED, f"{split}_labels_5class.npy"), y5)
    np.save(os.path.join(DATA_PROCESSED, f"{split}_signals_5class.npy"), X5)

with open(os.path.join(DATA_PROCESSED, "labels_6class.json"), "w") as f:
    json.dump({c: i for i, c in enumerate(main_classes)} | {"other": 5}, f)
with open(os.path.join(DATA_PROCESSED, "labels_5class.json"), "w") as f:
    json.dump({c: i for i, c in enumerate(main_classes)}, f)
print("Label mapping completed (unfiltered, 6-class, 5-class).")
