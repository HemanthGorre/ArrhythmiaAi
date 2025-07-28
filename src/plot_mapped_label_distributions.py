import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import WIN_SIZE

import numpy as np
import matplotlib.pyplot as plt
import json
import collections

DATA_PROCESSED = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("results", "eda")
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_distribution(label_file, map_file, title, suffix):
    labels = np.load(os.path.join(DATA_PROCESSED, label_file))
    with open(os.path.join(DATA_PROCESSED, map_file)) as f:
        label_map = json.load(f)
    # Map int ID to class name
    if all(isinstance(v, int) for v in label_map.values()):
        inv_map = {v: k for k, v in label_map.items()}
    else:
        inv_map = label_map
    cnts = collections.Counter(labels)
    xs = list(range(len(inv_map)))
    ys = [cnts[x] for x in xs]
    names = [inv_map[x] for x in xs]
    plt.figure(figsize=(12,5))
    plt.bar(xs, ys, tick_label=names)
    plt.title(f"{title} (WIN_SIZE={WIN_SIZE})")
    plt.xlabel("Class")
    plt.ylabel("Count")
    for i, v in enumerate(ys):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    out_file = os.path.join(RESULTS_DIR, f"label_distribution_{suffix}.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")

# ---- For 6-class ----
for split in ['train', 'val', 'test']:
    plot_distribution(
        f"{split}_labels_6class.npy",
        "labels_6class.json",
        f"6-class label distribution ({split})",
        f"6class_{split}"
    )

# ---- For 5-class ----
for split in ['train', 'val', 'test']:
    plot_distribution(
        f"{split}_labels_5class.npy",
        "labels_5class.json",
        f"5-class label distribution ({split})",
        f"5class_{split}"
    )

# ---- Optionally, aggregate/all splits together ----
# plot_distribution("all_labels_6class.npy", "labels_6class.json", "6-class label distribution (all)", "6class_all")
# plot_distribution("all_labels_5class.npy", "labels_5class.json", "5-class label distribution (all)", "5class_all")
