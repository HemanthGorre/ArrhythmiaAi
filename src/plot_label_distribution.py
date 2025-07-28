import os, sys, json, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DATA_PROCESSED = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("results", "eda")
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_dist(split, suffix, mapfile, title):
    y = np.load(os.path.join(DATA_PROCESSED, f"{split}_labels{suffix}.npy"))
    with open(os.path.join(DATA_PROCESSED, mapfile)) as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}
    xs = list(range(len(inv_map)))
    ys = [np.sum(y==x) for x in xs]
    names = [inv_map[x] for x in xs]
    plt.figure(figsize=(10,5))
    plt.bar(xs, ys, tick_label=names)
    plt.title(f"{title} ({split})")
    plt.xlabel("Class"); plt.ylabel("Count")
    for i, v in enumerate(ys):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"label_dist_{split}{suffix}.png"))
    plt.close()

versions = [
    ("_unfiltered", "labels_unfiltered.json", "Unfiltered"),
    ("_6class", "labels_6class.json", "6-class"),
    ("_5class", "labels_5class.json", "5-class"),
]
for suffix, mapfile, title in versions:
    for split in ["train", "val", "test"]:
        plot_dist(split, suffix, mapfile, f"{title} label distribution")
print("Saved EDA plots to results/eda/")
