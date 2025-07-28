# src/grand_challenge_gru.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
import numpy as np
import json
from model import GRUClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# ---- Paths and config ----
MODEL_PATH = 'models/gru_best_6class.pt'
LABEL_MAP_PATH = 'data/processed/labels_6class.json'
TEST_DATA_PATH = 'data/processed/test_signals_6class.npy'
TEST_LABELS_PATH = 'data/processed/test_labels_6class.npy'
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN_SIZE = 720  # adjust if not default

# ---- Load label map and class order ----
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
class_order = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
num_classes = len(class_order)

# ---- Load model ----
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model = GRUClassifier(num_classes=num_classes, win_size=WIN_SIZE)
model.load_state_dict(ckpt['model'])
model.to(DEVICE)
model.eval()
print(f"Loaded GRU model from {MODEL_PATH}")

# ---- Load test data ----
X = np.load(TEST_DATA_PATH)
Y_true = np.load(TEST_LABELS_PATH)
if X.ndim == 2:
    X = X[:, None, :]  # [N, 1, win_size]
X = torch.tensor(X, dtype=torch.float32)

# ---- Batched inference ----
all_preds = []
with torch.no_grad():
    for i in range(0, X.shape[0], BATCH_SIZE):
        xb = X[i:i+BATCH_SIZE].to(DEVICE)
        logits, _ = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
Y_pred = np.concatenate(all_preds)

# ---- Human-readable labels ----
inv_label_map = {v: k for k, v in label_map.items()}
Y_true_labels = [inv_label_map[y] for y in Y_true]
Y_pred_labels = [inv_label_map[y] for y in Y_pred]

# ---- Simulate "unknown" class (other) ----
other_idx = label_map['other']
is_other = (Y_true == other_idx)
other_pred_ids = Y_pred[is_other]
other_pred_labels = [inv_label_map[y] for y in other_pred_ids]

print("\nSimulating 'unknown class': other")
print(f"Total test windows of 'other': {is_other.sum()}")

from collections import Counter
counts = Counter(other_pred_labels)
print("Predicted class distribution for 'unknown':")
for k in class_order:
    n = counts.get(k, 0)
    pct = n / max(1, is_other.sum()) * 100
    print(f"  {k}: {n} ({pct:.1f}%)")

# Save as CSV
df_other = pd.DataFrame({
    'true': ['other']*len(other_pred_labels),
    'pred': other_pred_labels
})
csv_path = 'results/gru6class_grandchallenge_other_pred_dist.csv'
df_other['pred'].value_counts(normalize=True).sort_index().to_csv(csv_path)
print(f"\nSaved per-class prediction rates for 'other' to: {csv_path}")

# ---- Full classification report ----
report = classification_report(Y_true, Y_pred, target_names=class_order, digits=4)
print("\nFull classification report:\n", report)
# Save as text
with open('results/gru6class_grandchallenge_classification_report.txt', 'w') as f:
    f.write(report)
# Save as CSV
report_dict = classification_report(Y_true, Y_pred, target_names=class_order, digits=4, output_dict=True)
pd.DataFrame(report_dict).T.to_csv('results/gru6class_grandchallenge_classification_report.csv')

print("\nAll done! Metrics and per-class distributions saved to results/.")

