"""
Simulate 'grand challenge' scenario using MIT-BIH test set and existing 6-class ConvGRU model.
Pretend one class (e.g., 'other') is 'unknown' to the model at first. 
Evaluate predictions on these windows, then (optionally) simulate adaptation.
"""
import numpy as np
import json
import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deploy')))
from convgru_deployer import ConvGRUDeployer

# --- Settings
unknown_class = 'other'  # simulate as unseen class
model_path = 'models/convgru_6class_deploy.ptc'
label_map_path = 'data/processed/labels_6class.json'
test_signals = 'data/processed/test_signals_6class.npy'
test_labels = 'data/processed/test_labels_6class.npy'

# --- Load data
with open(label_map_path) as f:
    label_map = json.load(f)
class_idx = {v: k for k, v in label_map.items()}
unknown_class_id = label_map[unknown_class]
X = np.load(test_signals)
y = np.load(test_labels)

# --- Run model
deployer = ConvGRUDeployer(model_path, label_map_path, device='cpu')
preds, pred_labels = deployer.predict(X)

# --- Analyze "unknown" class predictions
mask = (y == unknown_class_id)
unknown_true = y[mask]
unknown_pred = np.array(preds)[mask]
unknown_pred_labels = np.array(pred_labels)[mask]

print(f"\nSimulating 'unknown class': {unknown_class}")
print(f"Total test windows of '{unknown_class}': {len(unknown_true)}")
unique, counts = np.unique(unknown_pred_labels, return_counts=True)
print("Predicted class distribution for 'unknown':")
for c, n in zip(unique, counts):
    print(f"  {c}: {n} ({n/len(unknown_true)*100:.1f}%)")

# --- Discuss/visualize results
# Optionally, pick 5 examples, simulate "few-shot adaptation" (describe in report if not implemented)

print("\nIf adaptation/fine-tuning was implemented, the model could now add a new output for this class...")
