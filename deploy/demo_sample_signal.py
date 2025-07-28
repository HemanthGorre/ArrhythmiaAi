import numpy as np
import json
import os

DATA_DIR = 'data/processed'
signals = np.load(os.path.join(DATA_DIR, 'test_signals_6class.npy'))
labels  = np.load(os.path.join(DATA_DIR, 'test_labels_6class.npy'))
label_map = json.load(open(os.path.join(DATA_DIR, 'labels_6class.json')))
inv_label_map = {v: k for k, v in label_map.items()}

class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]  # label order

for class_id, class_name in enumerate(class_names):
    idx = np.where(labels == class_id)[0]
    if len(idx) == 0:
        print(f"Warning: No test window found for {class_name}")
        continue
    window = signals[idx[0]]
    outpath = os.path.join(DATA_DIR, f'sample_{class_name}.csv')
    np.savetxt(outpath, window, delimiter=',')
    print(f"Saved {outpath} (class: {class_name}, id: {class_id})")
