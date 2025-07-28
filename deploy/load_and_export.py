# deploy/load_and_export.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
import json
from model import ConvGRUClassifier

win_size = 720
model_path = 'models/convgru_best_6class.pt'
label_map_path = 'data/processed/labels_6class.json'

with open(label_map_path) as f:
    label_map = json.load(f)
class_order = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
print("Class order (label IDs):", class_order)

num_classes = len(class_order)
model = ConvGRUClassifier(num_classes=num_classes, win_size=win_size)
ckpt = torch.load(model_path, map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded from {model_path}")
