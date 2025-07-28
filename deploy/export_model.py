# deploy/export_model.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
import json
from model import ConvGRUClassifier

win_size = 720
model_path = 'models/convgru_best_6class.pt'
torchscript_path = 'models/convgru_6class_deploy.ptc'
onnx_path = 'models/convgru_6class.onnx'
label_map_path = 'data/processed/labels_6class.json'

with open(label_map_path) as f:
    label_map = json.load(f)
num_classes = len(label_map)
model = ConvGRUClassifier(num_classes=num_classes, win_size=win_size)
ckpt = torch.load(model_path, map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

dummy = torch.randn(1, 1, win_size)
traced = torch.jit.trace(model, dummy)
traced.save(torchscript_path)
print(f"TorchScript model saved as: {torchscript_path}")

torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['input'],
    output_names=['logits', 'hidden'],
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17
)
print(f"ONNX model saved as: {onnx_path}")
