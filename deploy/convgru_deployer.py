# deploy/convgru_deployer.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch
import json
import numpy as np

class ConvGRUDeployer:
    def __init__(self, model_path, label_map_path, device='cpu'):
        with open(label_map_path) as f:
            label_map = json.load(f)
        self.class_order = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device).to(self.device)
        self.model.eval()

    def predict(self, windows):
        X = np.asarray(windows)
        if X.ndim == 2: X = X[:, None, :]
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        pred_labels = [self.class_order[idx] for idx in preds]
        return preds, pred_labels

    def predict_proba(self, windows):
        X = np.asarray(windows)
        if X.ndim == 2: X = X[:, None, :]
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def saliency(self, window):
        """
        Compute input-gradient saliency for a single window (1D np.array).
        Returns: saliency array (same shape as window)
        """
        X = torch.tensor(window[None, None, :], dtype=torch.float32, device=self.device, requires_grad=True)
        self.model.eval()
        logits, _ = self.model(X)
        pred = logits.argmax(dim=1)
        logit = logits[0, pred]
        self.model.zero_grad()
        logit.backward()
        sal = X.grad.detach().cpu().numpy()[0, 0]
        return sal

